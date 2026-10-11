from __future__ import annotations

import copy
import logging
import multiprocessing
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import cast

import polars as pl
from tqdm import tqdm

from limen.experiment._parallel_commit import ParallelCommit
from limen.experiment._parallel_resume import restore_parallel
from limen.experiment._parallel_worker import FrozenRoundSources, RoundPayload, evaluate_round as _evaluate_round, freeze_sources, startup_limits, validate_parallel, worker_init
from limen.experiment.checkpoint_manager import CheckpointManager
from limen.experiment.feedback_controller import FeedbackController
from limen.experiment.msq import MSQ
from limen.experiment.param_domain import ParamDomain
from limen.experiment.manifest_core import Manifest
from limen.experiment.param_search.search_strategy import SearchStrategy
from limen.experiment.reducer.pruning_strategy import PruningStrategy


logger = logging.getLogger(__name__)


def admit_parallel(host: object, workers: int, prep_each_round: object,
                   *, post_processing: bool, factorize: bool,
                   context_params: dict[str, object] | None) -> None:
    attributes = cast(dict[str, object], vars(host))
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    if workers == 1:
        return
    if not isinstance(prep_each_round, bool):
        raise TypeError('prep_each_round must be a bool')
    if not prep_each_round:
        raise ValueError('Parallel rounds require prep_each_round=True')
    reference, manifest, strategy = attributes['_compiled_yaml'], attributes['manifest'], attributes['_search_strategy']
    if reference is None or not isinstance(manifest, Manifest) or not isinstance(strategy, SearchStrategy):
        raise ValueError('Parallel rounds require compiled YAML and an MSQ search strategy')
    if post_processing or factorize or (attributes['prep'], attributes['model']) != attributes['_manifest_functions']:
        raise ValueError('Parallel rounds reject post_processing, factorize and overridden prep/model')
    validate_parallel(manifest, cast(dict[str, object], reference), strategy.domain.params, context_params)
    if attributes['_parallel_sources'] is None:
        attributes['_parallel_sources'] = freeze_sources(manifest, cast(pl.DataFrame, attributes['data']))
    attributes['_yaml_reference'] = copy.deepcopy(reference)


def _reap(pool: ProcessPoolExecutor) -> None:
    active = cast(dict[int, BaseProcess] | None, vars(pool)['_processes'])
    processes = tuple(active.values()) if active is not None else ()
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join()


def run_parallel(host: object, msq: MSQ, domain: ParamDomain,
                 feedback_controller: FeedbackController, checkpoint_manager: CheckpointManager,
                 *, content_hash: str, strategy_type: str, csv_path: Path,
                 round_data_path: Path | None, n_permutations: int,
                 context_params: dict[str, object] | None, resume: bool,
                 progress_bar: bool) -> None:
    attributes = cast(dict[str, object], vars(host))
    sources = cast(FrozenRoundSources | None, attributes['_parallel_sources'])
    reference = cast(dict[str, object] | None, attributes['_compiled_yaml'])
    strategy = cast(SearchStrategy | None, attributes['_search_strategy'])
    directory = cast(Path | None, attributes['_experiment_dir'])
    workers = cast(int, attributes['_workers'])
    checkpoint_call = cast(Callable[..., None], getattr(host, '_checkpoint', None))
    feedback_call = cast(Callable[[MSQ, SearchStrategy, FeedbackController, int], list[dict[str, object]]], getattr(host, '_trigger_feedback', None))
    if sources is None or reference is None or strategy is None:
        raise ValueError('Parallel rounds require admitted frozen inputs and compiled YAML')
    cast(Callable[[], None], getattr(host, '_register_shutdown_handler', None))()
    start = 0
    if resume:
        if round_data_path is None:
            raise ValueError('Parallel resume requires persisted round data')
        start = restore_parallel(host, msq, domain, feedback_controller, checkpoint_manager,
                                 content_hash=content_hash, strategy_type=strategy_type,
                                 csv_path=csv_path, round_data_path=round_data_path)
    elif directory is not None:
        cast(Callable[[], None], getattr(host, '_guard_stale_artifacts', None))()
        _ = cast(Callable[[Path, CheckpointManager], Path], getattr(host, '_initialize_fresh', None))(directory, checkpoint_manager)
        cast(Callable[[Path], None], getattr(host, '_write_metadata', None))(directory)
    writer = ParallelCommit(host, csv_path, round_data_path)
    committed = copy.deepcopy(msq.get_state())
    last = start - 1

    def checkpoint(*, rewind: bool = False) -> None:
        if directory is not None:
            if rewind:
                msq.set_state(committed)
            checkpoint_call(msq, domain, directory, checkpoint_manager,
                            last, n_permutations, strategy_type=strategy_type,
                            content_hash=content_hash, feedback_controller=feedback_controller,
                            pruning_strategies=cast(list[PruningStrategy], attributes['_pruning_strategies']))

    def barrier(index: int) -> bool:
        return feedback_controller.should_trigger(index) or checkpoint_manager.should_checkpoint(index)

    checkpoint()
    pending: deque[tuple[dict[str, object], Future[RoundPayload], dict[str, object]]] = deque()
    with startup_limits(), tqdm(initial=start, total=n_permutations,
                                disable=not progress_bar) as progress:
        pool = ProcessPoolExecutor(max_workers=workers,
                                   mp_context=multiprocessing.get_context('spawn'),
                                   initializer=worker_init,
                                   initargs=(reference, sources, cast(bool, attributes['_record_execution']), cast(bool, attributes['_record_model_outputs'])))
        exhausted = False
        try:
            while not attributes['_shutdown_requested']:
                while len(pending) < workers and not exhausted and not attributes['_shutdown_requested']:
                    if pending and barrier(cast(int, pending[-1][0]['_round_index'])):
                        break
                    try:
                        parameters = cast(dict[str, object], next(msq))
                    except StopIteration:
                        exhausted = True
                        break
                    task = parameters | (context_params or {})
                    pending.append((task, pool.submit(_evaluate_round, task),
                                    cast(dict[str, object], copy.deepcopy(msq.get_state()))))
                if not pending or attributes['_shutdown_requested']:
                    break
                parameters, future, drawn_state = pending[0]
                try:
                    payload = future.result(timeout=0.1)
                except FutureTimeoutError:
                    logger.debug('Waiting for parallel round %s', parameters['_round_index'])
                    continue
                except Exception as exc:
                    raise RuntimeError(f"Parallel worker failed for round {parameters['_round_index']} ({parameters['_id']}): {exc}") from exc
                if attributes['_shutdown_requested']:
                    break
                writer.commit(parameters, payload)
                last = cast(int, parameters['_round_index'])
                committed = drawn_state
                _ = pending.popleft()
                _ = progress.update(1)
                due = checkpoint_manager.should_checkpoint(last)
                if feedback_controller.should_trigger(last):
                    if pending:
                        raise RuntimeError('Parallel feedback crossed an uncommitted round')
                    interventions = feedback_call(msq, strategy, feedback_controller, last)
                    committed = cast(dict[str, object], copy.deepcopy(msq.get_state()))
                    due = due or bool(interventions)
                    exhausted = False
                if due:
                    if pending:
                        raise RuntimeError('Parallel checkpoint crossed an uncommitted round')
                    checkpoint()
        finally:
            try:
                checkpoint(rewind=msq.yielded_count != last + 1)
            finally:
                if pending or attributes['_shutdown_requested']:
                    for _, future, _ in pending:
                        _ = future.cancel()
                    _reap(pool)
                pool.shutdown(wait=True, cancel_futures=True)
    logger.info('Parallel experiment committed %d terminal rounds', last + 1)

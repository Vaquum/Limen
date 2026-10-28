from __future__ import annotations

from limen.experiment._exit_rules import admit_run, declaration

import copy
import json
import logging
import multiprocessing
import os
import signal
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from concurrent.futures.process import BrokenProcessPool
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import cast

import polars as pl
from tqdm import tqdm

from limen.experiment._dynamic_barriers import admit_research_run
from limen.experiment._holdout_composition import worker_authority
from limen.experiment._parallel_commit import ParallelCommit
from limen.experiment._parallel_resume import restore_parallel
from limen.experiment._parallel_sources import snapshot_sources
from limen.experiment._parallel_worker import FrozenRoundSources, RoundPayload, evaluate_round as _evaluate_round, freeze_sources, startup_limits, validate_parallel, worker_init
from limen.experiment.checkpoint_manager import CheckpointManager
from limen.experiment.feedback_controller import FeedbackController
from limen.experiment.msq import MSQ
from limen.experiment.param_domain import ParamDomain
from limen.experiment.manifest_core import Manifest
from limen.experiment.param_search.search_strategy import SearchStrategy
from limen.experiment.reducer.pruning_strategy import PruningStrategy


logger = logging.getLogger(__name__)


def configuration(sfd: object, yaml_reference: dict[str, object] | None) -> tuple[dict[str, object] | None, dict[str, object] | None, int, FrozenRoundSources | None]:
    from limen.yaml.compiler import CompiledSFD

    compiled = cast(dict[str, object], copy.deepcopy(vars(sfd)['_yaml'])) if type(sfd) is CompiledSFD else None
    return copy.deepcopy(yaml_reference), compiled, 1, None


def admit_parallel(host: object, workers: int, prep_each_round: bool,
                   *, post_processing: bool = False, factorize: bool = False,
                   context_params: dict[str, object] | None = None) -> None:
    attributes = cast(dict[str, object], vars(host))
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    if workers == 1:
        return
    if type(prep_each_round) is not bool:
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


_RoundFunction = Callable[..., dict[str, object]]


def begin_run(host: object, workers: int | None, prep_each_round: bool, resume: bool,
              post_processing: bool, factorize: bool, context_params: dict[str, object] | None,
              overrides: tuple[object, object, object]) -> tuple[_RoundFunction | None, _RoundFunction | None]:
    attributes = cast(dict[str, object], vars(host))
    directory = cast(Path | None, attributes['_experiment_dir'])
    path = directory / 'metadata.json' if directory is not None else None
    metadata: dict[str, object] = json.loads(path.read_text()) if resume and path is not None and path.exists() else {}
    admit_run(attributes['manifest'], metadata, resume=resume, factorize=factorize)
    effective = cast(int, metadata.get('workers', 1)) if workers is None else workers
    if resume and (effective != metadata.get('workers', 1) or metadata.get('worker_native_threads', 1) != 1):
        raise ValueError('Cannot resume with a different worker or native-thread policy')
    if type(effective) is int and effective > 1 and any(value is not None for value in overrides):
        raise ValueError('Parallel rounds reject parameter, preparation and model overrides')
    admit_research_run(attributes['manifest'], effective, resume)
    admit_parallel(host, effective, prep_each_round, post_processing=post_processing,
                   factorize=factorize, context_params=context_params)
    attributes.update(_workers=effective, _shutdown_requested=False)
    return cast(_RoundFunction | None, attributes['prep']), cast(_RoundFunction | None, attributes['model'])


def worker_metadata(workers: int, metadata: dict[str, object], *, context_parameter_keys: list[str] | None = None, manifest: object = None) -> dict[str, object]:
    metadata['exit_rules_declaration'] = declaration(manifest)
    if workers > 1:
        metadata.update(workers=workers, worker_native_threads=1, context_parameter_keys=context_parameter_keys or [])
    return metadata


def dispatch(host: object, components: dict[str, object], n_permutations: int,
             context_params: dict[str, object] | None, resume: bool, progress_bar: bool) -> bool:
    if cast(dict[str, object], vars(host))['_workers'] == 1:
        return False
    run_parallel(host, cast(MSQ, components['msq']), cast(ParamDomain, components['domain']),
                 cast(FeedbackController, components['feedback_controller']), cast(CheckpointManager, components['checkpoint_manager']),
                 content_hash=cast(str, components['content_hash']), strategy_type=cast(str, components['strategy_type']),
                 csv_path=cast(Path, components['csv_path']), round_data_path=cast(Path | None, components['round_data_path']),
                 n_permutations=n_permutations, context_params=context_params, resume=resume, progress_bar=progress_bar)
    return True


def _reap(pool: ProcessPoolExecutor) -> None:
    cast(Callable[[], None], getattr(vars(pool)['_call_queue'], 'cancel_join_thread', None))()
    # Let an interrupted result read reach EOF after the child writers die.
    cast(Connection, vars(vars(pool)['_result_queue'])['_writer']).close()
    active = cast(dict[int, BaseProcess] | None, vars(pool)['_processes'])
    processes = tuple(active.values()) if active is not None else ()
    for process in processes:
        if process.is_alive():
            process.kill()
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
    failed_round_params: list[dict[str, object]] = []
    if resume:
        if round_data_path is None:
            raise ValueError('Parallel resume requires persisted round data')
        start = restore_parallel(host, msq, domain, feedback_controller, checkpoint_manager,
                                 content_hash=content_hash, strategy_type=strategy_type,
                                 csv_path=csv_path, round_data_path=round_data_path,
                                 failed_round_params=failed_round_params, n_permutations=n_permutations, context_params=context_params)
    elif directory is not None:
        cast(Callable[[], None], getattr(host, '_guard_stale_artifacts', None))()
        _ = cast(Callable[[Path, CheckpointManager], Path], getattr(host, '_initialize_fresh', None))(directory, checkpoint_manager)
        cast(Callable[..., None], getattr(host, '_write_metadata', None))(directory, context_parameter_keys=sorted(key for key in (context_params or {}) if not key.startswith('_')))
    writer = ParallelCommit(host, csv_path, round_data_path, failed_round_params)
    committed = copy.deepcopy(msq.get_state())
    last = start - 1

    def checkpoint(*, rewind: bool = False) -> None:
        if directory is not None:
            checkpoint_call(committed if rewind else msq, domain, directory, checkpoint_manager,
                            last, n_permutations, strategy_type=strategy_type,
                            content_hash=content_hash, feedback_controller=feedback_controller,
                            pruning_strategies=cast(list[PruningStrategy], attributes['_pruning_strategies']),
                            failed_round_params=writer.failed_round_params)

    def barrier(index: int) -> bool:
        return feedback_controller.should_trigger(index) or checkpoint_manager.should_checkpoint(index)

    checkpoint()
    pending: deque[tuple[dict[str, object], Future[RoundPayload], dict[str, object]]] = deque()
    capacity = max(1, min(workers, n_permutations - start))
    capacity = next((index + 1 for index in range(1, capacity) if barrier(index)), capacity)
    with startup_limits(), snapshot_sources(sources.learning, sources.execution, sources.funding, csv_path.parent) as snapshot, tqdm(initial=start, total=n_permutations,
                                disable=not progress_bar) as progress:
        pool = ProcessPoolExecutor(max_workers=capacity,
                                   mp_context=multiprocessing.get_context('spawn'),
                                   initializer=worker_init,
                                   initargs=(reference, snapshot, cast(bool, attributes['_record_execution']), cast(bool, attributes['_record_model_outputs']), getattr(attributes.get('_walk_forward'), 'promotion', None), worker_authority(attributes['manifest'])))
        exhausted = False
        try:
            launch = cast(Callable[[], None], getattr(pool, '_launch_processes', None))
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
                    if not vars(pool)['_processes']:
                        if os.name == 'posix':
                            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, (signal.SIGINT, signal.SIGTERM))
                            try:
                                launch()
                            finally:
                                _ = signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                        else:
                            launch()
                    try:
                        future = pool.submit(_evaluate_round, task)
                    except BrokenProcessPool as exc:
                        if attributes['_shutdown_requested']:
                            break
                        failed = pending[0][0] if pending else task
                        raise RuntimeError(f"Parallel worker process or initializer failed while awaiting round {failed['_round_index']} ({failed['_id']}): {exc}") from exc
                    pending.append((task, future,
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
                    if attributes['_shutdown_requested']:
                        break
                    failure = 'Parallel worker process or initializer failed while awaiting' if isinstance(exc, BrokenProcessPool) else 'Parallel worker failed for'
                    raise RuntimeError(f"{failure} round {parameters['_round_index']} ({parameters['_id']}): {exc}") from exc
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
                if pending or attributes['_shutdown_requested'] or vars(pool)['_executor_manager_thread'] is None:
                    for _, future, _ in pending:
                        _ = future.cancel()
                    _reap(pool)
                pool.shutdown(wait=True, cancel_futures=True)
    logger.info('Parallel experiment committed %d terminal rounds', last + 1)

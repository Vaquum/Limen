from __future__ import annotations

import copy
import logging
import multiprocessing
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor, TimeoutError
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import TYPE_CHECKING, cast

from tqdm import tqdm

from limen.experiment._parallel_commit import ParallelCommit
from limen.experiment._parallel_resume import restore_parallel
from limen.experiment._parallel_worker import RoundPayload, evaluate_round as _evaluate_round, freeze_sources, startup_limits, validate_parallel, worker_init
from limen.experiment.checkpoint_manager import CheckpointManager
from limen.experiment.feedback_controller import FeedbackController
from limen.experiment.msq import MSQ
from limen.experiment.param_domain import ParamDomain
from limen.experiment.manifest_core import Manifest

if TYPE_CHECKING:
    from limen.experiment.experiment_core import UniversalExperimentLoop

logger = logging.getLogger(__name__)


def admit_parallel(host: UniversalExperimentLoop, workers: int, prep_each_round: bool,
                   *, post_processing: bool, factorize: bool,
                   context_params: dict[str, object] | None) -> None:
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    if workers == 1:
        return
    if not isinstance(prep_each_round, bool):
        raise TypeError('prep_each_round must be a bool')
    if not prep_each_round:
        raise ValueError('Parallel rounds require prep_each_round=True')
    if host._compiled_yaml is None or not isinstance(host.manifest, Manifest) or host._search_strategy is None:
        raise ValueError('Parallel rounds require compiled YAML and an MSQ search strategy')
    if post_processing or factorize or (host.prep, host.model) != host._manifest_functions:
        raise ValueError('Parallel rounds reject post_processing, factorize and overridden prep/model')
    validate_parallel(host.manifest, host._compiled_yaml, host._search_strategy.domain.params, context_params)
    if host._parallel_sources is None:
        host._parallel_sources = freeze_sources(host.manifest, host.data)
    host._yaml_reference = copy.deepcopy(host._compiled_yaml)


def _reap(pool: ProcessPoolExecutor) -> None:
    active = cast(dict[int, BaseProcess] | None, getattr(pool, '_processes'))
    processes = tuple(active.values()) if active is not None else ()
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join()


def run_parallel(host: UniversalExperimentLoop, msq: MSQ, domain: ParamDomain,
                 feedback_controller: FeedbackController, checkpoint_manager: CheckpointManager,
                 *, content_hash: str, strategy_type: str, csv_path: Path,
                 round_data_path: Path | None, n_permutations: int,
                 context_params: dict[str, object] | None, resume: bool,
                 progress_bar: bool) -> None:
    sources, reference = host._parallel_sources, host._compiled_yaml
    if sources is None or reference is None or host._search_strategy is None:
        raise ValueError('Parallel rounds require admitted frozen inputs and compiled YAML')
    host._register_shutdown_handler()
    start = 0
    if resume:
        if round_data_path is None:
            raise ValueError('Parallel resume requires persisted round data')
        start = restore_parallel(host, msq, domain, feedback_controller, checkpoint_manager,
                                 content_hash=content_hash, strategy_type=strategy_type,
                                 csv_path=csv_path, round_data_path=round_data_path)
    elif host._experiment_dir is not None:
        host._guard_stale_artifacts()
        _ = host._initialize_fresh(host._experiment_dir, checkpoint_manager)
        host._write_metadata(host._experiment_dir)
    writer = ParallelCommit(host, csv_path, round_data_path)
    committed = copy.deepcopy(msq.get_state())
    last = start - 1

    def checkpoint() -> None:
        if host._experiment_dir is not None:
            msq.set_state(committed)
            host._checkpoint(msq, domain, host._experiment_dir, checkpoint_manager,
                             last, n_permutations, strategy_type=strategy_type,
                             content_hash=content_hash, feedback_controller=feedback_controller,
                             pruning_strategies=host._pruning_strategies)

    def barrier(index: int) -> bool:
        return feedback_controller.should_trigger(index) or checkpoint_manager.should_checkpoint(index)

    checkpoint()
    pending: deque[tuple[dict[str, object], Future[RoundPayload], dict[str, object]]] = deque()
    with startup_limits(), tqdm(initial=start, total=n_permutations,
                                disable=not progress_bar) as progress:
        pool = ProcessPoolExecutor(max_workers=host._workers,
                                   mp_context=multiprocessing.get_context('spawn'),
                                   initializer=worker_init,
                                   initargs=(reference, sources, host._record_execution, host._record_model_outputs))
        exhausted = False
        try:
            while not host._shutdown_requested:
                while len(pending) < host._workers and not exhausted and not host._shutdown_requested:
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
                if not pending or host._shutdown_requested:
                    break
                parameters, future, drawn_state = pending[0]
                try:
                    payload = future.result(timeout=0.1)
                except TimeoutError:
                    continue
                except Exception as exc:
                    raise RuntimeError(f"Parallel worker failed for round {parameters['_round_index']} ({parameters['_id']}): {exc}") from exc
                if host._shutdown_requested:
                    break
                writer.commit(parameters, payload)
                last = cast(int, parameters['_round_index'])
                committed = drawn_state
                _ = pending.popleft()
                progress.update(1)
                due = checkpoint_manager.should_checkpoint(last)
                if feedback_controller.should_trigger(last):
                    if pending:
                        raise RuntimeError('Parallel feedback crossed an uncommitted round')
                    interventions = host._trigger_feedback(msq, host._search_strategy,
                                                          feedback_controller, last)
                    committed = cast(dict[str, object], copy.deepcopy(msq.get_state()))
                    due = due or bool(interventions)
                    exhausted = False
                if due:
                    if pending:
                        raise RuntimeError('Parallel checkpoint crossed an uncommitted round')
                    checkpoint()
        finally:
            checkpoint()
            if pending or host._shutdown_requested:
                for _, future, _ in pending:
                    _ = future.cancel()
                _reap(pool)
            pool.shutdown(wait=True, cancel_futures=True)
    logger.info('Parallel experiment committed %d terminal rounds', last + 1)

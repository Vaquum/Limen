from __future__ import annotations

import os
import copy
import signal
import time
import traceback
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import polars as pl

from limen.experiment._parallel_sources import SourceSnapshot
from limen.experiment._parallel_plan import FrozenManifestPlan, freeze_manifest, plan_identity
from limen.experiment._parallel_encoding import digest
from limen.experiment._parallel_evidence import round_record
from limen.experiment._parallel_resources import backend_imports, validate_resources, validate_domain
from limen.experiment.convergence_report import convergence_warning
from limen.experiment.errors import StrictModeError
from limen.experiment.manifest_core import DataSourceConfig, DataSourceResolver, Manifest, MLManifest
from limen.experiment._walk_forward_run import WalkForwardRun
from limen.sfd.reference_architecture.lightgbm_binary import lightgbm_binary
from limen.sfd.reference_architecture.logreg_binary import logreg_binary

__all__ = ['FrozenRoundSources', 'RoundPayload', 'evaluate_round', 'freeze_sources',
           'startup_limits', 'validate_parallel', 'worker_init']

_THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'POLARS_MAX_THREADS')


@dataclass(frozen=True)
class FrozenRoundSources:
    learning: pl.DataFrame
    execution: pl.DataFrame | None
    funding: pl.DataFrame | None


@dataclass(frozen=True)
class RoundPayload:
    results: dict[str, object]
    parameters: dict[str, object]
    alignment: dict[str, object]
    warnings: list[str]
    convergence: bool | None
    succeeded: bool
    elapsed: float
    pid: int
    started_ns: int
    ended_ns: int
    native_pools: list[dict[str, object]]


class _ThreadLimits(Protocol):
    def __enter__(self) -> object: ...
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> object: ...


def validate_parallel(manifest: Manifest, yaml_reference: Mapping[str, object],
                      domain: Mapping[str, list[object]], context: Mapping[str, object] | None) -> FrozenManifestPlan:
    if type(manifest) is not MLManifest or manifest.architecture_function not in (logreg_binary, lightgbm_binary):
        raise ValueError('Parallel rounds require compiled CPU logreg_binary or lightgbm_binary')
    effective = dict(domain) | {key: [value] for key, value in (context or {}).items()}
    devices = effective.get('device_type', effective.get('device', ['cpu']))
    if any(value != 'cpu' for value in devices):
        raise ValueError('Parallel rounds require the CPU backend')
    plan = freeze_manifest(manifest, yaml_reference)
    validate_domain(plan, domain, context)
    return plan


def freeze_sources(manifest: Manifest, learning: pl.DataFrame) -> FrozenRoundSources:
    config = manifest.backtest_config
    execution = config.execution_data_source if config is not None else None
    funding = config.funding.data_source if config is not None and config.funding is not None else None
    frames: list[pl.DataFrame | None] = []
    for source in (execution, funding):
        if source is None:
            frames.append(None)
        else:
            if not isinstance(source, DataSourceConfig):
                raise ValueError('Parallel rounds require reconstructable declared sources')
            frame = DataSourceResolver.resolve(source)
            frames.append(frame.clone())
    return FrozenRoundSources(learning.clone(), frames[0], frames[1])


@contextmanager
def startup_limits() -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in _THREAD_ENV}
    try:
        os.environ.update(dict.fromkeys(_THREAD_ENV, '1'))
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                _ = os.environ.pop(key, None)
            else:
                os.environ[key] = value


_worker_plan: FrozenManifestPlan | None = None
_worker_sources: FrozenRoundSources | None = None
_worker_record_execution = False
_worker_record_model_outputs = False
_worker_fold_promotion: dict[str, object] | None = None
_worker_identity: dict[str, object] = {}


def _native_pools() -> list[dict[str, object]]:
    info = cast(Callable[[], list[dict[str, object]]], vars(import_module('threadpoolctl'))['threadpool_info'])
    pools = info()
    if pl.thread_pool_size() != 1 or any(pool.get('num_threads') != 1 for pool in pools):
        raise ValueError('Parallel worker native thread pools must all be limited to one')
    return pools


def worker_init(plan: FrozenManifestPlan, sources: SourceSnapshot,
                record_execution: bool, record_model_outputs: bool, fold_promotion: dict[str, object] | None = None) -> None:
    global _worker_plan, _worker_sources, _worker_record_execution, _worker_record_model_outputs, _worker_fold_promotion
    _ = signal.signal(signal.SIGINT, signal.SIG_IGN)
    _ = signal.signal(signal.SIGTERM, signal.SIG_DFL)
    if os.name == 'posix':
        os.setpgrp()
        _ = signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGINT, signal.SIGTERM))
    if any(os.environ.get(key) != '1' for key in _THREAD_ENV):
        raise ValueError('Parallel worker startup limits were not applied before imports')
    if digest(plan.manifest) != plan.pipeline_digest:
        raise ValueError('Parallel worker pipeline import identity changed')
    for name in backend_imports(plan):
        _ = import_module(name)
    limit = cast(Callable[..., _ThreadLimits], vars(import_module('threadpoolctl'))['threadpool_limits'])
    with limit(limits=1):
        _ = _native_pools()
    restored = FrozenRoundSources(sources.learning.read(),
                                        sources.execution.read() if sources.execution is not None else None,
                                        sources.funding.read() if sources.funding is not None else None)
    _worker_plan, _worker_sources = plan, restored
    _worker_identity.clear()
    _worker_identity.update(plan_identity(plan, (restored.learning, restored.execution, restored.funding)))
    _worker_record_execution, _worker_record_model_outputs, _worker_fold_promotion = record_execution, record_model_outputs, fold_promotion


def evaluate_round(round_params: dict[str, object]) -> RoundPayload:
    if _worker_plan is None or _worker_sources is None:
        raise RuntimeError('Parallel worker initializer has not completed')
    started = time.perf_counter_ns()
    params = {key: value for key, value in round_params.items() if not key.startswith('_')}
    alignment: dict[str, object] = {}
    results: dict[str, object] = {}
    succeeded = False
    caught: list[warnings.WarningMessage] = []
    limit = cast(Callable[..., _ThreadLimits], vars(import_module('threadpoolctl'))['threadpool_limits'])
    try:
        with limit(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            manifest = copy.deepcopy(_worker_plan.manifest)
            validate_resources(_worker_plan, params)
            _ = _native_pools()
            vars(manifest).update(_frozen_trade_sources=(_worker_sources.execution, _worker_sources.funding), _parallel_evidence=True)
            folds = None
            if manifest.split_walk_forward is None:
                data = cast(dict[str, object], manifest.prepare_data(_worker_sources.learning, params))
            else:
                folds = WalkForwardRun(manifest, _worker_sources.learning, Path('.'), promotion=_worker_fold_promotion or False)
                data = folds.prepare(_worker_sources.learning, params)
            data['_record_execution'] = _worker_record_execution
            data['_record_model_outputs'] = _worker_record_model_outputs
            count = 0
            if folds is None:
                test = data['x_test'] if isinstance(manifest, MLManifest) else data['test']
                if not isinstance(test, pl.DataFrame):
                    raise ValueError('Parallel round requires independently prepared test rows')
                count = test.height
            if folds is None:
                results = cast(dict[str, object], manifest.run_model(data, params))
            else:
                results = folds.evaluate(data, params)
            raw_alignment = data.get('_alignment')
            if not isinstance(raw_alignment, dict):
                raise ValueError('Parallel round requires recorded alignment evidence')
            alignment = {key: value for key, value in cast(dict[str, object], raw_alignment).items()
                         if not key.startswith('_')}
            contract = next(stage.contract for stage in _worker_plan.stages if stage.path == 'architecture')
            alignment['parallel_record'] = round_record(round_params, params, results, alignment, _worker_identity, contract,
                count=count, execution=_worker_record_execution, outputs=_worker_record_model_outputs)
            for key in ('_model', 'models', 'extras'):
                _ = results.pop(key, None)
            pools = _native_pools()
            succeeded = True
    except StrictModeError as exc:
        results = {'strict_mode_error': str(exc)}
        pools = _native_pools()
    except Exception as exc:
        raise RuntimeError(f"Parallel round {round_params.get('_round_index')} ({round_params.get('_id')}) failed:\n{traceback.format_exc()}") from exc
    ended = time.perf_counter_ns()
    return RoundPayload(results, params, alignment, list(dict.fromkeys(str(item.message) for item in caught)),
                        convergence_warning(caught, succeeded), succeeded, (ended - started) / 1e9,
                        os.getpid(), started, ended, pools)

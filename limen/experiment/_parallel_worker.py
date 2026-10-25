from __future__ import annotations

import os
import copy
import signal
import time
import traceback
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, is_dataclass
from importlib import import_module
from inspect import isfunction, signature
from pathlib import Path
from string import Formatter
from typing import Protocol, cast

import polars as pl

from limen.experiment._parallel_sources import SourceSnapshot
import limen.experiment._holdout_composition as _protection
from limen.experiment.convergence_report import convergence_warning
from limen.experiment.errors import StrictModeError
from limen.experiment.manifest_core import DataSourceConfig, DataSourceResolver, Manifest
from limen.experiment._walk_forward_run import WalkForwardRun
from limen.sfd.reference_architecture.lightgbm_binary import lightgbm_binary

__all__ = ['FrozenRoundSources', 'RoundPayload', 'evaluate_round', 'freeze_sources',
           'startup_limits', 'validate_parallel', 'worker_init']

_THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'POLARS_MAX_THREADS')
_SCALER_FACTORIES = ('make_fitted_scaler.<locals>._factory',
                     'MLManifest.set_scaler_from_params.<locals>._scaler_factory')
_CPU_KNOBS = ('n_jobs', 'device', 'device_type')


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


def _shape(value: object) -> object:
    if isfunction(value):
        function = value
        if function.__module__ == 'limen.experiment.manifest_core' and function.__qualname__ in _SCALER_FACTORIES:
            return (function.__code__, _shape(function.__defaults__),
                    _shape(tuple(cell.cell_contents for cell in function.__closure__ or ())))
    if callable(value):
        module, name = getattr(value, '__module__', ''), getattr(value, '__qualname__', '')
        if not isinstance(module, str) or not module.startswith('limen.') or not isinstance(name, str) or '<' in name:
            raise ValueError('Parallel rounds require reconstructable built-in pipeline callables')
        resolved: object = import_module(module)
        for part in name.split('.'):
            resolved = getattr(resolved, part)
        if resolved is not value:
            raise ValueError('Parallel rounds require unchanged importable pipeline callables')
        return value
    if isinstance(value, Mapping):
        return {key: _shape(item) for key, item in cast(Mapping[object, object], value).items()}
    if isinstance(value, (tuple, list)):
        return tuple(_shape(item) for item in cast(tuple[object, ...] | list[object], value))
    if is_dataclass(value) and not isinstance(value, type):
        return _shape(vars(value))
    return value


def _static_source(value: object, domain: Mapping[str, list[object]]) -> None:
    if callable(value):
        raise ValueError('Parallel rounds require static source parameters')
    if isinstance(value, str):
        if value in domain or any(field is not None for _, field, _, _ in Formatter().parse(value)):
            raise ValueError('Parallel rounds reject parameter-dependent source declarations')
    elif isinstance(value, Mapping):
        for item in cast(Mapping[object, object], value).values():
            _static_source(item, domain)
    elif isinstance(value, (tuple, list)):
        for item in cast(tuple[object, ...] | list[object], value):
            _static_source(item, domain)


def _require_cpu(knobs: Mapping[str, list[object]]) -> None:
    if any(type(value) is not int or value != 1 for value in knobs.get('n_jobs', [])):
        raise ValueError('Parallel rounds require effective n_jobs=1 for every admitted round')
    if any(value not in (None, 'cpu') for name in ('device', 'device_type') for value in knobs.get(name, [])):
        raise ValueError('Parallel rounds require the CPU backend')


def validate_parallel(manifest: Manifest, yaml_reference: Mapping[str, object],
                      domain: Mapping[str, list[object]], context: Mapping[str, object] | None) -> None:
    from limen.yaml.compiler import build_manifest

    rebuilt = build_manifest(dict(yaml_reference))
    if _shape(vars(manifest)) != _shape(vars(rebuilt)):
        raise ValueError('Parallel rounds require the unchanged compiled YAML manifest pipeline')
    effective = {key: list(values) for key, values in domain.items()}
    effective.update({key: [value] for key, value in (context or {}).items()})
    if any(not values for values in effective.values()):
        raise ValueError('Parallel rounds require nonempty parameter domains')
    _ = _shape(effective)
    parameters = signature(cast(Callable[..., object], manifest.architecture_function)).parameters
    _require_cpu({name: effective[name] if name in effective else [parameters[name].default] for name in _CPU_KNOBS if name in effective or name in parameters})
    sources: list[object] = [manifest.data_source_config]
    config = manifest.backtest_config
    if config is not None:
        sources.extend((config.execution_data_source, config.funding.data_source if config.funding is not None else None))
    for source in sources:
        if source is not None:
            if not isinstance(source, DataSourceConfig):
                raise ValueError('Parallel rounds require reconstructable declared sources')
            _static_source(source.params, effective)


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


_worker_yaml: dict[str, object] | None = None
_worker_sources: FrozenRoundSources | None = None
_worker_record_execution = False
_worker_record_model_outputs = False
_worker_fold_promotion: dict[str, object] | None = None
_worker_authority: _protection.WorkerAuthority | None = None


def _native_pools() -> list[dict[str, object]]:
    info = cast(Callable[[], list[dict[str, object]]], vars(import_module('threadpoolctl'))['threadpool_info'])
    pools = info()
    if pl.thread_pool_size() != 1 or any(pool.get('num_threads') != 1 for pool in pools):
        raise ValueError('Parallel worker native thread pools must all be limited to one')
    return pools


def worker_init(yaml_reference: dict[str, object], sources: SourceSnapshot,
                record_execution: bool, record_model_outputs: bool, fold_promotion: dict[str, object] | None = None,
                authority: _protection.WorkerAuthority | None = None) -> None:
    from limen.yaml.compiler import build_manifest

    global _worker_yaml, _worker_sources, _worker_record_execution, _worker_record_model_outputs, _worker_fold_promotion, _worker_authority
    _ = signal.signal(signal.SIGINT, signal.SIG_IGN)
    _ = signal.signal(signal.SIGTERM, signal.SIG_DFL)
    if os.name == 'posix':
        os.setpgrp()
        _ = signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGINT, signal.SIGTERM))
    if any(os.environ.get(key) != '1' for key in _THREAD_ENV):
        raise ValueError('Parallel worker startup limits were not applied before imports')
    manifest = build_manifest(yaml_reference)
    if manifest.architecture_function is lightgbm_binary:
        _ = import_module('lightgbm')
    _ = _native_pools()
    restored = FrozenRoundSources(sources.learning.read(),
                                        sources.execution.read() if sources.execution is not None else None,
                                        sources.funding.read() if sources.funding is not None else None)
    _worker_yaml, _worker_sources = yaml_reference, restored
    _worker_record_execution, _worker_record_model_outputs, _worker_fold_promotion, _worker_authority = record_execution, record_model_outputs, fold_promotion, authority


def evaluate_round(round_params: dict[str, object]) -> RoundPayload:
    from limen.yaml.compiler import build_manifest

    if _worker_yaml is None or _worker_sources is None:
        raise RuntimeError('Parallel worker initializer has not completed')
    started = time.perf_counter_ns()
    params = {key: value for key, value in round_params.items() if not key.startswith('_')}
    alignment: dict[str, object] = {}
    results: dict[str, object] = {}
    succeeded = False
    caught: list[warnings.WarningMessage] = []
    protection: dict[str, object] | None = None
    limit = cast(Callable[..., _ThreadLimits], vars(import_module('threadpoolctl'))['threadpool_limits'])
    try:
        with limit(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            manifest = build_manifest(_worker_yaml)
            with _protection.worker_operation(manifest, _worker_sources.learning, _worker_authority, round_params.get('_id')) as protection:
                _require_cpu({name: [value] for name, value in manifest.resolve_model_kwargs(params).items() if name in _CPU_KNOBS})
                vars(manifest)['_frozen_trade_sources'] = (_worker_sources.execution, _worker_sources.funding)
                folds = None
                if manifest.split_walk_forward is None:
                    data = cast(dict[str, object], manifest.prepare_data(_worker_sources.learning, params))
                else:
                    folds = WalkForwardRun(manifest, _worker_sources.learning, Path('.'), promotion=_worker_fold_promotion or False)
                    data = folds.prepare(_worker_sources.learning, params)
                data['_record_execution'] = _worker_record_execution
                data['_record_model_outputs'] = _worker_record_model_outputs
                if folds is None:
                    results = cast(dict[str, object], manifest.run_model(data, params))
                else:
                    results = folds.evaluate(data, params)
                raw_alignment = data.get('_alignment')
                if not isinstance(raw_alignment, dict):
                    raise ValueError('Parallel round requires recorded alignment evidence')
                alignment = {key: value for key, value in cast(dict[str, object], raw_alignment).items()
                             if not key.startswith('_')}
                for key in ('_model', 'models', 'extras'):
                    _ = results.pop(key, None)
                pools = _native_pools()
                succeeded = True
    except StrictModeError as exc:
        results = {'strict_mode_error': str(exc)}
        alignment = {} if protection is None else {'holdout_failure': copy.deepcopy(protection)}
        pools = _native_pools()
    except Exception as exc:
        raise RuntimeError(f"Parallel round {round_params.get('_round_index')} ({round_params.get('_id')}) failed:\n{traceback.format_exc()}") from exc
    ended = time.perf_counter_ns()
    return RoundPayload(results, params, alignment, list(dict.fromkeys(str(item.message) for item in caught)),
                        convergence_warning(caught, succeeded), succeeded, (ended - started) / 1e9,
                        os.getpid(), started, ended, pools)

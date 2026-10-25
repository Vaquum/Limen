from __future__ import annotations

import copy
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import polars as pl

from limen.experiment._holdout_history import ProjectBinding
from limen.experiment._holdout_source import source_spec, source_workflow_active
from limen.experiment._prepare_trade_context import FrozenTradeSources, load_source, normalize_observations, source_interval
from limen.experiment._resolve_trade_policy import BacktestConfig, SourceConfig, resolve_json, resolve_trade_policy

__all__: list[str] = []


class _Operation(Protocol):
    manifest: object
    binding: ProjectBinding
    provenance: dict[str, object]
    replay_witness: Mapping[str, object] | None


class _Research(Protocol):
    def operation(self, manifest: object, role: str) -> AbstractContextManager[_Operation | None]: ...


@contextmanager
def source_operation(manifest: object, role: str = 'experiment') -> Iterator[None]:
    token = source_workflow_active.set(True)
    try:
        with cast(_Research, import_module('limen.experiment._holdout_research')).operation(manifest, role):
            yield
    finally:
        source_workflow_active.reset(token)


def _frame(frame: pl.DataFrame) -> dict[str, object]:
    buffer = frame.write_ipc(None, compression='uncompressed')
    return {'rows': frame.height, 'schema': {name: str(dtype) for name, dtype in frame.schema.items()},
            'sha256': sha256(buffer.getvalue()).hexdigest()}


@dataclass(frozen=True)
class _Config:
    method: Callable[..., object]
    params: Mapping[str, object]


@dataclass(frozen=True)
class _Source:
    data_source_config: _Config


def _identity(config: SourceConfig, params: Mapping[str, object], binding: ProjectBinding) -> dict[str, object]:
    source = _Source(_Config(config.method, cast(Mapping[str, object], resolve_json(config.params, params))))
    interval = source_interval(source.data_source_config)
    kind, identity, _ = source_spec(source, replace(binding, interval_seconds=interval) if type(interval) is int else binding)
    return {'kind': kind, 'method': f'{config.method.__module__}.{config.method.__qualname__}', 'declaration': identity,
            'content_sha256': sha256(Path(cast(str, identity['file_path_or_url'])).read_bytes()).hexdigest() if kind == 'local_file' else None}


def _ns(frame: pl.DataFrame, name: str) -> pl.Series:
    if name not in frame.columns:
        raise ValueError(f'Protected source lacks actual {name} support')
    values = frame[name]
    if isinstance(values.dtype, pl.Datetime):
        values = values.dt.epoch('ns')
    if values.dtype != pl.Int64 or values.null_count():
        raise ValueError(f'Protected source requires nonnull integer nanoseconds for {name}')
    return values


def _roles(config: BacktestConfig | None, params: Mapping[str, object], binding: ProjectBinding,
           frozen: FrozenTradeSources | None) -> tuple[tuple[pl.DataFrame | None, pl.DataFrame | None], dict[str, object]]:
    frames: list[pl.DataFrame | None] = [None, None]
    witnesses: dict[str, object] = {}
    if config is None:
        return (None, None), witnesses
    if callable(frozen):
        frozen = frozen(config, params)
    policy = resolve_trade_policy(config, params)
    sources = (config.execution_data_source, None if config.funding is None else config.funding.data_source)
    for index, (role, declared) in enumerate(zip(('execution', 'funding'), sources, strict=True)):
        if declared is None:
            continue
        identity = _identity(declared, params, binding)
        materialized = load_source(declared, params)
        if frozen is not None and (not isinstance(frozen[index], pl.DataFrame) or not materialized.equals(cast(pl.DataFrame, frozen[index]))):
            raise ValueError('Protected frozen source differs from its current declared content')
        if role == 'execution':
            materialized = normalize_observations(materialized, interval_seconds=resolve_json(source_interval(declared), params))
            times = {name: _ns(materialized, name) for name in ('start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns')}
            if ((times['end_ns'] < times['start_ns']) | (times['available_at_ns'] < times['end_ns'])
                    | (times['open_available_at_ns'] < times['start_ns']) | (times['open_available_at_ns'] > times['available_at_ns'])).any() or (times['start_ns'].diff().drop_nulls() < 0).any():
                raise ValueError('Protected execution source has invalid causal clocks or observation order')
            if ((times['end_ns'] <= binding.cutoff_ns) & (times['available_at_ns'] > binding.cutoff_ns)).any():
                raise ValueError('Protected execution availability crosses the reserved cutoff')
            selected = (times['end_ns'] <= binding.cutoff_ns) & (times['available_at_ns'] <= binding.cutoff_ns) & (times['open_available_at_ns'] <= binding.cutoff_ns)
            clocks = {name: values.filter(selected).to_list() for name, values in times.items()}
            clock = 'complete-observation-and-availability'
        else:
            if policy is None or policy.funding is None:
                raise ValueError('Protected funding lacks its actual resolved policy')
            discrete = policy.funding.mechanism == 'discrete'
            quoted = not discrete and policy.funding.params.get('history_interpretation') == 'quoted'
            support = _ns(materialized, 'settlement_at' if discrete else 'end')
            available = _ns(materialized, 'available_at_ns') if quoted or 'available_at_ns' in materialized.columns else support
            if not quoted and (available < support).any():
                raise ValueError('Realized funding cannot be available before its complete support')
            selected = (available <= binding.cutoff_ns) & (_ns(materialized, 'start') <= binding.cutoff_ns if quoted else support <= binding.cutoff_ns)
            clocks = {'actual_support_ns': support.filter(selected).to_list(), 'available_at_ns': available.filter(selected).to_list()}
            clock = 'quoted-as-of' if quoted else 'actual-settlement' if discrete else 'complete-realized-interval'
        admitted = materialized.filter(selected)
        frames[index] = admitted
        witnesses[role] = {'identity': identity, 'clock': clock, 'members': selected.arg_true().to_list(), 'clocks': clocks,
                           'materialized': _frame(materialized), 'admitted': _frame(admitted)}
    return (frames[0], frames[1]), witnesses


def _formation(manifest: object, native: list[pl.DataFrame], formed: list[pl.DataFrame], params: Mapping[str, object], binding: ProjectBinding) -> dict[str, object]:
    declaration = cast(tuple[Callable[..., pl.DataFrame], Mapping[str, object]] | None, getattr(manifest, 'bar_formation', None))
    active = declaration is not None and params.get('bar_type', 'base') != 'base'
    partitions: list[dict[str, object]] = []
    for before, after in zip(native, formed, strict=True):
        times = before['datetime'].dt.epoch('ns').to_list()
        available = _ns(before, 'available_at_ns').to_list() if 'available_at_ns' in before.columns else [cast(int, instant) + cast(int, binding.interval_seconds) * 1_000_000_000 for instant in times]
        offsets = {instant: index for index, instant in enumerate(times)}
        if active and 'bar_count' not in after.columns:
            raise ValueError('Protected formation requires recoverable native constituent counts')
        counts = after['bar_count'].to_list() if active else [1] * after.height
        members: list[list[int]] = []
        last_available: list[int] = []
        previous = 0
        for instant, count in zip(after['datetime'].dt.epoch('ns'), counts, strict=True):
            start = offsets.get(instant)
            if start is None or type(count) is not int or count <= 0 or start < previous or start + count > len(times):
                raise ValueError('Protected formation lost its actual native constituents')
            previous = start + count
            members.append(cast(list[int], times[start:previous]))
            complete = max(cast(list[int], available[start:previous]))
            if complete != available[previous - 1]:
                raise ValueError('Protected formation requires complete constituent availability at its native execution clock')
            last_available.append(complete)
        partitions.append({'native': _frame(before), 'derived': _frame(after), 'members': members,
            'last_available_ns': last_available})
    return {'active': active, 'bar_type': params.get('bar_type', 'base'), 'partitions': partitions,
            'declaration': None if declaration is None else {'method': f'{declaration[0].__module__}.{declaration[0].__qualname__}', 'params': resolve_json(declaration[1], params)}}


def verify_sources(manifest: object, native: list[pl.DataFrame], formed: list[pl.DataFrame], params: Mapping[str, object],
                   frozen: FrozenTradeSources | None) -> FrozenTradeSources | None:
    if not source_workflow_active.get():
        return frozen
    current = cast(ContextVar[_Operation | None], vars(import_module('limen.experiment._holdout_research'))['_CURRENT']).get()
    if current is None or current.manifest is not manifest:
        return frozen
    frozen, roles = _roles(cast(BacktestConfig | None, getattr(manifest, 'backtest_config', None)), params, current.binding, frozen)
    workflow: dict[str, object] = {'version': 1, 'kind': 'sources', 'formation': _formation(manifest, native, formed, params, current.binding), 'roles': roles}
    if current.replay_witness is not None and workflow != current.replay_witness.get('workflow'):
        raise ValueError('Protected native or derived source support changed before fitting')
    current.provenance['workflow'] = copy.deepcopy(workflow)
    return frozen

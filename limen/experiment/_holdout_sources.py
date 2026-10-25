from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from hashlib import sha256
from importlib import import_module
from typing import Protocol, cast

import polars as pl

from limen.experiment._holdout_history import ProjectBinding
from limen.experiment._holdout_source import source_workflow_active
from limen.experiment._prepare_trade_context import FrozenTradeSources

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


def _formation(manifest: object, native: list[pl.DataFrame], formed: list[pl.DataFrame], params: Mapping[str, object], binding: ProjectBinding) -> dict[str, object]:
    declaration = getattr(manifest, 'bar_formation', None)
    active = declaration is not None and params.get('bar_type', 'base') != 'base'
    partitions: list[dict[str, object]] = []
    for before, after in zip(native, formed, strict=True):
        times = before['datetime'].dt.epoch('ns').to_list()
        offsets = {instant: index for index, instant in enumerate(times)}
        if active and 'bar_count' not in after.columns:
            raise ValueError('Protected formation requires recoverable native constituent counts')
        counts = after['bar_count'].to_list() if active else [1] * after.height
        members: list[list[int]] = []
        previous = 0
        for instant, count in zip(after['datetime'].dt.epoch('ns'), counts, strict=True):
            start = offsets.get(instant)
            if start is None or type(count) is not int or count <= 0 or start < previous or start + count > len(times):
                raise ValueError('Protected formation lost its actual native constituents')
            previous = start + count
            members.append(cast(list[int], times[start:previous]))
        partitions.append({'native': _frame(before), 'derived': _frame(after), 'members': members,
            'last_available_ns': [group[-1] + cast(int, binding.interval_seconds) * 1_000_000_000 for group in members]})
    return {'active': active, 'bar_type': params.get('bar_type', 'base'), 'partitions': partitions}


def verify_sources(manifest: object, native: list[pl.DataFrame], formed: list[pl.DataFrame], params: Mapping[str, object],
                   frozen: FrozenTradeSources | None) -> FrozenTradeSources | None:
    if not source_workflow_active.get():
        return frozen
    current = cast(ContextVar[_Operation | None], vars(import_module('limen.experiment._holdout_research'))['_CURRENT']).get()
    if current is None or current.manifest is not manifest:
        return frozen
    workflow: dict[str, object] = {'version': 1, 'kind': 'sources', 'formation': _formation(manifest, native, formed, params, current.binding)}
    if current.replay_witness is not None and workflow != current.replay_witness.get('workflow'):
        raise ValueError('Protected native or derived source support changed before fitting')
    current.provenance['workflow'] = copy.deepcopy(workflow)
    return frozen

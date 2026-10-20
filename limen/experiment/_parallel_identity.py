from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
from importlib import metadata, util
from pathlib import Path

_providers: ContextVar[Mapping[str, list[str]] | None] = ContextVar('parallel_distribution_providers', default=None)
_snapshot: ContextVar[dict[str, list[object]] | None] = ContextVar('parallel_identity_snapshot', default=None)


@cache
def package_distributions() -> Mapping[str, list[str]]:
    return metadata.packages_distributions()


@contextmanager
def distribution_providers(providers: Mapping[str, list[str]]) -> Iterator[None]:
    token = _providers.set(providers)
    try:
        yield
    finally:
        _providers.reset(token)


@contextmanager
def identity_snapshot() -> Iterator[None]:
    token = _snapshot.set({})
    try:
        yield
    finally:
        _snapshot.reset(token)


def module_identity(name: str) -> list[object]:
    snapshot = _snapshot.get()
    if snapshot is None:
        return _read_identity(name)
    if name not in snapshot:
        snapshot[name] = _read_identity(name)
    return snapshot[name]


def _read_identity(name: str) -> list[object]:
    spec = util.find_spec(name)
    path = None if spec is None else spec.origin
    if path in ('built-in', 'frozen'):
        return [name, path, sys.version]
    if path is None:
        raise ValueError(f'Parallel import has no source file: {name}')
    top = name.split('.', 1)[0]
    providers = _providers.get()
    distributions = (providers if providers is not None else package_distributions()).get(top, []) if top not in ('limen', 'tests') else []
    return [name, hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            [[name, metadata.version(name)] for name in sorted(distributions)]]

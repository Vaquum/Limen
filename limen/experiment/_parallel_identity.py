from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
from importlib import metadata, util
from pathlib import Path
from types import FunctionType, MethodType

_providers: ContextVar[Mapping[str, list[str]] | None] = ContextVar('parallel_distribution_providers', default=None)
_snapshot: ContextVar[tuple[dict[str, list[object]], dict[tuple[type, tuple[str, int, int]], type]] | None] = ContextVar('parallel_identity_snapshot', default=None)


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
    token = _snapshot.set(_snapshot.get() or ({}, {}))
    try:
        yield
    finally:
        _snapshot.reset(token)


def module_identity(name: str) -> list[object]:
    context = _snapshot.get()
    snapshot = None if context is None else context[0]
    if snapshot is None:
        return _read_identity(name)
    if name not in snapshot:
        snapshot[name] = _read_identity(name)
    return snapshot[name]


def execution_class(cls: type, identity: tuple[str, int, int], attributes: dict[str, object]) -> type:
    context = _snapshot.get()
    if context is None:
        raise ValueError('Captured parallel class requires an identity snapshot')
    classes, key = context[1], (cls, identity)
    if key not in classes:
        classes[key] = type(cls.__name__, (cls,), attributes)
    return classes[key]


def declaration_changed(source: object, original: object, fingerprint: Callable[[object], object]) -> bool:
    return isinstance(original, (FunctionType, MethodType)) and fingerprint(getattr(source, '__parallel_round_contract__', None)) != fingerprint(getattr(original, '__parallel_round_contract__', None))


def restore_function_metadata(function: FunctionType, source: FunctionType, wrapped: object) -> None:
    function.__module__, function.__qualname__, function.__annotations__ = source.__module__, source.__qualname__, source.__annotations__.copy()
    for name, value in {'__wrapped__': wrapped, '__parallel_round_contract__': vars(source).get('__parallel_round_contract__')}.items():
        if value is not None:
            vars(function)[name] = value


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

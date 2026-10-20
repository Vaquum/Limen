from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date
from pathlib import Path
from types import FunctionType
from typing import cast

_encoded: ContextVar[tuple[dict[int, tuple[int, object]], set[int]] | None] = ContextVar('parallel_encoded_graph', default=None)
_decoded: ContextVar[list[object] | None] = ContextVar('parallel_decoded_graph', default=None)
_copy: ContextVar[tuple[dict[int, tuple[object, object]], dict[int, object]] | None] = ContextVar('parallel_shared_copy', default=None)
_VALUE_ARITY = {'tuple': 2, 'function': 6, 'ref': 2}
_EMPTY_IDS: frozenset[int] = frozenset()


@contextmanager
def shared_copy(memo: dict[int, tuple[object, object]]) -> Iterator[None]:
    token = _copy.set((memo, {}))
    try:
        yield
    finally:
        _copy.reset(token)


def _value_only(value: object, *, functions: bool = True) -> bool:
    if isinstance(value, FunctionType):
        return functions and value.__closure__ is None and value.__kwdefaults__ is None and '__wrapped__' not in vars(value) and _value_only(value.__defaults__)
    if callable(value) and not isinstance(value, type):
        return False
    if isinstance(value, tuple):
        return all(_value_only(item, functions=False) for item in cast(tuple[object, ...], value))
    return isinstance(value, (type(None), str, bool, int, float, date, Path, type))


def _value_node(node: object, *, functions: bool = True) -> bool:
    if not isinstance(node, list) or not node:
        return False
    parts = cast(list[object], node)
    if parts[0] == 'tuple' and len(parts) == _VALUE_ARITY['tuple'] and isinstance(parts[1], list):
        return all(_value_node(item, functions=False) for item in cast(list[object], parts[1]))
    if parts[0] == 'function' and len(parts) == _VALUE_ARITY['function']:
        return functions and _value_node(parts[2]) and parts[3:] == [['null'], ['null'], ['null']]
    return parts[0] in ('null', 'str', 'bool', 'int', 'float', 'date', 'datetime', 'path', 'import')


def encode_graph(value: object, visit: Callable[[], list[object]]) -> list[object]:
    token = _encoded.set(({}, set())) if _encoded.get() is None else None
    try:
        context = _encoded.get()
        assert context is not None
        memo, active = context
        if _value_only(value):
            return visit()
        key = id(value)
        if key in active:
            raise ValueError('Parallel encoding rejects cycles')
        if key in memo:
            return ['ref', memo[key][0]]
        memo[key] = (len(memo), value)
        if (copy := _copy.get()) is not None:
            _ = copy[1].setdefault(len(memo) - 1, value)
        active.add(key)
        try:
            return visit()
        finally:
            active.remove(key)
    finally:
        if token is not None:
            _encoded.reset(token)


def decode_graph(node: object, visit: Callable[[], object], verify: Callable[[object], list[object]]) -> object:
    token = _decoded.set([]) if _decoded.get() is None else None
    try:
        memo = _decoded.get()
        assert memo is not None
        parts = cast(list[object], node) if isinstance(node, list) else []
        if parts and parts[0] == 'ref':
            if len(parts) != _VALUE_ARITY['ref'] or type(parts[1]) is not int or not 0 <= parts[1] < len(memo) or memo[parts[1]] is None:
                raise ValueError('Invalid parallel shared-state reference')
            result = memo[parts[1]]
        else:
            index = len(memo) if parts and not _value_node(parts) else None
            if index is not None:
                memo.append(None)
            result = visit()
            if index is not None:
                if (copy := _copy.get()) is not None:
                    original = copy[1][index]
                    result = copy[0].setdefault(id(original), (original, result))[1]
                    copy[0][id(result)] = (result, result)
                memo[index] = result
    finally:
        if token is not None:
            _decoded.reset(token)
    if token is not None and verify(result) != node:
        raise ValueError('Noncanonical parallel parameter encoding')
    return result

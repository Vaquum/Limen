from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from datetime import date
from pathlib import Path
from types import FunctionType
from typing import cast

_encoded: ContextVar[tuple[dict[int, tuple[int, object]], set[int]] | None] = ContextVar('parallel_encoded_graph', default=None)
_decoded: ContextVar[list[object] | None] = ContextVar('parallel_decoded_graph', default=None)
_COMPOSITES = frozenset(('list', 'tuple', 'map', 'keyed_map', 'state_map', 'function', 'method', 'dataclass', 'instance'))


def _value_only(value: object, *, functions: bool = True) -> bool:
    if isinstance(value, tuple):
        return all(_value_only(item, functions=False) for item in cast(tuple[object, ...], value))
    if isinstance(value, FunctionType):
        return functions and value.__closure__ is None and value.__kwdefaults__ is None and '__wrapped__' not in vars(value) and _value_only(value.__defaults__)
    return isinstance(value, (type(None), str, bool, int, float, date, Path, type))


def _value_node(node: object, *, functions: bool = True) -> bool:
    if not isinstance(node, list) or not node:
        return False
    parts = cast(list[object], node)
    if parts[0] == 'tuple' and len(parts) == 2 and isinstance(parts[1], list):
        return all(_value_node(item, functions=False) for item in cast(list[object], parts[1]))
    if parts[0] == 'function' and len(parts) == 6:
        return functions and _value_node(parts[2]) and parts[3:] == [['null'], ['null'], ['null']]
    return parts[0] in ('null', 'str', 'bool', 'int', 'float', 'date', 'datetime', 'path', 'import')


def encode_graph(value: object, visit: Callable[[], list[object]]) -> list[object]:
    context = _encoded.get()
    token = _encoded.set(({}, set())) if context is None else None
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
        active.add(key)
        try:
            return visit()
        finally:
            active.remove(key)
    finally:
        if token is not None:
            _encoded.reset(token)


def decode_graph(node: object, visit: Callable[[], object], verify: Callable[[object], list[object]]) -> object:
    context = _decoded.get()
    token = _decoded.set([]) if context is None else None
    try:
        memo = _decoded.get()
        assert memo is not None
        parts = cast(list[object], node) if isinstance(node, list) else []
        if parts and parts[0] == 'ref':
            if len(parts) != 2 or type(parts[1]) is not int or not 0 <= parts[1] < len(memo) or memo[parts[1]] is None:
                raise ValueError('Invalid parallel shared-state reference')
            result = memo[parts[1]]
        else:
            index = len(memo) if parts and isinstance(parts[0], str) and parts[0] in _COMPOSITES and not _value_node(parts) else None
            if index is not None:
                memo.append(None)
            result = visit()
            if index is not None:
                memo[index] = result
    finally:
        if token is not None:
            _decoded.reset(token)
    if token is not None and verify(result) != node:
        raise ValueError('Noncanonical parallel parameter encoding')
    return result

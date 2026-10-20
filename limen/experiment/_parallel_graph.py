from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from datetime import date
from pathlib import Path
from types import FunctionType

_encoded: ContextVar[tuple[dict[int, tuple[int, object]], set[int]] | None] = ContextVar('parallel_encoded_graph', default=None)
_decoded: ContextVar[list[object] | None] = ContextVar('parallel_decoded_graph', default=None)
_COMPOSITES = frozenset(('list', 'tuple', 'map', 'keyed_map', 'state_map', 'function', 'method', 'dataclass', 'instance'))


def _value_only(value: object) -> bool:
    if isinstance(value, tuple):
        return all(_value_only(item) for item in value)
    if isinstance(value, FunctionType):
        return value.__closure__ is None and value.__kwdefaults__ is None and '__wrapped__' not in vars(value) and _value_only(value.__defaults__)
    return isinstance(value, (type(None), str, bool, int, float, date, Path, type))


def _value_node(node: object) -> bool:
    if not isinstance(node, list) or not node:
        return False
    if node[0] == 'tuple' and len(node) == 2 and isinstance(node[1], list):
        return all(_value_node(item) for item in node[1])
    if node[0] == 'function' and len(node) == 6:
        return _value_node(node[2]) and node[3:] == [['null'], ['null'], ['null']]
    return node[0] in ('null', 'str', 'bool', 'int', 'float', 'date', 'datetime', 'path', 'import')


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
        if isinstance(node, list) and node and node[0] == 'ref':
            if len(node) != 2 or type(node[1]) is not int or not 0 <= node[1] < len(memo) or memo[node[1]] is None:
                raise ValueError('Invalid parallel shared-state reference')
            result = memo[node[1]]
        else:
            index = len(memo) if isinstance(node, list) and node and isinstance(node[0], str) and node[0] in _COMPOSITES and not _value_node(node) else None
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

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from datetime import date
from pathlib import Path

_encoded: ContextVar[tuple[dict[int, tuple[int, object]], set[int]] | None] = ContextVar('parallel_encoded_graph', default=None)
_decoded: ContextVar[list[object] | None] = ContextVar('parallel_decoded_graph', default=None)
_COMPOSITES = frozenset(('list', 'tuple', 'map', 'keyed_map', 'state_map', 'function', 'method', 'dataclass', 'instance'))


def encode_graph(value: object, visit: Callable[[], list[object]]) -> list[object]:
    context = _encoded.get()
    token = _encoded.set(({}, set())) if context is None else None
    try:
        context = _encoded.get()
        assert context is not None
        memo, active = context
        if isinstance(value, (type(None), str, bool, int, float, date, Path, type)):
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
            index = len(memo) if isinstance(node, list) and node and node[0] in _COMPOSITES else None
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


def reuse_graph(captured: object, original: object) -> None:
    memo = _decoded.get()
    assert memo is not None
    memo[:] = [original if item is captured else item for item in memo]
    if isinstance(captured, tuple) and isinstance(original, tuple):
        for left, right in zip(captured, original, strict=True):
            reuse_graph(left, right)

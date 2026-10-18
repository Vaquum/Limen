from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from sys import getsizeof
from threading import RLock
from typing import TypeVar, cast

import numpy as np
import numpy.typing as npt
import polars as pl

__all__: list[str] = []

_Value = TypeVar('_Value')
_Key = tuple[object, ...]
LineArrays = tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.float64]]
_MIN_MAX_DURATION_HOURS = 2


class _BoundedCache:
    def __init__(self, max_bytes: int) -> None:
        super().__init__()
        self.max_bytes = max_bytes
        self.bytes = 0
        self.entries: OrderedDict[_Key, tuple[object, int]] = OrderedDict()
        self.lock = RLock()
        self.active = True

    def resolve(self, key: _Key, compute: Callable[[], _Value],
                size: Callable[[_Value], int]) -> _Value:
        with self.lock:
            if not self.active:
                return compute()
            saved = self.entries.get(key)
            if saved is not None:
                self.entries.move_to_end(key)
                return cast(_Value, saved[0])
            value = compute()
            weight = size(value) + sum(getsizeof(part) for part in key) + 1024
            if weight <= self.max_bytes:
                while self.entries and self.bytes + weight > self.max_bytes:
                    _, (_, removed) = self.entries.popitem(last=False)
                    self.bytes -= removed
                self.entries[key] = (value, weight)
                self.bytes += weight
            return value

    def close(self) -> None:
        with self.lock:
            self.active = False
            self.entries.clear()
            self.bytes = 0


class ComputationCache:
    def __init__(self, max_bytes: int = 128 * 1024 * 1024) -> None:
        super().__init__()
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError('Computation cache max_bytes must be a nonnegative integer')
        self.primitives = _BoundedCache(max_bytes // 2)
        self.features = _BoundedCache(max_bytes - max_bytes // 2)


_current: ContextVar[ComputationCache | None] = ContextVar('limen_computation_cache', default=None)


def current_cache() -> ComputationCache | None:
    return _current.get()


@contextmanager
def computation_reuse(max_bytes: int = 128 * 1024 * 1024) -> Iterator[ComputationCache]:
    cache = ComputationCache(max_bytes)
    token = _current.set(cache)
    try:
        yield cache
    finally:
        _current.reset(token)
        cache.primitives.close()
        cache.features.close()


def array_key(kind: str, values: npt.NDArray[np.float64],
              params: tuple[object, ...]) -> _Key | None:
    if (any(type(value) not in (bool, int, float) for value in params)
            or not np.isfinite(values).all() or (values == 0).any()):
        return None
    return kind, values.shape, values.tobytes(), tuple((type(value), value) for value in params)


def reuse_series(cache: ComputationCache | None, kind: str,
                 values: npt.NDArray[np.float64], params: tuple[object, ...],
                 compute: Callable[[], pl.Series]) -> pl.Series:
    key = array_key(kind, values, params) if cache is not None else None
    if cache is None or key is None:
        return compute()
    return cache.features.resolve(key, compute, lambda value: int(value.estimated_size())).clone()


def reuse_line_arrays(close: npt.ArrayLike, max_duration_hours: int, min_height_pct: float,
                      cache: ComputationCache | None,
                      compute: Callable[[npt.ArrayLike, int, float], tuple[LineArrays, LineArrays]],
                      ) -> tuple[LineArrays, LineArrays]:
    if cache is None:
        return compute(close, max_duration_hours, min_height_pct)
    if max_duration_hours < _MIN_MAX_DURATION_HOURS:
        raise ValueError('find_price_lines max_duration_hours must be at least 2')
    if min_height_pct <= 0:
        raise ValueError('find_price_lines min_height_pct must be positive')
    values = np.asarray(close, dtype=np.float64)
    key = array_key('price_line_arrays', values, (max_duration_hours, min_height_pct))
    if key is None:
        return compute(values, max_duration_hours, min_height_pct)
    result = cache.primitives.resolve(
        key, lambda: compute(values, max_duration_hours, min_height_pct),
        lambda lines: sum(array.nbytes for direction in lines for array in direction),
    )
    return ((result[0][0].copy(), result[0][1].copy(), result[0][2].copy()),
            (result[1][0].copy(), result[1][1].copy(), result[1][2].copy()))

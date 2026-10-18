from __future__ import annotations

import numpy as np
import numpy.typing as npt

from limen.utils._computation_cache import ComputationCache, LineArrays, reuse_line_arrays

__all__: list[str] = []

MIN_DURATION_BARS = 1
MIN_MAX_DURATION_HOURS = 2
MIN_PRICES = 2


def cached_price_line_arrays(close: npt.ArrayLike, max_duration_hours: int,
                             min_height_pct: float,
                             cache: ComputationCache | None) -> tuple[LineArrays, LineArrays]:
    return reuse_line_arrays(close, max_duration_hours, min_height_pct, cache, _compute_price_line_arrays)


def _compute_price_line_arrays(close: npt.ArrayLike, max_duration_hours: int,
                               min_height_pct: float) -> tuple[LineArrays, LineArrays]:
    if max_duration_hours < MIN_MAX_DURATION_HOURS:
        raise ValueError('find_price_lines max_duration_hours must be at least 2')
    if min_height_pct <= 0:
        raise ValueError('find_price_lines min_height_pct must be positive')
    close = np.asarray(close, dtype=np.float64)
    starts_per_duration: list[npt.NDArray[np.intp]] = []
    ends_per_duration: list[npt.NDArray[np.intp]] = []
    heights_per_duration: list[npt.NDArray[np.float64]] = []
    n_prices = close.shape[0]
    if n_prices < MIN_PRICES:
        empty: LineArrays = (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp), np.empty(0))
        return empty, empty
    for duration in range(MIN_DURATION_BARS, max_duration_hours):
        if duration >= n_prices:
            break
        heights = (close[duration:] - close[:-duration]) / close[:-duration]
        starts = np.nonzero(np.abs(heights) >= min_height_pct)[0]
        starts_per_duration.append(starts)
        ends_per_duration.append(starts + duration)
        heights_per_duration.append(heights[starts])
    starts = np.concatenate(starts_per_duration)
    ends = np.concatenate(ends_per_duration)
    heights = np.concatenate(heights_per_duration)
    long = heights > 0
    return (starts[long], ends[long], heights[long]), (starts[~long], ends[~long], heights[~long])

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ['LineArrays', 'find_price_lines', 'price_line_arrays']

MIN_DURATION_BARS = 1
MIN_MAX_DURATION_HOURS = 2
MIN_PRICES = 2


def find_price_lines(close: npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]],
                     max_duration_hours: int,
                     min_height_pct: float) -> tuple[list[dict[str, float]], list[dict[str, float]]]:

    '''
    Compute linear price movements (lines) from a close-price series.

    A line is any pair of bars (start, end) with duration in
    [1, max_duration_hours) whose relative close-to-close change is at
    least min_height_pct in absolute terms. Lines with a positive change
    are long lines, negative are short lines.

    Args:
        close (np.ndarray): One-dimensional array of close prices
        max_duration_hours (int): Exclusive upper bound on line duration in bars
        min_height_pct (float): Minimum absolute height as a fraction of the start price

    Returns:
        tuple[list[dict], list[dict]]: (long_lines, short_lines), each a list of
            dicts with keys 'start_idx', 'end_idx', 'height_pct', 'duration_hours'

    Raises:
        ValueError: If max_duration_hours is below 2 or min_height_pct is not positive

    NOTE: Lines are detected only within the array passed in, so per-split
    invocation cannot observe other splits.
    '''

    long_lines, short_lines = price_line_arrays(close, max_duration_hours, min_height_pct)
    results = tuple([
        {'start_idx': int(start), 'end_idx': int(end), 'height_pct': float(height),
         'duration_hours': int(end - start)}
        for start, end, height in zip(*lines, strict=True)
    ] for lines in (long_lines, short_lines))
    return results[0], results[1]


LineArrays = tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.float64]]


def price_line_arrays(close: npt.ArrayLike,
                      max_duration_hours: int, min_height_pct: float) -> tuple[LineArrays, LineArrays]:
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

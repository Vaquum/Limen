from typing import Any

import numpy as np
import numpy.typing as npt

from limen.utils._computation_cache import LineArrays, current_cache
from limen.utils._line_arrays import cached_price_line_arrays

__all__ = ['LineArrays', 'find_price_lines', 'price_line_arrays']

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


def price_line_arrays(close: npt.ArrayLike,
                      max_duration_hours: int, min_height_pct: float) -> tuple[LineArrays, LineArrays]:
    return cached_price_line_arrays(close, max_duration_hours, min_height_pct, current_cache())

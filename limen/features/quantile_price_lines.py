from heapq import heappop, heappush

import numpy as np
import numpy.typing as npt
import polars as pl

from limen.features.price_lines import line_active_count, line_recency
from limen.utils.find_price_lines import LineArrays
from limen.utils._line_arrays import cached_price_line_arrays
from limen.utils._computation_cache import ComputationCache, current_cache, reuse_series
import limen._feature_constraints as _check

DEFAULT_QUANTILE_THRESHOLD = 0.75
DEFAULT_DENSITY_LOOKBACK_HOURS = 48
DEFAULT_MOMENTUM_LOOKBACK_HOURS = 6
DEFAULT_HEIGHT_LOOKBACK_HOURS = 24
STRUCT_COL = '_quantile_price_lines'
INTERPOLATION_MIDPOINT = 0.5


def quantile_price_lines(data: pl.DataFrame,
                         max_duration_hours: int,
                         min_height_pct: float,
                         quantile_threshold: float = DEFAULT_QUANTILE_THRESHOLD,
                         density_lookback_hours: int = DEFAULT_DENSITY_LOOKBACK_HOURS,
                         momentum_lookback_hours: int = DEFAULT_MOMENTUM_LOOKBACK_HOURS,
                         height_lookback_hours: int = DEFAULT_HEIGHT_LOOKBACK_HOURS,
                         include_research_only: object = True) -> pl.DataFrame:

    '''
    Compute quantile-line end features from lines completed through each row.
    Select heights at or above each direction's expanding quantile, with ties.
    End windows include both boundaries [t-window, t]; recency caps at density_lookback_hours.
    Other empty windows return zero; active_quantile_count retains retrospective whole-frame spans.

    Args:
        data (pl.DataFrame | pl.LazyFrame): Klines with a close column
        max_duration_hours (int): Exclusive upper bound on line duration in bars
        min_height_pct (float): Minimum absolute height as a fraction of start price
        quantile_threshold (float): Per-direction height quantile
        density_lookback_hours (int): Density window and recency cap
        momentum_lookback_hours (int): Signed-height window
        height_lookback_hours (int): Mean-height and direction-bias window
        include_research_only (bool): Add the retrospective active_quantile_count

    Returns:
        pl.DataFrame | pl.LazyFrame: Input type with five end columns and optional span count

    Raises:
        ValueError: If a lookback is below 1 or include_research_only is not bool
    '''

    _check.integers(lower=1, max_duration_hours=max_duration_hours, density_lookback_hours=density_lookback_hours, momentum_lookback_hours=momentum_lookback_hours, height_lookback_hours=height_lookback_hours)
    _check.numbers(min_height_pct=min_height_pct)
    _check.numbers(lower=0, upper=1, quantile_threshold=quantile_threshold)
    if not isinstance(include_research_only, bool):
        raise ValueError('quantile_price_lines include_research_only must be a bool')





    momentum_col = f"quantile_momentum_{momentum_lookback_hours}h"
    height_col = f"avg_quantile_height_{height_lookback_hours}h"
    density_col = f"quantile_line_density_{density_lookback_hours}h"

    return_fields = {
        'hours_since_quantile_line': pl.Float64,
        density_col: pl.Int64,
        momentum_col: pl.Float64,
        height_col: pl.Float64,
        'quantile_direction_bias': pl.Float64,
    }
    if include_research_only:
        return_fields = {
            'hours_since_quantile_line': pl.Float64,
            'active_quantile_count': pl.Int64,
            **{k: v for k, v in return_fields.items() if k != 'hours_since_quantile_line'},
        }

    return_dtype = pl.Struct(return_fields)

    cache = current_cache()

    def columns(series: pl.Series) -> pl.Series:
        close = series.to_numpy()
        return reuse_series(cache, 'quantile_price_lines', close,
                            (max_duration_hours, min_height_pct, quantile_threshold, density_lookback_hours, momentum_lookback_hours, height_lookback_hours, include_research_only),
                            lambda: _quantile_line_columns(close, max_duration_hours, min_height_pct, quantile_threshold, density_lookback_hours, momentum_lookback_hours, height_lookback_hours, include_research_only, _cache=cache))

    return data.with_columns(
        pl.col('close')
        .cast(pl.Float64)
        .map_batches(
            columns,
            return_dtype=return_dtype,
        )
        .alias(STRUCT_COL)
    ).unnest(STRUCT_COL)


def _quantile_line_columns(close: npt.NDArray[np.float64],
                           max_duration_hours: int,
                           min_height_pct: float,
                           quantile_threshold: float,
                           density_lookback_hours: int,
                           momentum_lookback_hours: int,
                           height_lookback_hours: int,
                           include_research_only: bool, *,
                           _cache: ComputationCache | None = None) -> pl.Series:

    long_lines, short_lines = cached_price_line_arrays(close, max_duration_hours, min_height_pct, _cache)

    long_lines_q = _filter_quantile(long_lines, quantile_threshold)
    short_lines_q = _filter_quantile(short_lines, quantile_threshold)

    n_rows = close.shape[0]
    momentum_col = f"quantile_momentum_{momentum_lookback_hours}h"
    height_col = f"avg_quantile_height_{height_lookback_hours}h"

    density_col = f"quantile_line_density_{density_lookback_hours}h"
    if quantile_threshold == 0:
        ends, heights, directions = _end_events(long_lines_q, short_lines_q)

        momentum = _windowed_sum(ends, heights * directions, n_rows, momentum_lookback_hours)

        height_sums = _windowed_sum(ends, heights, n_rows, height_lookback_hours)
        signed_sums = _windowed_sum(ends, heights * directions, n_rows, height_lookback_hours)
        counts = _windowed_sum(ends, np.ones_like(heights), n_rows, height_lookback_hours)

        avg_height = np.where(counts > 0, height_sums / np.maximum(counts, 1.0), 0.0)
        direction_bias = np.where(height_sums > 0, signed_sums / np.maximum(height_sums, 1e-12), 0.0)

        recency = line_recency(ends, n_rows, density_lookback_hours)
        density = (np.searchsorted(ends, np.arange(n_rows), side='right')
                   - np.searchsorted(ends, np.arange(n_rows) - density_lookback_hours, side='left'))
    else:
        recency, density, momentum, avg_height, direction_bias = _causal_columns(
            long_lines, short_lines, n_rows, quantile_threshold,
            density_lookback_hours, momentum_lookback_hours, height_lookback_hours)

    columns = [
        pl.Series('hours_since_quantile_line', recency),
        pl.Series(density_col, density),
        pl.Series(momentum_col, momentum),
        pl.Series(height_col, avg_height),
        pl.Series('quantile_direction_bias', direction_bias),
    ]
    if include_research_only:
        columns.insert(1, pl.Series('active_quantile_count', line_active_count(long_lines_q, short_lines_q, n_rows)))

    return pl.DataFrame(columns).to_struct(STRUCT_COL)


def _filter_quantile(lines: LineArrays, quantile: float) -> LineArrays:
    if not 0 <= quantile <= 1:
        raise ValueError('filter_lines_by_quantile quantile must be between 0 and 1')
    if lines[2].size == 0:
        return lines
    heights = np.abs(lines[2])
    keep = heights >= np.quantile(heights, quantile)
    return lines[0][keep], lines[1][keep], lines[2][keep]


def _end_events(long: LineArrays, short: LineArrays) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    ends = np.concatenate([long[1], short[1]])
    heights = np.abs(np.concatenate([long[2], short[2]]))
    directions = np.concatenate([np.ones(long[2].size), -np.ones(short[2].size)])
    order = np.argsort(ends, kind='stable')
    return ends[order], heights[order], directions[order]


def _windowed_sum(ends: npt.NDArray[np.intp],
                  values: npt.NDArray[np.float64],
                  n_rows: int,
                  window: int) -> npt.NDArray[np.float64]:

    if ends.size == 0 or n_rows == 0:
        return np.zeros(n_rows, dtype=np.float64)

    prefix = np.concatenate([[0.0], np.cumsum(values)])
    positions = np.arange(n_rows)

    upper = np.searchsorted(ends, positions, side='right')
    lower = np.searchsorted(ends, positions - window, side='left')

    return prefix[upper] - prefix[lower]


def _expanding_quantiles(lines: LineArrays, n_rows: int, quantile: float) -> npt.NDArray[np.float64]:
    thresholds = np.full(n_rows, np.inf)
    lower: list[float] = []
    upper: list[float] = []
    order = np.argsort(lines[1], kind='stable')
    cursor = 0
    for row in range(n_rows):
        while cursor < order.size and lines[1][order[cursor]] <= row:
            height = float(abs(lines[2][order[cursor]]))
            if lower and height <= -lower[0]:
                heappush(lower, -height)
            else:
                heappush(upper, height)
            cursor += 1
        if cursor:
            rank = (cursor - 1) * quantile
            target = int(rank) + 1
            while len(lower) < target:
                heappush(lower, -heappop(upper))
            while len(lower) > target:
                heappush(upper, -heappop(lower))
            # Match NumPy's linear interpolation, including its rounding at ties.
            left, right = -lower[0], upper[0] if upper else -lower[0]
            fraction = rank - int(rank)
            thresholds[row] = left + (right - left) * fraction if fraction < INTERPOLATION_MIDPOINT else right - (right - left) * (1 - fraction)
    return thresholds


def _causal_columns(long: LineArrays, short: LineArrays, n_rows: int, quantile: float,
                    density_window: int, momentum_window: int, height_window: int) -> tuple[
                        npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.float64],
                        npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    thresholds = (_expanding_quantiles(long, n_rows, quantile),
                  _expanding_quantiles(short, n_rows, quantile))
    ends, heights, directions = _end_events(long, short)
    recency = np.full(n_rows, float(density_window))
    density = np.zeros(n_rows, dtype=np.int64)
    momentum, average, bias = np.zeros((3, n_rows))
    window = max(density_window, momentum_window, height_window)
    for row in range(n_rows):
        start = np.searchsorted(ends, row - window, side='left')
        stop = np.searchsorted(ends, row, side='right')
        recent = slice(start, stop)
        keep = heights[recent] >= np.where(directions[recent] > 0, thresholds[0][row], thresholds[1][row])
        selected_ends, selected_heights = ends[recent][keep], heights[recent][keep]
        signed = selected_heights * directions[recent][keep]
        if selected_ends.size:
            recency[row] = min(row - selected_ends[-1], density_window)
        density[row] = np.count_nonzero(selected_ends >= row - density_window)
        momentum[row] = signed[selected_ends >= row - momentum_window].sum()
        height_keep = selected_ends >= row - height_window
        total = selected_heights[height_keep].sum()
        if np.any(height_keep):
            average[row] = total / np.count_nonzero(height_keep)
            bias[row] = signed[height_keep].sum() / max(total, 1e-12)
    return recency, density, momentum, average, bias

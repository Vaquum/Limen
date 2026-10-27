import numpy as np
import numpy.typing as npt
import polars as pl

from limen.utils.find_price_lines import LineArrays
from limen.utils._line_arrays import cached_price_line_arrays
from limen.utils._computation_cache import ComputationCache, current_cache, reuse_series
import limen._feature_constraints as _check

__all__ = ['line_active_count', 'line_recency', 'price_lines']

DEFAULT_BIG_MOVE_LOOKBACK_HOURS = 168
DEFAULT_MOMENTUM_LOOKBACK_HOURS = 6
STRUCT_COL = '_price_lines'


def price_lines(data: pl.DataFrame,
                max_duration_hours: int,
                min_height_pct: float,
                big_move_lookback_hours: int = DEFAULT_BIG_MOVE_LOOKBACK_HOURS,
                momentum_lookback_hours: int = DEFAULT_MOMENTUM_LOOKBACK_HOURS,
                include_research_only: object = True) -> pl.DataFrame:

    '''
    Compute line-based context features from internally detected price lines.

    Detects lines on the frame it receives (per split under the manifest
    pipeline) and adds five columns: 'active_lines' (count of lines spanning
    the bar), 'hours_since_big_move' (bars since the most recent line end,
    capped at big_move_lookback_hours), 'line_momentum_<m>h' (long minus
    short line ends in the trailing window [t-m, t)), 'trending_score'
    (signed end-count balance in [-1, 1]), and 'reversal_potential'
    (min/max end-count ratio in [0, 1]).

    Args:
        data (pl.DataFrame | pl.LazyFrame): Klines dataset with a 'close' column
        max_duration_hours (int): Exclusive upper bound on line duration in bars
        min_height_pct (float): Minimum absolute line height as a fraction of start price
        big_move_lookback_hours (int): Recency cap for 'hours_since_big_move'
        momentum_lookback_hours (int): Trailing window for the end-count columns
        include_research_only (bool): Include the non-live-computable 'active_lines'
            span-count column. Set False for live-safe feature surfaces.

    Returns:
        pl.DataFrame | pl.LazyFrame: The input data, matching the input frame
            type, with five new line-based columns

    Raises:
        ValueError: If a lookback is below 1

    NOTE: Column access is expression-based so the feature runs on both eager
    DataFrames and the LazyFrames piped by the manifest feature pipeline.

    NOTE: 'active_lines' counts lines that span the bar before the line's end
    is knowable, a within-line lookahead inherited from the research design.
    It is research-only and not computable live; the end-event columns are causal.
    '''

    _check.integers(lower=1, max_duration_hours=max_duration_hours, big_move_lookback_hours=big_move_lookback_hours, momentum_lookback_hours=momentum_lookback_hours)
    _check.numbers(min_height_pct=min_height_pct)
    _check.booleans(include_research_only=include_research_only)

    if big_move_lookback_hours < 1:
        raise ValueError('price_lines big_move_lookback_hours must be at least 1')

    if momentum_lookback_hours < 1:
        raise ValueError('price_lines momentum_lookback_hours must be at least 1')

    if not isinstance(include_research_only, bool):
        raise ValueError('price_lines include_research_only must be a bool')

    momentum_col = f"line_momentum_{momentum_lookback_hours}h"

    return_fields = {
        'hours_since_big_move': pl.Float64,
        momentum_col: pl.Float64,
        'trending_score': pl.Float64,
        'reversal_potential': pl.Float64,
    }
    if include_research_only:
        return_fields = {'active_lines': pl.Int64, **return_fields}

    return_dtype = pl.Struct(return_fields)

    cache = current_cache()

    def columns(series: pl.Series) -> pl.Series:
        close = series.to_numpy()
        return reuse_series(cache, 'price_lines', close,
                            (max_duration_hours, min_height_pct, big_move_lookback_hours, momentum_lookback_hours, include_research_only),
                            lambda: _price_line_columns(close, max_duration_hours, min_height_pct, big_move_lookback_hours, momentum_lookback_hours, include_research_only, _cache=cache))

    return data.with_columns(
        pl.col('close')
        .cast(pl.Float64)
        .map_batches(
            columns,
            return_dtype=return_dtype,
        )
        .alias(STRUCT_COL)
    ).unnest(STRUCT_COL)


def _price_line_columns(close: npt.NDArray[np.float64],
                        max_duration_hours: int,
                        min_height_pct: float,
                        big_move_lookback_hours: int,
                        momentum_lookback_hours: int,
                        include_research_only: bool, *,
                        _cache: ComputationCache | None = None) -> pl.Series:

    '''
    Compute the four or five line-based columns as a struct series.

    Args:
        close (np.ndarray): Close prices for the frame
        max_duration_hours (int): Exclusive upper bound on line duration in bars
        min_height_pct (float): Minimum absolute line height as a fraction of start price
        big_move_lookback_hours (int): Recency cap for 'hours_since_big_move'
        momentum_lookback_hours (int): Trailing window for the end-count columns
        include_research_only (bool): Include the research-only span-count column

    Returns:
        pl.Series: Struct series with the line-based fields
    '''

    long_lines, short_lines = cached_price_line_arrays(close, max_duration_hours, min_height_pct, _cache)

    n_rows = close.shape[0]
    ends = np.sort(np.concatenate([long_lines[1], short_lines[1]]))
    momentum_col = f"line_momentum_{momentum_lookback_hours}h"

    long_counts = _end_counts_in_window(long_lines, n_rows, momentum_lookback_hours)
    short_counts = _end_counts_in_window(short_lines, n_rows, momentum_lookback_hours)

    total = long_counts + short_counts
    balance = long_counts - short_counts

    trending = np.where(total > 0, balance / np.maximum(total, 1), 0.0)
    reversal = np.where(
        total > 0,
        np.minimum(long_counts, short_counts) / np.maximum(np.maximum(long_counts, short_counts), 1),
        0.0,
    )

    columns = [
        pl.Series('hours_since_big_move', line_recency(ends, n_rows, big_move_lookback_hours)),
        pl.Series(momentum_col, balance.astype(np.float64)),
        pl.Series('trending_score', trending),
        pl.Series('reversal_potential', reversal),
    ]
    if include_research_only:
        columns.insert(0, pl.Series('active_lines', line_active_count(long_lines, short_lines, n_rows)))

    return pl.DataFrame(columns).to_struct(STRUCT_COL)


def _end_counts_in_window(lines: LineArrays, n_rows: int, window: int) -> npt.NDArray[np.intp]:
    ends = np.sort(lines[1])
    positions = np.arange(n_rows)
    return (np.searchsorted(ends, positions, side='left')
            - np.searchsorted(ends, positions - window, side='left'))


def line_recency(ends: npt.NDArray[np.intp], n_rows: int, lookback: int) -> npt.NDArray[np.float64]:
    positions = np.arange(n_rows)
    previous = np.searchsorted(ends, positions, side='right') - 1
    result = np.full(n_rows, float(lookback))
    known = previous >= 0
    result[known] = np.minimum(positions[known] - ends[previous[known]], lookback)
    return result


def line_active_count(long: LineArrays, short: LineArrays, n_rows: int) -> npt.NDArray[np.intp]:
    positions = np.arange(n_rows)
    starts = np.sort(np.concatenate([long[0], short[0]]))
    stops = np.sort(np.concatenate([long[1], short[1]]) + 1)
    return (np.searchsorted(starts, positions, side='right')
            - np.searchsorted(stops, positions, side='right'))

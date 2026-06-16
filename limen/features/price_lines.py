import numpy as np
import polars as pl

from limen.features.active_lines import active_lines
from limen.features.hours_since_big_move import hours_since_big_move
from limen.utils.find_price_lines import find_price_lines

DEFAULT_BIG_MOVE_LOOKBACK_HOURS = 168
DEFAULT_MOMENTUM_LOOKBACK_HOURS = 6
STRUCT_COL = '_price_lines'


def price_lines(data: pl.DataFrame,
                max_duration_hours: int,
                min_height_pct: float,
                big_move_lookback_hours: int = DEFAULT_BIG_MOVE_LOOKBACK_HOURS,
                momentum_lookback_hours: int = DEFAULT_MOMENTUM_LOOKBACK_HOURS,
                include_research_only: bool = True) -> pl.DataFrame:

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

    return data.with_columns(
        pl.col('close')
        .cast(pl.Float64)
        .map_batches(
            lambda s: _price_line_columns(
                s.to_numpy(),
                max_duration_hours,
                min_height_pct,
                big_move_lookback_hours,
                momentum_lookback_hours,
                include_research_only,
            ),
            return_dtype=return_dtype,
        )
        .alias(STRUCT_COL)
    ).unnest(STRUCT_COL)


def _price_line_columns(close: np.ndarray,
                        max_duration_hours: int,
                        min_height_pct: float,
                        big_move_lookback_hours: int,
                        momentum_lookback_hours: int,
                        include_research_only: bool) -> pl.Series:

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

    long_lines, short_lines = find_price_lines(close, max_duration_hours, min_height_pct)

    frame = pl.DataFrame({'close': close})
    if include_research_only:
        frame = active_lines(frame, long_lines, short_lines)
    frame = hours_since_big_move(frame, long_lines, short_lines, big_move_lookback_hours)

    n_rows = close.shape[0]
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
        pl.col('hours_since_big_move').cast(pl.Float64),
        pl.Series(momentum_col, balance.astype(np.float64)),
        pl.Series('trending_score', trending),
        pl.Series('reversal_potential', reversal),
    ]
    if include_research_only:
        columns.insert(0, pl.col('active_lines').cast(pl.Int64))

    frame = frame.with_columns(columns)

    return frame.drop('close').to_struct(STRUCT_COL)


def _end_counts_in_window(lines: list[dict], n_rows: int, window: int) -> np.ndarray:

    '''
    Compute per-bar counts of line ends inside the trailing window [t-window, t).

    Args:
        lines (list[dict]): Line dicts with an 'end_idx' key
        n_rows (int): Number of bars in the frame
        window (int): Trailing window length in bars

    Returns:
        np.ndarray: Integer end counts per bar
    '''

    if not lines or n_rows == 0:
        return np.zeros(n_rows, dtype=np.int64)

    ends = np.sort(np.array([line['end_idx'] for line in lines]))
    positions = np.arange(n_rows)

    upper = np.searchsorted(ends, positions, side='left')
    lower = np.searchsorted(ends, positions - window, side='left')

    return upper - lower

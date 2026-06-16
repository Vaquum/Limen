import numpy as np
import polars as pl

from limen.features.active_quantile_count import active_quantile_count
from limen.features.hours_since_quantile_line import hours_since_quantile_line
from limen.features.quantile_line_density import quantile_line_density
from limen.utils.filter_lines_by_quantile import filter_lines_by_quantile
from limen.utils.find_price_lines import find_price_lines

DEFAULT_QUANTILE_THRESHOLD = 0.75
DEFAULT_DENSITY_LOOKBACK_HOURS = 48
DEFAULT_MOMENTUM_LOOKBACK_HOURS = 6
DEFAULT_HEIGHT_LOOKBACK_HOURS = 24
STRUCT_COL = '_quantile_price_lines'


def quantile_price_lines(data: pl.DataFrame,
                         max_duration_hours: int,
                         min_height_pct: float,
                         quantile_threshold: float = DEFAULT_QUANTILE_THRESHOLD,
                         density_lookback_hours: int = DEFAULT_DENSITY_LOOKBACK_HOURS,
                         momentum_lookback_hours: int = DEFAULT_MOMENTUM_LOOKBACK_HOURS,
                         height_lookback_hours: int = DEFAULT_HEIGHT_LOOKBACK_HOURS,
                         include_research_only: bool = True) -> pl.DataFrame:

    '''
    Compute quantile-line context features from internally detected price lines.

    Detects lines on the frame it receives (per split under the manifest
    pipeline), keeps lines at or above the height quantile per direction, and
    adds six columns: 'hours_since_quantile_line' (bars since the most recent
    quantile-line end, capped at density_lookback_hours),
    'active_quantile_count' (count of quantile lines spanning the bar),
    'quantile_line_density_<d>h' (quantile-line ends in the trailing window),
    'quantile_momentum_<m>h' (signed sum of line heights over ends in
    [t-m, t]), 'avg_quantile_height_<h>h' (mean absolute height over ends in
    [t-h, t]), and 'quantile_direction_bias' (height-weighted direction in
    [-1, 1] over the same window).

    Args:
        data (pl.DataFrame | pl.LazyFrame): Klines dataset with a 'close' column
        max_duration_hours (int): Exclusive upper bound on line duration in bars
        min_height_pct (float): Minimum absolute line height as a fraction of start price
        quantile_threshold (float): Height quantile below which lines are dropped
        density_lookback_hours (int): Window for density and the recency cap
        momentum_lookback_hours (int): Window for the signed momentum column
        height_lookback_hours (int): Window for the height and direction-bias columns
        include_research_only (bool): Include the non-live-computable
            'active_quantile_count' span-count column. Set False for live-safe
            feature surfaces.

    Returns:
        pl.DataFrame | pl.LazyFrame: The input data, matching the input frame
            type, with six new quantile-line columns

    Raises:
        ValueError: If a lookback is below 1

    NOTE: Column access is expression-based so the feature runs on both eager
    DataFrames and the LazyFrames piped by the manifest feature pipeline.

    NOTE: 'active_quantile_count' counts lines that span the bar before the
    line's end is knowable, a within-line lookahead inherited from the research
    design. It is research-only and not computable live; the end-event columns
    are causal.
    '''

    if density_lookback_hours < 1:
        raise ValueError('quantile_price_lines density_lookback_hours must be at least 1')

    if momentum_lookback_hours < 1:
        raise ValueError('quantile_price_lines momentum_lookback_hours must be at least 1')

    if height_lookback_hours < 1:
        raise ValueError('quantile_price_lines height_lookback_hours must be at least 1')

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

    return data.with_columns(
        pl.col('close')
        .cast(pl.Float64)
        .map_batches(
            lambda s: _quantile_line_columns(
                s.to_numpy(),
                max_duration_hours,
                min_height_pct,
                quantile_threshold,
                density_lookback_hours,
                momentum_lookback_hours,
                height_lookback_hours,
                include_research_only,
            ),
            return_dtype=return_dtype,
        )
        .alias(STRUCT_COL)
    ).unnest(STRUCT_COL)


def _quantile_line_columns(close: np.ndarray,
                           max_duration_hours: int,
                           min_height_pct: float,
                           quantile_threshold: float,
                           density_lookback_hours: int,
                           momentum_lookback_hours: int,
                           height_lookback_hours: int,
                           include_research_only: bool) -> pl.Series:

    '''
    Compute the six quantile-line columns as a struct series.

    Args:
        close (np.ndarray): Close prices for the frame
        max_duration_hours (int): Exclusive upper bound on line duration in bars
        min_height_pct (float): Minimum absolute line height as a fraction of start price
        quantile_threshold (float): Height quantile below which lines are dropped
        density_lookback_hours (int): Window for density and the recency cap
        momentum_lookback_hours (int): Window for the signed momentum column
        height_lookback_hours (int): Window for the height and direction-bias columns
        include_research_only (bool): Include the research-only span-count column

    Returns:
        pl.Series: Struct series with the six quantile-line fields
    '''

    long_lines, short_lines = find_price_lines(close, max_duration_hours, min_height_pct)

    long_lines_q = filter_lines_by_quantile(long_lines, quantile_threshold)
    short_lines_q = filter_lines_by_quantile(short_lines, quantile_threshold)

    frame = pl.DataFrame({'close': close})
    frame = hours_since_quantile_line(frame, long_lines_q, short_lines_q, density_lookback_hours)
    if include_research_only:
        frame = active_quantile_count(frame, long_lines_q, short_lines_q)
    frame = quantile_line_density(frame, long_lines_q, short_lines_q, density_lookback_hours)

    n_rows = close.shape[0]
    momentum_col = f"quantile_momentum_{momentum_lookback_hours}h"
    height_col = f"avg_quantile_height_{height_lookback_hours}h"

    ends, heights, directions = _end_events(long_lines_q, short_lines_q)

    momentum = _windowed_sum(ends, heights * directions, n_rows, momentum_lookback_hours)

    height_sums = _windowed_sum(ends, heights, n_rows, height_lookback_hours)
    signed_sums = _windowed_sum(ends, heights * directions, n_rows, height_lookback_hours)
    counts = _windowed_sum(ends, np.ones_like(heights), n_rows, height_lookback_hours)

    avg_height = np.where(counts > 0, height_sums / np.maximum(counts, 1.0), 0.0)
    direction_bias = np.where(height_sums > 0, signed_sums / np.maximum(height_sums, 1e-12), 0.0)

    density_col = f"quantile_line_density_{density_lookback_hours}h"

    columns = [
        pl.col('hours_since_quantile_line').cast(pl.Float64),
        pl.col(density_col).cast(pl.Int64),
        pl.Series(momentum_col, momentum),
        pl.Series(height_col, avg_height),
        pl.Series('quantile_direction_bias', direction_bias),
    ]
    if include_research_only:
        columns.insert(1, pl.col('active_quantile_count').cast(pl.Int64))

    frame = frame.with_columns(columns)

    return frame.drop('close').to_struct(STRUCT_COL)


def _end_events(long_lines_q: list[dict],
                short_lines_q: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    '''
    Compute end-sorted event arrays from quantile-filtered lines.

    Args:
        long_lines_q (list[dict]): Quantile-filtered long lines
        short_lines_q (list[dict]): Quantile-filtered short lines

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: End indices, absolute
            heights, and directions (+1 long, -1 short), sorted by end index
    '''

    ends = np.array(
        [line['end_idx'] for line in long_lines_q] + [line['end_idx'] for line in short_lines_q],
        dtype=np.int64,
    )
    heights = np.abs(np.array(
        [line['height_pct'] for line in long_lines_q] + [line['height_pct'] for line in short_lines_q],
        dtype=np.float64,
    ))
    directions = np.array(
        [1.0] * len(long_lines_q) + [-1.0] * len(short_lines_q), dtype=np.float64
    )

    order = np.argsort(ends, kind='stable')

    return ends[order], heights[order], directions[order]


def _windowed_sum(ends: np.ndarray,
                  values: np.ndarray,
                  n_rows: int,
                  window: int) -> np.ndarray:

    '''
    Compute per-bar sums of values whose end index falls in [t-window, t].

    Args:
        ends (np.ndarray): End indices sorted ascending
        values (np.ndarray): Values aligned to ends
        n_rows (int): Number of bars in the frame
        window (int): Trailing window length in bars

    Returns:
        np.ndarray: Windowed sums per bar
    '''

    if ends.size == 0 or n_rows == 0:
        return np.zeros(n_rows, dtype=np.float64)

    prefix = np.concatenate([[0.0], np.cumsum(values)])
    positions = np.arange(n_rows)

    upper = np.searchsorted(ends, positions, side='right')
    lower = np.searchsorted(ends, positions - window, side='left')

    return prefix[upper] - prefix[lower]

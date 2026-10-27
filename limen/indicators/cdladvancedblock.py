import numpy as np
import numpy.typing as npt
import polars as pl
import limen._feature_constraints as _check

__all__ = ['candle_totals', 'cdladvancedblock']

CDLADVANCEDBLOCK_BODY_LONG_AVG_PERIOD = 10
CDLADVANCEDBLOCK_BODY_LONG_PERIOD_TOTAL = 0.0
CDLADVANCEDBLOCK_FAR_AVG_PERIOD = 5
CDLADVANCEDBLOCK_FAR_FACTOR = 0.6
CDLADVANCEDBLOCK_NEAR_AVG_PERIOD = 5
CDLADVANCEDBLOCK_NEAR_FACTOR = 0.2
CDLADVANCEDBLOCK_SHADOW_LONG_AVG_PERIOD = 0
CDLADVANCEDBLOCK_SHADOW_SHORT_AVG_PERIOD = 10



def candle_totals(
    values: npt.NDArray[np.float64], period: int, start: int, count: int,
    initial: float = 0.0,
) -> npt.NDArray[np.float64]:
    total = initial
    for value in values[start - period:start]:
        total += value
    out = np.empty(count)
    out[0] = total
    out[1:] = (values[start:start + count - 1]
               - values[start - period:start - period + count - 1])
    return np.cumsum(out, out=out)

def _cdladvancedblock_impl(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Advance Block candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdladvancedblock'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    body_long_avg_period = CDLADVANCEDBLOCK_BODY_LONG_AVG_PERIOD
    shadow_short_avg_period = CDLADVANCEDBLOCK_SHADOW_SHORT_AVG_PERIOD
    shadow_long_avg_period = CDLADVANCEDBLOCK_SHADOW_LONG_AVG_PERIOD
    near_avg_period = CDLADVANCEDBLOCK_NEAR_AVG_PERIOD
    near_factor = CDLADVANCEDBLOCK_NEAR_FACTOR
    far_avg_period = CDLADVANCEDBLOCK_FAR_AVG_PERIOD
    far_factor = CDLADVANCEDBLOCK_FAR_FACTOR

    lookback_total = max(shadow_long_avg_period, shadow_short_avg_period, far_avg_period, near_avg_period, body_long_avg_period) + 2

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdladvancedblock', values=out))

    count = n - lookback_total
    body = np.abs(close_values - open_values)
    upper = high_values - np.where(close_values > open_values, close_values, open_values)
    price_range = high_values - low_values
    o0, o1, o2 = (open_values[lookback_total - lag:n - lag] for lag in range(3))
    c0, c1, c2 = (close_values[lookback_total - lag:n - lag] for lag in range(3))
    real_body_i0, real_body_i1, real_body_i2 = (body[lookback_total - lag:n - lag] for lag in range(3))
    upper_shadow_i0, upper_shadow_i1, upper_shadow_i2 = (upper[lookback_total - lag:n - lag] for lag in range(3))
    lower = np.where(close_values < open_values, close_values, open_values) - low_values
    shadows = upper + lower
    shadow_short_avg_i0, shadow_short_avg_i1, shadow_short_avg_i2 = (
        (candle_totals(shadows, shadow_short_avg_period, lookback_total - lag, count) / shadow_short_avg_period) / 2.0
        for lag in range(3))
    shadow_long_avg_i0 = (real_body_i0 if shadow_long_avg_period == 0 else
                         candle_totals(body, shadow_long_avg_period, lookback_total, count) / shadow_long_avg_period)
    body_long_avg_i2 = candle_totals(body, body_long_avg_period, lookback_total - 2, count,
                                   CDLADVANCEDBLOCK_BODY_LONG_PERIOD_TOTAL) / body_long_avg_period
    first_white, second_white, third_white = c2 >= o2, c1 >= o1, c0 >= o0
    far_avg_i1, far_avg_i2 = (
        far_factor * (candle_totals(price_range, far_avg_period, lookback_total - lag, count) / far_avg_period)
        for lag in (1, 2))
    near_avg_i1, near_avg_i2 = (
        near_factor * (candle_totals(price_range, near_avg_period, lookback_total - lag, count) / near_avg_period)
        for lag in (1, 2))
    matched = (
        first_white
        & second_white
        & third_white
        & (c0 > c1)
        & (c1 > c2)
        & (o1 > o2)
        & (o1 <= c2 + near_avg_i2)
        & (o0 > o1)
        & (o0 <= c1 + near_avg_i1)
        & (real_body_i2 > body_long_avg_i2)
        & (upper_shadow_i2 < shadow_short_avg_i2)
        & ((real_body_i1 < real_body_i2 - far_avg_i2)
        & (real_body_i0 < real_body_i1 + near_avg_i1)
        | (real_body_i0 < real_body_i1 - far_avg_i1)
        | (real_body_i0 < real_body_i1)
        & (real_body_i1 < real_body_i2)
        & ((upper_shadow_i0 > shadow_short_avg_i0)
        | (upper_shadow_i1 > shadow_short_avg_i1))
        | (real_body_i0 < real_body_i1)
        & (upper_shadow_i0 > shadow_long_avg_i0))
    )
    out[lookback_total:][matched] = -100

    return data.with_columns(pl.Series(name='cdladvancedblock', values=out))


def cdladvancedblock(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    _check.names(open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col)

    out_col = 'cdladvancedblock'
    input_cols = [open_col, high_col, low_col, close_col]
    return data.with_columns(
        pl.struct(input_cols).map_batches(
            lambda s: _cdladvancedblock_impl(
                pl.DataFrame({col: s.struct.field(col) for col in input_cols}),
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            ).get_column(out_col),
            return_dtype=pl.Int32,
        ).alias(out_col)
    )

import numpy as np
import polars as pl

from limen.indicators.cdladvancedblock import candle_totals as _candle_totals
import limen._feature_constraints as _check

CDL3WHITESOLDIERS_BODY_SHORT_AVG_PERIOD = 10
CDL3WHITESOLDIERS_BODY_SHORT_PERIOD_TOTAL = 0.0
CDL3WHITESOLDIERS_FAR_AVG_PERIOD = 5
CDL3WHITESOLDIERS_FAR_FACTOR = 0.6
CDL3WHITESOLDIERS_NEAR_AVG_PERIOD = 5
CDL3WHITESOLDIERS_NEAR_FACTOR = 0.2
CDL3WHITESOLDIERS_SHADOW_VS_AVG_PERIOD = 10
CDL3WHITESOLDIERS_SHADOW_VS_FACTOR = 0.1


def _cdl3whitesoldiers_impl(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Three Advancing White Soldiers candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdl3whitesoldiers'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    shadow_vs_avg_period = CDL3WHITESOLDIERS_SHADOW_VS_AVG_PERIOD
    shadow_vs_factor = CDL3WHITESOLDIERS_SHADOW_VS_FACTOR
    body_short_avg_period = CDL3WHITESOLDIERS_BODY_SHORT_AVG_PERIOD
    near_avg_period = CDL3WHITESOLDIERS_NEAR_AVG_PERIOD
    near_factor = CDL3WHITESOLDIERS_NEAR_FACTOR
    far_avg_period = CDL3WHITESOLDIERS_FAR_AVG_PERIOD
    far_factor = CDL3WHITESOLDIERS_FAR_FACTOR

    lookback_total = max(shadow_vs_avg_period, body_short_avg_period, far_avg_period, near_avg_period) + 2

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdl3whitesoldiers', values=out))

    count = n - lookback_total
    body = np.abs(close_values - open_values)
    upper = high_values - np.where(close_values > open_values, close_values, open_values)
    price_range = high_values - low_values
    o0, o1, o2 = (open_values[lookback_total - lag:n - lag] for lag in range(3))
    c0, c1, c2 = (close_values[lookback_total - lag:n - lag] for lag in range(3))
    real_body_i0, real_body_i1, real_body_i2 = (body[lookback_total - lag:n - lag] for lag in range(3))
    upper_shadow_i0, upper_shadow_i1, upper_shadow_i2 = (upper[lookback_total - lag:n - lag] for lag in range(3))
    far_avg_i1, far_avg_i2 = (
        far_factor * (_candle_totals(price_range, far_avg_period, lookback_total - lag, count) / far_avg_period)
        for lag in (1, 2))
    near_avg_i1, near_avg_i2 = (
        near_factor * (_candle_totals(price_range, near_avg_period, lookback_total - lag, count) / near_avg_period)
        for lag in (1, 2))
    shadow_vs_avg_i0, shadow_vs_avg_i1, shadow_vs_avg_i2 = (
        shadow_vs_factor * (_candle_totals(price_range, shadow_vs_avg_period, lookback_total - lag, count) / shadow_vs_avg_period)
        for lag in range(3))
    body_short_avg_i0 = _candle_totals(body, body_short_avg_period, lookback_total, count,
                                    CDL3WHITESOLDIERS_BODY_SHORT_PERIOD_TOTAL) / body_short_avg_period
    matched = (
        (c2 >= o2)
        & (upper_shadow_i2 < shadow_vs_avg_i2)
        & (c1 >= o1)
        & (upper_shadow_i1 < shadow_vs_avg_i1)
        & (c0 >= o0)
        & (upper_shadow_i0 < shadow_vs_avg_i0)
        & (c0 > c1)
        & (c1 > c2)
        & (o1 > o2)
        & (o1 <= c2 + near_avg_i2)
        & (o0 > o1)
        & (o0 <= c1 + near_avg_i1)
        & (real_body_i1 > real_body_i2 - far_avg_i2)
        & (real_body_i0 > real_body_i1 - far_avg_i1)
        & (real_body_i0 > body_short_avg_i0)
    )
    out[lookback_total:][matched] = 100

    return data.with_columns(pl.Series(name='cdl3whitesoldiers', values=out))


def cdl3whitesoldiers(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    _check.names(open_col=open_col, high_col=high_col, low_col=low_col, close_col=close_col)

    out_col = 'cdl3whitesoldiers'
    input_cols = [open_col, high_col, low_col, close_col]
    return data.with_columns(
        pl.struct(input_cols).map_batches(
            lambda s: _cdl3whitesoldiers_impl(
                pl.DataFrame({col: s.struct.field(col) for col in input_cols}),
                open_col=open_col,
                high_col=high_col,
                low_col=low_col,
                close_col=close_col,
            ).get_column(out_col),
            return_dtype=pl.Int32,
        ).alias(out_col)
    )

import numpy as np
import polars as pl

CDLSHOOTINGSTAR_BODY_PERIOD_TOTAL = 0.0
CDLSHOOTINGSTAR_BODY_SHORT_AVG_PERIOD = 10
CDLSHOOTINGSTAR_SHADOW_LONG_AVG_PERIOD = 0
CDLSHOOTINGSTAR_SHADOW_LONG_FACTOR = 1.0
CDLSHOOTINGSTAR_SHADOW_LONG_PERIOD_TOTAL = 0.0
CDLSHOOTINGSTAR_SHADOW_VS_AVG_PERIOD = 10
CDLSHOOTINGSTAR_SHADOW_VS_FACTOR = 0.1
CDLSHOOTINGSTAR_SHADOW_VS_PERIOD_TOTAL = 0.0


def cdlshootingstar(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Shooting Star candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdlshootingstar'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    body_short_avg_period = CDLSHOOTINGSTAR_BODY_SHORT_AVG_PERIOD
    shadow_long_avg_period = CDLSHOOTINGSTAR_SHADOW_LONG_AVG_PERIOD
    shadow_long_factor = CDLSHOOTINGSTAR_SHADOW_LONG_FACTOR
    shadow_vs_avg_period = CDLSHOOTINGSTAR_SHADOW_VS_AVG_PERIOD
    shadow_vs_factor = CDLSHOOTINGSTAR_SHADOW_VS_FACTOR
    lookback_total = max(
        max(body_short_avg_period, shadow_long_avg_period),
        shadow_vs_avg_period,
    ) + 1

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlshootingstar', values=out))

    body_period_total = CDLSHOOTINGSTAR_BODY_PERIOD_TOTAL
    shadow_long_period_total = CDLSHOOTINGSTAR_SHADOW_LONG_PERIOD_TOTAL
    shadow_vs_period_total = CDLSHOOTINGSTAR_SHADOW_VS_PERIOD_TOTAL
    body_trailing_idx = lookback_total - body_short_avg_period
    shadow_long_trailing_idx = lookback_total - shadow_long_avg_period
    shadow_vs_trailing_idx = lookback_total - shadow_vs_avg_period

    i = body_trailing_idx
    while i < lookback_total:
        body_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = shadow_long_trailing_idx
    while i < lookback_total:
        shadow_long_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = shadow_vs_trailing_idx
    while i < lookback_total:
        shadow_vs_period_total += high_values[i] - low_values[i]
        i += 1

    i = lookback_total
    while i < n:
        real_body = abs(close_values[i] - open_values[i])
        upper_shadow = high_values[i] - max(open_values[i], close_values[i])
        lower_shadow = min(open_values[i], close_values[i]) - low_values[i]
        body_avg = body_period_total / body_short_avg_period
        shadow_vs_avg = shadow_vs_factor * (shadow_vs_period_total / shadow_vs_avg_period)

        if shadow_long_avg_period == 0:
            shadow_long_avg = shadow_long_factor * real_body
        else:
            shadow_long_avg = shadow_long_factor * (shadow_long_period_total / shadow_long_avg_period)

        real_body_gap_up = min(open_values[i], close_values[i]) > max(open_values[i - 1], close_values[i - 1])

        if (
            real_body < body_avg
            and upper_shadow > shadow_long_avg
            and lower_shadow < shadow_vs_avg
            and real_body_gap_up
        ):
            out[i] = -100

        body_period_total += (
            abs(close_values[i] - open_values[i])
            - abs(close_values[body_trailing_idx] - open_values[body_trailing_idx])
        )
        shadow_long_period_total += (
            abs(close_values[i] - open_values[i])
            - abs(close_values[shadow_long_trailing_idx] - open_values[shadow_long_trailing_idx])
        )
        shadow_vs_period_total += (
            (high_values[i] - low_values[i])
            - (high_values[shadow_vs_trailing_idx] - low_values[shadow_vs_trailing_idx])
        )
        i += 1
        body_trailing_idx += 1
        shadow_long_trailing_idx += 1
        shadow_vs_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdlshootingstar', values=out))

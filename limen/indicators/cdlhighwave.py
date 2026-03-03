import numpy as np
import polars as pl

CDLHIGHWAVE_BODY_PERIOD_TOTAL = 0.0
CDLHIGHWAVE_BODY_SHORT_AVG_PERIOD = 10
CDLHIGHWAVE_SHADOW_PERIOD_TOTAL = 0.0
CDLHIGHWAVE_SHADOW_VL_AVG_PERIOD = 0
CDLHIGHWAVE_SHADOW_VL_FACTOR = 2.0


def cdlhighwave(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute High-Wave Candle pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdlhighwave'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    body_short_avg_period = CDLHIGHWAVE_BODY_SHORT_AVG_PERIOD
    shadow_vl_avg_period = CDLHIGHWAVE_SHADOW_VL_AVG_PERIOD
    shadow_vl_factor = CDLHIGHWAVE_SHADOW_VL_FACTOR
    lookback_total = max(body_short_avg_period, shadow_vl_avg_period)

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlhighwave', values=out))

    body_period_total = CDLHIGHWAVE_BODY_PERIOD_TOTAL
    shadow_period_total = CDLHIGHWAVE_SHADOW_PERIOD_TOTAL
    body_trailing_idx = lookback_total - body_short_avg_period
    shadow_trailing_idx = lookback_total - shadow_vl_avg_period

    i = body_trailing_idx
    while i < lookback_total:
        body_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = shadow_trailing_idx
    while i < lookback_total:
        shadow_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = lookback_total
    while i < n:
        real_body = abs(close_values[i] - open_values[i])
        upper_shadow = high_values[i] - max(open_values[i], close_values[i])
        lower_shadow = min(open_values[i], close_values[i]) - low_values[i]

        body_avg = body_period_total / body_short_avg_period
        if shadow_vl_avg_period == 0:
            shadow_avg = shadow_vl_factor * real_body
        else:
            shadow_avg = shadow_vl_factor * (shadow_period_total / shadow_vl_avg_period)

        if (
            real_body < body_avg
            and upper_shadow > shadow_avg
            and lower_shadow > shadow_avg
        ):
            color = 1 if close_values[i] >= open_values[i] else -1
            out[i] = color * 100

        body_period_total += (
            abs(close_values[i] - open_values[i])
            - abs(close_values[body_trailing_idx] - open_values[body_trailing_idx])
        )
        shadow_period_total += (
            abs(close_values[i] - open_values[i])
            - abs(close_values[shadow_trailing_idx] - open_values[shadow_trailing_idx])
        )
        i += 1
        body_trailing_idx += 1
        shadow_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdlhighwave', values=out))

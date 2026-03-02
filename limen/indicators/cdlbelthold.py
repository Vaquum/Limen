import numpy as np
import polars as pl


def cdlbelthold(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Belt-hold candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdlbelthold'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)

    # TA-Lib default candle settings:
    # BodyLong: rangeType=RealBody, avgPeriod=10, factor=1.0
    # ShadowVeryShort: rangeType=HighLow, avgPeriod=10, factor=0.1
    body_long_avg_period = 10
    shadow_vs_avg_period = 10
    shadow_vs_factor = 0.1
    lookback_total = max(body_long_avg_period, shadow_vs_avg_period)

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlbelthold', values=out))

    body_long_period_total = 0.0
    shadow_vs_period_total = 0.0
    body_long_trailing_idx = lookback_total - body_long_avg_period
    shadow_vs_trailing_idx = lookback_total - shadow_vs_avg_period

    i = body_long_trailing_idx
    while i < lookback_total:
        body_long_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = shadow_vs_trailing_idx
    while i < lookback_total:
        shadow_vs_period_total += high_values[i] - low_values[i]
        i += 1

    i = lookback_total
    while i < n:
        real_body = abs(close_values[i] - open_values[i])
        color = 1 if close_values[i] >= open_values[i] else -1
        lower_shadow = min(open_values[i], close_values[i]) - low_values[i]
        upper_shadow = high_values[i] - max(open_values[i], close_values[i])

        body_long_avg = body_long_period_total / body_long_avg_period
        shadow_vs_avg = shadow_vs_factor * (shadow_vs_period_total / shadow_vs_avg_period)

        if real_body > body_long_avg and (
            (color == 1 and lower_shadow < shadow_vs_avg)
            or (color == -1 and upper_shadow < shadow_vs_avg)
        ):
            out[i] = color * 100

        body_long_period_total += (
            abs(close_values[i] - open_values[i])
            - abs(close_values[body_long_trailing_idx] - open_values[body_long_trailing_idx])
        )
        shadow_vs_period_total += (
            (high_values[i] - low_values[i])
            - (high_values[shadow_vs_trailing_idx] - low_values[shadow_vs_trailing_idx])
        )

        i += 1
        body_long_trailing_idx += 1
        shadow_vs_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdlbelthold', values=out))

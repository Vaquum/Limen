import numpy as np
import polars as pl


def cdlshortline(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Short Line Candle pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdlshortline'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)

    # TA-Lib default candle settings:
    # BodyShort: rangeType=RealBody, avgPeriod=10, factor=1.0
    # ShadowShort: rangeType=Shadows, avgPeriod=10, factor=1.0
    body_short_avg_period = 10
    shadow_short_avg_period = 10
    lookback_total = max(body_short_avg_period, shadow_short_avg_period)

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlshortline', values=out))

    body_period_total = 0.0
    shadow_period_total = 0.0
    body_trailing_idx = lookback_total - body_short_avg_period
    shadow_trailing_idx = lookback_total - shadow_short_avg_period

    i = body_trailing_idx
    while i < lookback_total:
        body_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = shadow_trailing_idx
    while i < lookback_total:
        upper_shadow = high_values[i] - max(open_values[i], close_values[i])
        lower_shadow = min(open_values[i], close_values[i]) - low_values[i]
        shadow_period_total += upper_shadow + lower_shadow
        i += 1

    i = lookback_total
    while i < n:
        real_body = abs(close_values[i] - open_values[i])
        upper_shadow = high_values[i] - max(open_values[i], close_values[i])
        lower_shadow = min(open_values[i], close_values[i]) - low_values[i]
        color = 1 if close_values[i] >= open_values[i] else -1

        body_short_avg = body_period_total / body_short_avg_period
        # ShadowShort uses Shadows range type, TA_CANDLEAVERAGE divides by 2.0.
        shadow_short_avg = (shadow_period_total / shadow_short_avg_period) / 2.0

        if (
            real_body < body_short_avg
            and upper_shadow < shadow_short_avg
            and lower_shadow < shadow_short_avg
        ):
            out[i] = color * 100

        body_period_total += (
            abs(close_values[i] - open_values[i])
            - abs(close_values[body_trailing_idx] - open_values[body_trailing_idx])
        )
        shadow_period_total += (
            (high_values[i] - max(open_values[i], close_values[i]))
            + (min(open_values[i], close_values[i]) - low_values[i])
            - (high_values[shadow_trailing_idx] - max(open_values[shadow_trailing_idx], close_values[shadow_trailing_idx]))
            - (min(open_values[shadow_trailing_idx], close_values[shadow_trailing_idx]) - low_values[shadow_trailing_idx])
        )
        i += 1
        body_trailing_idx += 1
        shadow_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdlshortline', values=out))

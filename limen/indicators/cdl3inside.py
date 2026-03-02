import numpy as np
import polars as pl


def cdl3inside(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Three Inside Up/Down candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdl3inside'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)

    # TA-Lib default candle settings:
    # BodyLong: rangeType=RealBody, avgPeriod=10, factor=1.0
    # BodyShort: rangeType=RealBody, avgPeriod=10, factor=1.0
    body_long_avg_period = 10
    body_short_avg_period = 10
    lookback_total = max(body_short_avg_period, body_long_avg_period) + 2

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdl3inside', values=out))

    body_long_period_total = 0.0
    body_short_period_total = 0.0
    body_long_trailing_idx = lookback_total - 2 - body_long_avg_period
    body_short_trailing_idx = lookback_total - 1 - body_short_avg_period

    i = body_long_trailing_idx
    while i < lookback_total - 2:
        body_long_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = body_short_trailing_idx
    while i < lookback_total - 1:
        body_short_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = lookback_total
    while i < n:
        real_body_i2 = abs(close_values[i - 2] - open_values[i - 2])
        real_body_i1 = abs(close_values[i - 1] - open_values[i - 1])
        body_long_avg_i2 = body_long_period_total / body_long_avg_period
        body_short_avg_i1 = body_short_period_total / body_short_avg_period

        max_body_i1 = max(close_values[i - 1], open_values[i - 1])
        min_body_i1 = min(close_values[i - 1], open_values[i - 1])
        max_body_i2 = max(close_values[i - 2], open_values[i - 2])
        min_body_i2 = min(close_values[i - 2], open_values[i - 2])

        color_i2 = 1 if close_values[i - 2] >= open_values[i - 2] else -1
        color_i0 = 1 if close_values[i] >= open_values[i] else -1

        if (
            real_body_i2 > body_long_avg_i2
            and real_body_i1 <= body_short_avg_i1
            and max_body_i1 < max_body_i2
            and min_body_i1 > min_body_i2
            and (
                (color_i2 == 1 and color_i0 == -1 and close_values[i] < open_values[i - 2])
                or (color_i2 == -1 and color_i0 == 1 and close_values[i] > open_values[i - 2])
            )
        ):
            out[i] = -color_i2 * 100

        body_long_period_total += (
            abs(close_values[i - 2] - open_values[i - 2])
            - abs(close_values[body_long_trailing_idx] - open_values[body_long_trailing_idx])
        )
        body_short_period_total += (
            abs(close_values[i - 1] - open_values[i - 1])
            - abs(close_values[body_short_trailing_idx] - open_values[body_short_trailing_idx])
        )

        i += 1
        body_long_trailing_idx += 1
        body_short_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdl3inside', values=out))

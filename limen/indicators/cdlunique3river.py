import numpy as np
import polars as pl

CDLUNIQUE3RIVER_BODY_LONG_AVG_PERIOD = 10
CDLUNIQUE3RIVER_BODY_LONG_PERIOD_TOTAL = 0.0
CDLUNIQUE3RIVER_BODY_SHORT_AVG_PERIOD = 10
CDLUNIQUE3RIVER_BODY_SHORT_PERIOD_TOTAL = 0.0


def cdlunique3river(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Unique 3 River candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdlunique3river'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)


    body_long_avg_period = CDLUNIQUE3RIVER_BODY_LONG_AVG_PERIOD
    body_short_avg_period = CDLUNIQUE3RIVER_BODY_SHORT_AVG_PERIOD
    lookback_total = max(body_short_avg_period, body_long_avg_period) + 2

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlunique3river', values=out))

    body_long_period_total = CDLUNIQUE3RIVER_BODY_LONG_PERIOD_TOTAL
    body_short_period_total = CDLUNIQUE3RIVER_BODY_SHORT_PERIOD_TOTAL
    body_long_trailing_idx = lookback_total - 2 - body_long_avg_period
    body_short_trailing_idx = lookback_total - body_short_avg_period

    i = body_long_trailing_idx
    while i < lookback_total - 2:
        body_long_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = body_short_trailing_idx
    while i < lookback_total:
        body_short_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = lookback_total
    while i < n:
        color_i2 = 1 if close_values[i - 2] >= open_values[i - 2] else -1
        color_i1 = 1 if close_values[i - 1] >= open_values[i - 1] else -1
        color_i0 = 1 if close_values[i] >= open_values[i] else -1

        real_body_i2 = abs(close_values[i - 2] - open_values[i - 2])
        real_body_i0 = abs(close_values[i] - open_values[i])
        body_long_avg_i2 = body_long_period_total / body_long_avg_period
        body_short_avg_i0 = body_short_period_total / body_short_avg_period

        if (
            real_body_i2 > body_long_avg_i2
            and color_i2 == -1
            and color_i1 == -1
            and close_values[i - 1] > close_values[i - 2]
            and open_values[i - 1] <= open_values[i - 2]
            and low_values[i - 1] < low_values[i - 2]
            and real_body_i0 < body_short_avg_i0
            and color_i0 == 1
            and open_values[i] > low_values[i - 1]
        ):
            out[i] = 100

        body_long_period_total += (
            abs(close_values[i - 2] - open_values[i - 2])
            - abs(close_values[body_long_trailing_idx] - open_values[body_long_trailing_idx])
        )
        body_short_period_total += (
            abs(close_values[i] - open_values[i])
            - abs(close_values[body_short_trailing_idx] - open_values[body_short_trailing_idx])
        )

        i += 1
        body_long_trailing_idx += 1
        body_short_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdlunique3river', values=out))

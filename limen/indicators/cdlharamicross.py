import numpy as np
import polars as pl


def cdlharamicross(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Harami Cross candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdlharamicross'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)

    # TA-Lib default candle settings:
    # BodyLong: rangeType=RealBody, avgPeriod=10, factor=1.0
    # BodyDoji: rangeType=HighLow, avgPeriod=10, factor=0.1
    body_long_avg_period = 10
    body_doji_avg_period = 10
    body_doji_factor = 0.1
    lookback_total = max(body_doji_avg_period, body_long_avg_period) + 1

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlharamicross', values=out))

    body_doji_period_total = 0.0
    body_long_period_total = 0.0
    body_long_trailing_idx = lookback_total - 1 - body_long_avg_period
    body_doji_trailing_idx = lookback_total - body_doji_avg_period

    i = body_long_trailing_idx
    while i < lookback_total - 1:
        body_long_period_total += abs(close_values[i] - open_values[i])
        i += 1

    i = body_doji_trailing_idx
    while i < lookback_total:
        body_doji_period_total += high_values[i] - low_values[i]
        i += 1

    i = lookback_total
    while i < n:
        real_body_i1 = abs(close_values[i - 1] - open_values[i - 1])
        real_body_i0 = abs(close_values[i] - open_values[i])
        body_long_avg_i1 = body_long_period_total / body_long_avg_period
        body_doji_avg_i0 = body_doji_factor * (body_doji_period_total / body_doji_avg_period)

        if real_body_i1 > body_long_avg_i1 and real_body_i0 <= body_doji_avg_i0:
            high_i0 = max(close_values[i], open_values[i])
            low_i0 = min(close_values[i], open_values[i])
            high_i1 = max(close_values[i - 1], open_values[i - 1])
            low_i1 = min(close_values[i - 1], open_values[i - 1])
            color_i1 = 1 if close_values[i - 1] >= open_values[i - 1] else -1

            if high_i0 < high_i1 and low_i0 > low_i1:
                out[i] = -color_i1 * 100
            elif high_i0 <= high_i1 and low_i0 >= low_i1:
                out[i] = -color_i1 * 80

        body_long_period_total += (
            abs(close_values[i - 1] - open_values[i - 1])
            - abs(close_values[body_long_trailing_idx] - open_values[body_long_trailing_idx])
        )
        body_doji_period_total += (
            (high_values[i] - low_values[i])
            - (high_values[body_doji_trailing_idx] - low_values[body_doji_trailing_idx])
        )
        i += 1
        body_long_trailing_idx += 1
        body_doji_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdlharamicross', values=out))

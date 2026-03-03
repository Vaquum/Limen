import numpy as np
import polars as pl


def cdlonneck(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute On-Neck candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'cdlonneck'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)

    # TA-Lib default candle settings:
    # Equal: rangeType=HighLow, avgPeriod=5, factor=0.05
    # BodyLong: rangeType=RealBody, avgPeriod=10, factor=1.0
    equal_avg_period = 5
    equal_factor = 0.05
    body_long_avg_period = 10
    lookback_total = max(equal_avg_period, body_long_avg_period) + 1

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdlonneck', values=out))

    equal_period_total = 0.0
    body_long_period_total = 0.0
    equal_trailing_idx = lookback_total - equal_avg_period
    body_long_trailing_idx = lookback_total - body_long_avg_period

    i = equal_trailing_idx
    while i < lookback_total:
        equal_period_total += high_values[i - 1] - low_values[i - 1]
        i += 1

    i = body_long_trailing_idx
    while i < lookback_total:
        body_long_period_total += abs(close_values[i - 1] - open_values[i - 1])
        i += 1

    i = lookback_total
    while i < n:
        color_i1 = 1 if close_values[i - 1] >= open_values[i - 1] else -1
        color_i0 = 1 if close_values[i] >= open_values[i] else -1
        real_body_i1 = abs(close_values[i - 1] - open_values[i - 1])
        body_long_avg_i1 = body_long_period_total / body_long_avg_period
        equal_avg_i1 = equal_factor * (equal_period_total / equal_avg_period)

        if (
            color_i1 == -1
            and real_body_i1 > body_long_avg_i1
            and color_i0 == 1
            and open_values[i] < low_values[i - 1]
            and close_values[i] <= low_values[i - 1] + equal_avg_i1
            and close_values[i] >= low_values[i - 1] - equal_avg_i1
        ):
            out[i] = -100

        equal_period_total += (
            (high_values[i - 1] - low_values[i - 1])
            - (high_values[equal_trailing_idx - 1] - low_values[equal_trailing_idx - 1])
        )
        body_long_period_total += (
            abs(close_values[i - 1] - open_values[i - 1])
            - abs(close_values[body_long_trailing_idx - 1] - open_values[body_long_trailing_idx - 1])
        )
        i += 1
        equal_trailing_idx += 1
        body_long_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdlonneck', values=out))

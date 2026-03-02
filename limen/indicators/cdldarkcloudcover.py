import numpy as np
import polars as pl


def cdldarkcloudcover(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    penetration: float = 0.5,
) -> pl.DataFrame:

    '''
    Compute Dark Cloud Cover candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices
        penetration (float): Percentage of penetration of the 2nd candle into the 1st real body

    Returns:
        pl.DataFrame: The input data with a new column 'cdldarkcloudcover'
    '''

    if penetration < 0.0:
        raise ValueError('penetration must be >= 0')

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)

    # TA-Lib default candle setting for BodyLong:
    # rangeType=RealBody, avgPeriod=10, factor=1.0
    body_long_avg_period = 10
    lookback_total = body_long_avg_period + 1

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='cdldarkcloudcover', values=out))

    body_long_period_total = 0.0
    body_long_trailing_idx = lookback_total - body_long_avg_period

    i = body_long_trailing_idx
    while i < lookback_total:
        body_long_period_total += abs(close_values[i - 1] - open_values[i - 1])
        i += 1

    i = lookback_total
    while i < n:
        real_body_i1 = abs(close_values[i - 1] - open_values[i - 1])
        body_long_avg_i1 = body_long_period_total / body_long_avg_period

        color_i1 = 1 if close_values[i - 1] >= open_values[i - 1] else -1
        color_i0 = 1 if close_values[i] >= open_values[i] else -1

        if (
            color_i1 == 1
            and real_body_i1 > body_long_avg_i1
            and color_i0 == -1
            and open_values[i] > high_values[i - 1]
            and close_values[i] > open_values[i - 1]
            and close_values[i] < close_values[i - 1] - (real_body_i1 * penetration)
        ):
            out[i] = -100

        body_long_period_total += (
            abs(close_values[i - 1] - open_values[i - 1])
            - abs(close_values[body_long_trailing_idx - 1] - open_values[body_long_trailing_idx - 1])
        )
        i += 1
        body_long_trailing_idx += 1

    return data.with_columns(pl.Series(name='cdldarkcloudcover', values=out))

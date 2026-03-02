import numpy as np
import polars as pl


def coldoji(
    data: pl.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Doji candlestick pattern.

    Args:
        data (pl.DataFrame): Dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'coldoji'
    '''

    open_values = data[open_col].to_numpy().astype(float, copy=False)
    high_values = data[high_col].to_numpy().astype(float, copy=False)
    low_values = data[low_col].to_numpy().astype(float, copy=False)
    close_values = data[close_col].to_numpy().astype(float, copy=False)
    n = len(data)

    # TA-Lib default candle setting for BodyDoji:
    # rangeType=HighLow, avgPeriod=10, factor=0.1
    body_doji_avg_period = 10
    body_doji_factor = 0.1
    lookback_total = body_doji_avg_period

    out = np.zeros(n, dtype=np.int32)
    if n <= lookback_total:
        return data.with_columns(pl.Series(name='coldoji', values=out))

    body_doji_period_total = 0.0
    body_doji_trailing_idx = lookback_total - body_doji_avg_period

    i = body_doji_trailing_idx
    while i < lookback_total:
        body_doji_period_total += high_values[i] - low_values[i]
        i += 1

    i = lookback_total
    while i < n:
        real_body = abs(close_values[i] - open_values[i])
        body_doji_avg = body_doji_factor * (body_doji_period_total / body_doji_avg_period)

        if real_body <= body_doji_avg:
            out[i] = 100

        body_doji_period_total += (
            (high_values[i] - low_values[i])
            - (high_values[body_doji_trailing_idx] - low_values[body_doji_trailing_idx])
        )
        i += 1
        body_doji_trailing_idx += 1

    return data.with_columns(pl.Series(name='coldoji', values=out))

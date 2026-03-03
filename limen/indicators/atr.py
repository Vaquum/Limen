import polars as pl
import numpy as np


def atr(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    period: int = 14,
) -> pl.DataFrame:

    '''
    Compute Average True Range (ATR).

    Args:
        data (pl.DataFrame): Klines dataset with high/low/close columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices
        period (int): Number of periods for ATR calculation

    Returns:
        pl.DataFrame: The input data with a new column 'atr_{period}'
    '''

    if period < 1 or period > 100000:
        raise ValueError('period must be between 1 and 100000')

    high = data[high_col].to_numpy()
    low = data[low_col].to_numpy()
    close = data[close_col].to_numpy()
    n = len(data)

    out = np.full(n, np.nan, dtype=float)
    if n <= 1:
        return data.with_columns(pl.Series(name=f"atr_{period}", values=out))

    tr = np.full(n, np.nan, dtype=float)
    prev_close = close[:-1]
    tr[1:] = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - prev_close),
        np.abs(low[1:] - prev_close),
    ])

    if period == 1:
        out[1:] = tr[1:]
        return data.with_columns(pl.Series(name=f"atr_{period}", values=out))

    if n <= period:
        return data.with_columns(pl.Series(name=f"atr_{period}", values=out))

    prev_atr = tr[1:period + 1].mean()
    out[period] = prev_atr

    for i in range(period + 1, n):
        prev_atr = ((prev_atr * (period - 1)) + tr[i]) / period
        out[i] = prev_atr

    return data.with_columns(pl.Series(name=f"atr_{period}", values=out))

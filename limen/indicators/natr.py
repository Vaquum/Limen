import numpy as np
import polars as pl


def natr(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    period: int = 14,
) -> pl.DataFrame:

    '''
    Compute Normalized Average True Range (NATR).

    Args:
        data (pl.DataFrame): Klines dataset with high/low/close columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices
        period (int): Number of periods for NATR calculation (>= 1)

    Returns:
        pl.DataFrame: The input data with a new column 'natr_{period}'
    '''

    if period < 1:
        raise ValueError('period must be >= 1')

    high = data[high_col].to_numpy()
    low = data[low_col].to_numpy()
    close = data[close_col].to_numpy()
    n = len(data)

    out = np.full(n, np.nan, dtype=float)
    if n <= 1:
        return data.with_columns(pl.Series(name=f'natr_{period}', values=out))

    tr = np.full(n, np.nan, dtype=float)
    prev_close = close[:-1]
    tr[1:] = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - prev_close),
        np.abs(low[1:] - prev_close),
    ])

    if period == 1:
        out[1:] = tr[1:]
        return data.with_columns(pl.Series(name=f'natr_{period}', values=out))

    if n <= period:
        return data.with_columns(pl.Series(name=f'natr_{period}', values=out))

    prev_atr = tr[1:period + 1].mean()
    out[period] = prev_atr

    for i in range(period + 1, n):
        prev_atr = ((prev_atr * (period - 1)) + tr[i]) / period
        out[i] = prev_atr

    valid = ~np.isnan(out)
    non_zero_close = valid & (close != 0.0)
    zero_close = valid & (close == 0.0)

    out[non_zero_close] = (out[non_zero_close] / close[non_zero_close]) * 100.0
    out[zero_close] = 0.0

    return data.with_columns(pl.Series(name=f'natr_{period}', values=out))

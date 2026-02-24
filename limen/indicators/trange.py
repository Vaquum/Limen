import numpy as np
import polars as pl


def trange(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute True Range (TRANGE).

    Args:
        data (pl.DataFrame): Klines dataset with high/low/close columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'trange'
    '''

    high = data[high_col].to_numpy()
    low = data[low_col].to_numpy()
    close = data[close_col].to_numpy()
    n = len(data)

    out = np.full(n, np.nan, dtype=float)
    if n <= 1:
        return data.with_columns(pl.Series(name='trange', values=out))

    prev_close = close[:-1]
    out[1:] = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - prev_close),
        np.abs(low[1:] - prev_close),
    ])

    return data.with_columns(pl.Series(name='trange', values=out))

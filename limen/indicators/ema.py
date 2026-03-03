import numpy as np
import polars as pl

from limen.indicators._ema import _ema_talib_default_segment


def ema(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 30,
) -> pl.DataFrame:

    '''
    Compute Exponential Moving Average (EMA).

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods

    Returns:
        pl.DataFrame: The input data with a new column 'ema_{period}'
    '''

    if period < 2 or period > 100000:
        raise ValueError('period must be between 2 and 100000')

    values = data[price_col].to_numpy().astype(float, copy=False)
    n = len(values)
    out_col = f'ema_{period}'
    out = np.full(n, np.nan, dtype=float)

    lookback = period - 1
    if n <= lookback:
        return data.with_columns(pl.Series(name=out_col, values=out))

    start_idx = lookback
    end_idx = n - 1
    _, ema_values = _ema_talib_default_segment(values, period, start_idx, end_idx)
    out[start_idx:start_idx + len(ema_values)] = ema_values

    return data.with_columns(pl.Series(name=out_col, values=out))

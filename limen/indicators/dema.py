import numpy as np
import polars as pl

from limen.indicators.ema import _ema_talib_default_segment


def dema(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 30,
) -> pl.DataFrame:

    '''
    Compute Double Exponential Moving Average (DEMA).

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods

    Returns:
        pl.DataFrame: The input data with a new column 'dema_{period}'
    '''

    if period < 2:
        raise ValueError('period must be >= 2')

    values = data[price_col].to_numpy().astype(float, copy=False)
    n = len(values)
    out_col = f'dema_{period}'
    out = np.full(n, np.nan, dtype=float)

    lookback_ema = period - 1
    lookback_total = lookback_ema * 2

    if n <= lookback_total:
        return data.with_columns(pl.Series(name=out_col, values=out))

    start_idx = lookback_total
    end_idx = n - 1

    _, first_ema = _ema_talib_default_segment(values, period, start_idx - lookback_ema, end_idx)
    if first_ema.size == 0:
        return data.with_columns(pl.Series(name=out_col, values=out))

    _, second_ema = _ema_talib_default_segment(first_ema, period, 0, len(first_ema) - 1)
    if second_ema.size == 0:
        return data.with_columns(pl.Series(name=out_col, values=out))

    dema_values = (2.0 * first_ema[lookback_ema:]) - second_ema
    out_count = len(dema_values)
    out[start_idx:start_idx + out_count] = dema_values

    return data.with_columns(pl.Series(name=out_col, values=out))

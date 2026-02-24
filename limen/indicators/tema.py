import numpy as np
import polars as pl

from limen.indicators._ema import _ema_talib_default_segment


def tema(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 30,
) -> pl.DataFrame:

    '''
    Compute Triple Exponential Moving Average (TEMA).

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods

    Returns:
        pl.DataFrame: The input data with a new column 'tema_{period}'
    '''

    if period < 2 or period > 100000:
        raise ValueError('period must be between 2 and 100000')

    values = data[price_col].to_numpy().astype(float, copy=False)
    n = len(values)
    out_col = f'tema_{period}'
    out = np.full(n, np.nan, dtype=float)

    lookback_ema = period - 1
    lookback_total = lookback_ema * 3
    if n <= lookback_total:
        return data.with_columns(pl.Series(name=out_col, values=out))

    _, first_ema = _ema_talib_default_segment(values, period, lookback_ema, n - 1)
    if first_ema.size == 0:
        return data.with_columns(pl.Series(name=out_col, values=out))

    _, second_ema = _ema_talib_default_segment(first_ema, period, 0, len(first_ema) - 1)
    if second_ema.size == 0:
        return data.with_columns(pl.Series(name=out_col, values=out))

    _, third_ema = _ema_talib_default_segment(second_ema, period, 0, len(second_ema) - 1)
    if third_ema.size == 0:
        return data.with_columns(pl.Series(name=out_col, values=out))

    tema_values = third_ema + (3.0 * first_ema[lookback_ema * 2:]) - (3.0 * second_ema[lookback_ema:])
    out[lookback_total:lookback_total + len(tema_values)] = tema_values

    return data.with_columns(pl.Series(name=out_col, values=out))

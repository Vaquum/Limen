import numpy as np
import polars as pl

from limen.indicators.dema import dema
from limen.indicators.ema import ema
from limen.indicators.kama import kama
from limen.indicators.mama import mama
from limen.indicators.sma import sma
from limen.indicators.t3 import t3
from limen.indicators.tema import tema
from limen.indicators.trima import trima
from limen.indicators.wma import wma


def ma(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 30,
    ma_type: int = 0,
) -> pl.DataFrame:

    '''
    Compute Moving Average with selectable MA type (TA-Lib MA).

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods
        ma_type (int): TA-Lib MA type

    Returns:
        pl.DataFrame: The input data with a new column 'ma_{period}_{ma_type}'
    '''

    if period < 1:
        raise ValueError('period must be >= 1')
    if ma_type < 0 or ma_type > 8:
        raise ValueError('ma_type must be between 0 and 8')

    values = data[price_col].to_numpy().astype(float, copy=False)
    n = len(values)
    out_col = f'ma_{period}_{ma_type}'
    out = np.full(n, np.nan, dtype=float)

    if n == 0:
        return data.with_columns(pl.Series(name=out_col, values=out))

    # TA-Lib MA: period=1 copies input regardless of ma_type.
    if period == 1:
        out[:] = values
        return data.with_columns(pl.Series(name=out_col, values=out))

    if ma_type == 0:
        out = sma(data, price_col=price_col, period=period)[f'sma_{period}'].to_numpy().astype(float, copy=False)
    elif ma_type == 1:
        out = ema(data, price_col=price_col, period=period)[f'ema_{period}'].to_numpy().astype(float, copy=False)
    elif ma_type == 2:
        out = wma(data, price_col=price_col, period=period)[f'wma_{period}'].to_numpy().astype(float, copy=False)
    elif ma_type == 3:
        dema_col = f'dema_{period}'
        out = dema(data, price_col=price_col, period=period)[dema_col].to_numpy().astype(float, copy=False)
    elif ma_type == 4:
        out = tema(data, price_col=price_col, period=period)[f'tema_{period}'].to_numpy().astype(float, copy=False)
    elif ma_type == 5:
        out = trima(data, price_col=price_col, period=period)[f'trima_{period}'].to_numpy().astype(float, copy=False)
    elif ma_type == 6:
        kama_col = f'kama_{period}'
        out = kama(data, price_col=price_col, period=period)[kama_col].to_numpy().astype(float, copy=False)
    elif ma_type == 7:
        # TA-Lib MA(MAMA) ignores period and uses default fast/slow limits.
        out = mama(data, price_col=price_col, fast_limit=0.5, slow_limit=0.05)['mama'].to_numpy().astype(float, copy=False)
    elif ma_type == 8:
        out = t3(data, price_col=price_col, period=period, vfactor=0.7)[f't3_{period}_0.7'].to_numpy().astype(float, copy=False)

    return data.with_columns(pl.Series(name=out_col, values=out))

import numpy as np
import polars as pl

from limen.indicators.ema import _ema_talib_default_segment


CMP_N_100000 = 100000
CMP_N_2 = 2

def _dema_from_values(values: np.ndarray, period: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)

    lookback_ema = period - 1
    lookback_total = lookback_ema * 2
    if n <= lookback_total:
        return out

    start_idx = lookback_total
    end_idx = n - 1

    _, first_ema = _ema_talib_default_segment(values, period, start_idx - lookback_ema, end_idx)
    if first_ema.size == 0:
        return out

    _, second_ema = _ema_talib_default_segment(first_ema, period, 0, len(first_ema) - 1)
    if second_ema.size == 0:
        return out

    dema_values = (2.0 * first_ema[lookback_ema:]) - second_ema
    out_count = len(dema_values)
    out[start_idx:start_idx + out_count] = dema_values
    return out


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

    if period < CMP_N_2 or period > CMP_N_100000:
        raise ValueError('dema period must be between 2 and 100000')

    out_col = f"dema_{period}"
    frame = data
    dema_expr = pl.col(price_col).map_batches(
        lambda s: pl.Series(
            _dema_from_values(
                s.to_numpy().astype(float, copy=False),
                period,
            )
        ),
        return_dtype=pl.Float64,
    ).alias(out_col)

    return frame.with_columns(dema_expr)

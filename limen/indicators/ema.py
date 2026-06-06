import numpy as np
import polars as pl

from limen.indicators._ema import _ema_talib_default_segment


CMP_N_100000 = 100000
CMP_N_2 = 2

def _ema_from_values(values: np.ndarray, period: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    lookback = period - 1
    if n <= lookback:
        return out

    start_idx = lookback
    end_idx = n - 1
    _, ema_values = _ema_talib_default_segment(values, period, start_idx, end_idx)
    out[start_idx:start_idx + len(ema_values)] = ema_values
    return out


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

    if period < CMP_N_2 or period > CMP_N_100000:
        raise ValueError('ema period must be between 2 and 100000')

    out_col = f"ema_{period}"
    frame = data
    ema_expr = pl.col(price_col).map_batches(
        lambda s: pl.Series(
            _ema_from_values(
                s.to_numpy().astype(float, copy=False),
                period,
            )
        ),
        return_dtype=pl.Float64,
    ).alias(out_col)

    return frame.with_columns(ema_expr)

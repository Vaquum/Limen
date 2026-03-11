import numpy as np
import polars as pl

from limen.indicators._ema import _ema_talib_default_segment


CMP_N_100000 = 100000

def _trix_from_values(values: np.ndarray, period: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)

    ema_lookback = period - 1
    total_lookback = (ema_lookback * 3) + 1
    if n <= total_lookback:
        return out

    start_idx = total_lookback
    end_idx = n - 1

    _, temp = _ema_talib_default_segment(values, period, start_idx - total_lookback, end_idx)
    nb_element_to_output = n - 1

    nb_element_to_output -= ema_lookback
    _, temp = _ema_talib_default_segment(temp, period, 0, nb_element_to_output)

    nb_element_to_output -= ema_lookback
    _, temp = _ema_talib_default_segment(temp, period, 0, nb_element_to_output)

    nb_element_to_output -= ema_lookback
    roc_vals = np.zeros(nb_element_to_output, dtype=float)
    for i in range(1, nb_element_to_output + 1):
        prev = temp[i - 1]
        if prev != 0.0:
            roc_vals[i - 1] = ((temp[i] / prev) - 1.0) * 100.0
        else:
            roc_vals[i - 1] = 0.0

    out[start_idx:start_idx + len(roc_vals)] = roc_vals
    return out


def trix(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 30,
) -> pl.DataFrame:

    '''
    Compute TRIX: 1-day ROC of a triple-smoothed EMA.

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods (1..100000)

    Returns:
        pl.DataFrame: The input data with a new column 'trix_{period}'
    '''

    if period < 1 or period > CMP_N_100000:
        raise ValueError('period must be between 1 and 100000')

    out_col = f'trix_{period}'
    frame = data
    trix_expr = pl.col(price_col).map_batches(
        lambda s: pl.Series(
            _trix_from_values(
                s.to_numpy().astype(float, copy=False),
                period,
            )
        ),
        return_dtype=pl.Float64,
    ).alias(out_col)

    return frame.with_columns(trix_expr)

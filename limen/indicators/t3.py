import numpy as np
import polars as pl


CMP_N_100000 = 100000
CMP_N_2 = 2

def _t3_from_values(values: np.ndarray, period: int, vfactor: float) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)

    lookback_total = 6 * (period - 1)
    if n <= lookback_total:
        return out

    start_idx = lookback_total
    end_idx = n - 1
    today = start_idx - lookback_total

    k = 2.0 / (period + 1.0)
    one_minus_k = 1.0 - k

    temp_real = values[today]
    today += 1
    i = period - 1
    while i > 0:
        temp_real += values[today]
        today += 1
        i -= 1
    e1 = temp_real / period

    temp_real = e1
    i = period - 1
    while i > 0:
        e1 = (k * values[today]) + (one_minus_k * e1)
        today += 1
        temp_real += e1
        i -= 1
    e2 = temp_real / period

    temp_real = e2
    i = period - 1
    while i > 0:
        e1 = (k * values[today]) + (one_minus_k * e1)
        today += 1
        e2 = (k * e1) + (one_minus_k * e2)
        temp_real += e2
        i -= 1
    e3 = temp_real / period

    temp_real = e3
    i = period - 1
    while i > 0:
        e1 = (k * values[today]) + (one_minus_k * e1)
        today += 1
        e2 = (k * e1) + (one_minus_k * e2)
        e3 = (k * e2) + (one_minus_k * e3)
        temp_real += e3
        i -= 1
    e4 = temp_real / period

    temp_real = e4
    i = period - 1
    while i > 0:
        e1 = (k * values[today]) + (one_minus_k * e1)
        today += 1
        e2 = (k * e1) + (one_minus_k * e2)
        e3 = (k * e2) + (one_minus_k * e3)
        e4 = (k * e3) + (one_minus_k * e4)
        temp_real += e4
        i -= 1
    e5 = temp_real / period

    temp_real = e5
    i = period - 1
    while i > 0:
        e1 = (k * values[today]) + (one_minus_k * e1)
        today += 1
        e2 = (k * e1) + (one_minus_k * e2)
        e3 = (k * e2) + (one_minus_k * e3)
        e4 = (k * e3) + (one_minus_k * e4)
        e5 = (k * e4) + (one_minus_k * e5)
        temp_real += e5
        i -= 1
    e6 = temp_real / period

    while today <= start_idx:
        e1 = (k * values[today]) + (one_minus_k * e1)
        today += 1
        e2 = (k * e1) + (one_minus_k * e2)
        e3 = (k * e2) + (one_minus_k * e3)
        e4 = (k * e3) + (one_minus_k * e4)
        e5 = (k * e4) + (one_minus_k * e5)
        e6 = (k * e5) + (one_minus_k * e6)

    temp_real = vfactor * vfactor
    c1 = -(temp_real * vfactor)
    c2 = 3.0 * (temp_real - c1)
    c3 = -6.0 * temp_real - 3.0 * (vfactor - c1)
    c4 = 1.0 + (3.0 * vfactor) - c1 + (3.0 * temp_real)

    out_idx = start_idx
    out[out_idx] = (c1 * e6) + (c2 * e5) + (c3 * e4) + (c4 * e3)
    out_idx += 1

    while today <= end_idx:
        e1 = (k * values[today]) + (one_minus_k * e1)
        today += 1
        e2 = (k * e1) + (one_minus_k * e2)
        e3 = (k * e2) + (one_minus_k * e3)
        e4 = (k * e3) + (one_minus_k * e4)
        e5 = (k * e4) + (one_minus_k * e5)
        e6 = (k * e5) + (one_minus_k * e6)
        out[out_idx] = (c1 * e6) + (c2 * e5) + (c3 * e4) + (c4 * e3)
        out_idx += 1

    return out


def t3(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 5,
    vfactor: float = 0.7,
) -> pl.DataFrame:

    '''
    Compute Triple Exponential Moving Average (T3).

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods
        vfactor (float): Volume factor

    Returns:
        pl.DataFrame: The input data with a new column 't3_{period}_{vfactor}'
    '''

    if period < CMP_N_2 or period > CMP_N_100000:
        raise ValueError('t3 period must be between 2 and 100000')
    if vfactor < 0.0 or vfactor > 1.0:
        raise ValueError('t3 vfactor must be between 0 and 1')

    out_col = f"t3_{period}_{vfactor:g}"
    frame = data
    t3_expr = pl.col(price_col).map_batches(
        lambda s: pl.Series(
            _t3_from_values(
                s.to_numpy().astype(float, copy=False),
                period,
                vfactor,
            )
        ),
        return_dtype=pl.Float64,
    ).alias(out_col)
    return frame.with_columns(t3_expr)

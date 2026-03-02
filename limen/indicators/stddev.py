import numpy as np
import polars as pl

_TA_EPSILON = 1e-14


def stddev(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 5,
    nb_dev: float = 1.0,
) -> pl.DataFrame:

    '''
    Compute Standard Deviation.

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods
        nb_dev (float): Number of deviations to scale the output

    Returns:
        pl.DataFrame: The input data with a new column 'stddev_{period}_{nb_dev:g}'
    '''

    if period < 2 or period > 100000:
        raise ValueError('period must be between 2 and 100000')
    if nb_dev < -3e37 or nb_dev > 3e37:
        raise ValueError('nb_dev must be between -3e37 and 3e37')

    values = data[price_col].to_numpy().astype(float, copy=False)
    n = len(values)
    out_col = f'stddev_{period}_{nb_dev:g}'
    out = np.full(n, np.nan, dtype=float)

    lookback_total = period - 1
    if n <= lookback_total:
        return data.with_columns(pl.Series(name=out_col, values=out))

    start_idx = lookback_total
    end_idx = n - 1

    period_total1 = 0.0
    period_total2 = 0.0
    trailing_idx = start_idx - lookback_total

    i = trailing_idx
    if period > 1:
        while i < start_idx:
            temp_real = values[i]
            i += 1
            period_total1 += temp_real
            period_total2 += temp_real * temp_real

    out_idx = 0
    while i <= end_idx:
        temp_real = values[i]
        i += 1
        period_total1 += temp_real
        period_total2 += temp_real * temp_real

        mean_value1 = period_total1 / period
        mean_value2 = period_total2 / period

        temp_real = values[trailing_idx]
        trailing_idx += 1
        period_total1 -= temp_real
        period_total2 -= temp_real * temp_real

        variance = mean_value2 - (mean_value1 * mean_value1)
        if variance < _TA_EPSILON:
            out[start_idx + out_idx] = 0.0
        elif nb_dev != 1.0:
            out[start_idx + out_idx] = np.sqrt(variance) * nb_dev
        else:
            out[start_idx + out_idx] = np.sqrt(variance)
        out_idx += 1

    return data.with_columns(pl.Series(name=out_col, values=out))

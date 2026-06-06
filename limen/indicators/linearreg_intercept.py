import numpy as np
import polars as pl


CMP_N_100000 = 100000
CMP_N_2 = 2

def _linearreg_intercept_from_values(values: np.ndarray, period: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)

    lookback_total = period - 1
    if n <= lookback_total:
        return out

    start_idx = lookback_total
    end_idx = n - 1

    sum_x = period * (period - 1) * 0.5
    sum_x_sqr = period * (period - 1) * (2 * period - 1) / 6.0
    divisor = (sum_x * sum_x) - (period * sum_x_sqr)

    today = start_idx
    while today <= end_idx:
        sum_xy = 0.0
        sum_y = 0.0

        i = period
        while i != 0:
            i -= 1
            temp_value = values[today - i]
            sum_y += temp_value
            sum_xy += float(i) * temp_value

        m = ((period * sum_xy) - (sum_x * sum_y)) / divisor
        out[today] = (sum_y - (m * sum_x)) / float(period)
        today += 1

    return out


def linearreg_intercept(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 14,
) -> pl.DataFrame:

    '''
    Compute Linear Regression Intercept.

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods

    Returns:
        pl.DataFrame: The input data with a new column 'linearreg_intercept_{period}'
    '''

    if period < CMP_N_2 or period > CMP_N_100000:
        raise ValueError('linearreg_intercept period must be between 2 and 100000')

    out_col = f"linearreg_intercept_{period}"
    return data.with_columns(
        pl.col(price_col).map_batches(
            lambda s: pl.Series(
                _linearreg_intercept_from_values(
                    s.to_numpy().astype(float, copy=False),
                    period,
                )
            ),
            return_dtype=pl.Float64,
        ).alias(out_col)
    )

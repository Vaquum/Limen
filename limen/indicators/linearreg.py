import numpy as np
import polars as pl


def linearreg(
    data: pl.DataFrame,
    price_col: str = 'close',
    period: int = 14,
) -> pl.DataFrame:

    '''
    Compute Linear Regression.

    Args:
        data (pl.DataFrame): Dataset with input price column
        price_col (str): Column name for input price
        period (int): Number of periods

    Returns:
        pl.DataFrame: The input data with a new column 'linearreg_{period}'
    '''

    if period < 2 or period > 100000:
        raise ValueError('period must be between 2 and 100000')

    values = data[price_col].to_numpy().astype(float, copy=False)
    n = len(values)
    out_col = f'linearreg_{period}'
    out = np.full(n, np.nan, dtype=float)

    lookback_total = period - 1
    if n <= lookback_total:
        return data.with_columns(pl.Series(name=out_col, values=out))

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
        b = (sum_y - (m * sum_x)) / float(period)
        out[today] = b + (m * float(period - 1))
        today += 1

    return data.with_columns(pl.Series(name=out_col, values=out))

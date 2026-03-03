<<<<<<< HEAD
import numpy as np
import polars as pl


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
=======
import polars as pl


def linearreg_intercept(data: pl.DataFrame,
                        col: str = 'close',
                        period: int = 14) -> pl.DataFrame:

    '''
    Compute Linear Regression Intercept (LINEARREG_INTERCEPT) indicator.

    Equivalent to TA-Lib LINEARREG_INTERCEPT: the y-intercept of the least-squares
    regression line fitted to each `period`-bar window.

    Uses a vectorised closed-form OLS formula with time indices [0, 1, ..., period-1]:
        slope     = (period * sum_ty - SUM_T * sum_y) / DENOM
        intercept = (sum_y - slope * SUM_T) / period

    where SUM_T, SUM_T2, and DENOM are constants for a fixed window size,
    sum_y is the rolling sum of prices, and sum_ty is derived from the
    cumulative sum using: sum_ty = (period-1)*cum_y - rolling_sum(cum_y.shift(1), period-1).

    Args:
        data (pl.DataFrame): Klines dataset with price column
        col (str): Column name for price data
        period (int): Number of periods for the rolling window
>>>>>>> origin/main

    Returns:
        pl.DataFrame: The input data with a new column 'linearreg_intercept_{period}'
    '''

<<<<<<< HEAD
    if period < 2 or period > 100000:
        raise ValueError('period must be between 2 and 100000')

    values = data[price_col].to_numpy().astype(float, copy=False)
    n = len(values)
    out_col = f'linearreg_intercept_{period}'
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
        out[today] = (sum_y - (m * sum_x)) / float(period)
        today += 1

    return data.with_columns(pl.Series(name=out_col, values=out))
=======
    n = period
    SUM_T = n * (n - 1) / 2
    SUM_T2 = n * (n - 1) * (2 * n - 1) / 6
    DENOM = n * SUM_T2 - SUM_T ** 2

    y = pl.col(col)
    cum_y = y.cum_sum()
    sum_y = y.rolling_sum(window_size=n)
    sum_ty = (n - 1) * cum_y - cum_y.shift(1).rolling_sum(window_size=n - 1)
    slope = (n * sum_ty - SUM_T * sum_y) / DENOM
    intercept = (sum_y - slope * SUM_T) / n

    return data.with_columns([intercept.alias(f"linearreg_intercept_{period}")])
>>>>>>> origin/main

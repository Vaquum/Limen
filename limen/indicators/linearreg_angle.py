import math

import polars as pl


DEGREES_PER_RADIAN = 180.0 / math.pi


def linearreg_angle(data: pl.DataFrame,
                    col: str = 'close',
                    period: int = 14) -> pl.DataFrame:

    '''
    Compute Linear Regression Angle (LINEARREG_ANGLE) indicator.

    Equivalent to TA-Lib LINEARREG_ANGLE: the angle in degrees of the slope of
    the least-squares regression line fitted to each `period`-bar window.

    Computed as: atan(slope) * (180 / pi)

    Uses a vectorised closed-form OLS formula with time indices [0, 1, ..., period-1]:
        slope = (period * sum_ty - SUM_T * sum_y) / DENOM

    where SUM_T, SUM_T2, and DENOM are constants for a fixed window size,
    sum_y is the rolling sum of prices, and sum_ty is derived from the
    cumulative sum using: sum_ty = (period-1)*cum_y - rolling_sum(cum_y.shift(1), period-1).

    Args:
        data (pl.DataFrame): Klines dataset with price column
        col (str): Column name for price data
        period (int): Number of periods for the rolling window

    Returns:
        pl.DataFrame: The input data with a new column 'linearreg_angle_{period}'
    '''

    n = period
    SUM_T = n * (n - 1) / 2
    SUM_T2 = n * (n - 1) * (2 * n - 1) / 6
    DENOM = n * SUM_T2 - SUM_T ** 2

    y = pl.col(col)
    cum_y = y.cum_sum()
    sum_y = y.rolling_sum(window_size=n)
    sum_ty = (n - 1) * cum_y - cum_y.shift(1).rolling_sum(window_size=n - 1)
    slope = (n * sum_ty - SUM_T * sum_y) / DENOM

    return (
        data
        .with_columns([slope.alias('__slope')])
        .with_columns([
            (pl.col('__slope').arctan() * DEGREES_PER_RADIAN).alias(f'linearreg_angle_{period}')
        ])
        .drop('__slope')
    )

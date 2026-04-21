import math

import polars as pl


def parkinson_volatility(
    data: pl.DataFrame,
    window: int = 20,
    high_col: str = 'high',
    low_col: str = 'low',
) -> pl.DataFrame:

    '''
    Compute Parkinson range-based volatility over a rolling window.

    Args:
        data (pl.DataFrame): Klines dataset with high and low price columns
        window (int): Number of periods used for the rolling estimator
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices

    Returns:
        pl.DataFrame: The input data with a new column 'parkinson_volatility'
    '''

    scale = 4.0 * math.log(2.0)

    return (
        data
        .with_columns(
            ((pl.col(high_col).log() - pl.col(low_col).log()) ** 2).alias('_parkinson_log_range_sq')
        )
        .with_columns(
            (
                (pl.col('_parkinson_log_range_sq').rolling_mean(window_size=window) / scale)
                .clip(lower_bound=0.0)
                .sqrt()
            ).alias('parkinson_volatility')
        )
        .drop('_parkinson_log_range_sq')
    )

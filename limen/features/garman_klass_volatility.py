import math

import polars as pl


def garman_klass_volatility(
    data: pl.DataFrame,
    window: int = 20,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Garman-Klass range-based volatility over a rolling window.

    Args:
        data (pl.DataFrame): Klines dataset with open, high, low, and close price columns
        window (int): Number of periods used for the rolling estimator
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'garman_klass_volatility'
    '''

    close_open = pl.col(close_col).log() - pl.col(open_col).log()
    high_low = pl.col(high_col).log() - pl.col(low_col).log()
    variance = (0.5 * (high_low ** 2)) - (((2.0 * math.log(2.0)) - 1.0) * (close_open ** 2))

    return (
        data
        .with_columns(variance.alias('_garman_klass_variance'))
        .with_columns(
            (
                pl.col('_garman_klass_variance')
                .rolling_mean(window_size=window)
                .clip(lower_bound=0.0)
                .sqrt()
            ).alias('garman_klass_volatility')
        )
        .drop('_garman_klass_variance')
    )

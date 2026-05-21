import math

import polars as pl


PARKINSON_SCALE = 4.0 * math.log(2.0)


def _parkinson_variance(high_col: str = 'high', low_col: str = 'low') -> pl.Expr:
    return ((pl.col(high_col).log() - pl.col(low_col).log()) ** 2) / PARKINSON_SCALE


def parkinson_vol_of_vol(
    data: pl.DataFrame,
    window: int = 20,
    high_col: str = 'high',
    low_col: str = 'low',
    output_col: str = 'parkinson_vol_of_vol',
) -> pl.DataFrame:

    '''
    Compute rolling standard deviation of Parkinson variance.

    Args:
        data (pl.DataFrame): Klines dataset with high and low price columns
        window (int): Number of periods in the rolling standard deviation
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with the Parkinson vol-of-vol column appended
    '''

    return data.with_columns(
        _parkinson_variance(high_col, low_col)
        .rolling_std(window_size=window)
        .alias(output_col)
    )

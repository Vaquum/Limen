import math

import polars as pl


PARKINSON_SCALE = 4.0 * math.log(2.0)


def _parkinson_variance(high_col: str = 'high', low_col: str = 'low') -> pl.Expr:
    return ((pl.col(high_col).log() - pl.col(low_col).log()) ** 2) / PARKINSON_SCALE


def volume_volatility_correlation(
    data: pl.DataFrame,
    window: int = 20,
    volume_col: str = 'volume',
    high_col: str = 'high',
    low_col: str = 'low',
    output_col: str = 'volume_volatility_correlation',
) -> pl.DataFrame:

    '''
    Compute rolling correlation between volume and Parkinson variance.

    Args:
        data (pl.DataFrame): Klines dataset with volume, high, and low columns
        window (int): Number of periods in the rolling correlation
        volume_col (str): Column name for traded volume
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with the volume-volatility correlation appended
    '''

    return data.with_columns(
        pl.rolling_corr(
            pl.col(volume_col),
            _parkinson_variance(high_col, low_col),
            window_size=window,
        ).alias(output_col)
    )

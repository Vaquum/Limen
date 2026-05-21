import math

import polars as pl


EPSILON = 1e-10
PARKINSON_SCALE = 4.0 * math.log(2.0)


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(denominator.is_null() | (denominator.abs() <= EPSILON))
        .then(None)
        .otherwise(numerator / denominator)
    )


def _parkinson_variance(high_col: str = 'high', low_col: str = 'low') -> pl.Expr:
    return ((pl.col(high_col).log() - pl.col(low_col).log()) ** 2) / PARKINSON_SCALE


def volatility_spike(
    data: pl.DataFrame,
    window: int = 20,
    high_col: str = 'high',
    low_col: str = 'low',
    output_col: str = 'volatility_spike',
) -> pl.DataFrame:

    '''
    Compute current Parkinson variance relative to its fixed-lag value.

    Args:
        data (pl.DataFrame): Klines dataset with high and low price columns
        window (int): Lag used as the denominator
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with the volatility-spike column appended
    '''

    variance = _parkinson_variance(high_col, low_col)

    return data.with_columns(_safe_divide(variance, variance.shift(window)).alias(output_col))

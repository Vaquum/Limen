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


def volatility_ratio(
    data: pl.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
    high_col: str = 'high',
    low_col: str = 'low',
    output_col: str = 'volatility_ratio',
) -> pl.DataFrame:

    '''
    Compare short and long rolling means of Parkinson variance.

    Args:
        data (pl.DataFrame): Klines dataset with high and low price columns
        short_window (int): Number of periods in the short variance mean
        long_window (int): Number of periods in the long variance mean
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with the volatility-ratio column appended
    '''

    variance = _parkinson_variance(high_col, low_col)

    return data.with_columns(
        _safe_divide(
            variance.rolling_mean(window_size=short_window),
            variance.rolling_mean(window_size=long_window),
        ).alias(output_col)
    )

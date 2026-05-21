import polars as pl


EPSILON = 1e-10


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(denominator.is_null() | (denominator.abs() <= EPSILON))
        .then(None)
        .otherwise(numerator / denominator)
    )


def close_position_rolling(
    data: pl.DataFrame,
    window: int = 20,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    output_col: str = 'close_position_rolling',
) -> pl.DataFrame:

    '''
    Compute rolling mean close position inside the high-low range.

    Args:
        data (pl.DataFrame): Klines dataset with high, low, and close columns
        window (int): Number of periods in the rolling mean
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with the rolling close-position column appended
    '''

    position = _safe_divide(
        pl.col(close_col) - pl.col(low_col),
        pl.col(high_col) - pl.col(low_col),
    )

    return data.with_columns(position.rolling_mean(window_size=window).alias(output_col))

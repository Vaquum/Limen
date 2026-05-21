import polars as pl


EPSILON = 1e-10


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(denominator.is_null() | (denominator.abs() <= EPSILON))
        .then(None)
        .otherwise(numerator / denominator)
    )


def narrow_range(
    data: pl.DataFrame,
    window: int = 20,
    high_col: str = 'high',
    low_col: str = 'low',
    output_col: str = 'narrow_range',
) -> pl.DataFrame:

    '''
    Compute current range relative to the trailing maximum range.

    Args:
        data (pl.DataFrame): Klines dataset with high and low price columns
        window (int): Number of periods in the trailing maximum
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with the narrow-range column appended
    '''

    price_range = pl.col(high_col) - pl.col(low_col)

    return data.with_columns(
        _safe_divide(price_range, price_range.rolling_max(window_size=window)).alias(
            output_col
        )
    )

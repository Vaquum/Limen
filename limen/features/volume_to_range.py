import polars as pl


EPSILON = 1e-10


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(denominator.is_null() | (denominator.abs() <= EPSILON))
        .then(None)
        .otherwise(numerator / denominator)
    )


def volume_to_range(
    data: pl.DataFrame,
    window: int = 20,
    volume_col: str = 'volume',
    high_col: str = 'high',
    low_col: str = 'low',
    output_col: str = 'volume_to_range',
) -> pl.DataFrame:

    '''
    Compute rolling mean volume per unit of high-low range.

    Args:
        data (pl.DataFrame): Klines dataset with volume, high, and low columns
        window (int): Number of periods in the rolling mean
        volume_col (str): Column name for traded volume
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with the volume-to-range column appended
    '''

    return data.with_columns(
        _safe_divide(pl.col(volume_col), pl.col(high_col) - pl.col(low_col))
        .rolling_mean(window_size=window)
        .alias(output_col)
    )

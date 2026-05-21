import polars as pl


EPSILON = 1e-10


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(denominator.is_null() | (denominator.abs() <= EPSILON))
        .then(None)
        .otherwise(numerator / denominator)
    )


def distance_from_ma(
    data: pl.DataFrame,
    window: int = 20,
    close_col: str = 'close',
    output_col: str = 'distance_from_ma',
) -> pl.DataFrame:

    '''
    Compute close distance from its rolling moving average.

    Args:
        data (pl.DataFrame): Klines dataset with a close price column
        window (int): Number of periods in the moving average
        close_col (str): Column name for close prices
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with the moving-average distance column appended
    '''

    moving_average = pl.col(close_col).rolling_mean(window_size=window)

    return data.with_columns(
        _safe_divide(pl.col(close_col) - moving_average, moving_average).alias(output_col)
    )

import polars as pl


EPSILON = 1e-10


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(denominator.is_null() | (denominator.abs() <= EPSILON))
        .then(None)
        .otherwise(numerator / denominator)
    )


def kaufman_efficiency_ratio(
    data: pl.DataFrame,
    window: int = 10,
    close_col: str = 'close',
    output_col: str = 'kaufman_efficiency_ratio',
) -> pl.DataFrame:

    '''
    Compute Kaufman's efficiency ratio over a rolling window.

    Args:
        data (pl.DataFrame): Klines dataset with a close price column
        window (int): Number of periods used for displacement and path length
        close_col (str): Column name for close prices
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with the efficiency-ratio column appended
    '''

    displacement = (pl.col(close_col) - pl.col(close_col).shift(window)).abs()
    path_length = (
        (pl.col(close_col) - pl.col(close_col).shift(1))
        .abs()
        .rolling_sum(window_size=window)
    )

    return data.with_columns(_safe_divide(displacement, path_length).alias(output_col))

import polars as pl


EPSILON = 1e-10


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(denominator.is_null() | (denominator.abs() <= EPSILON))
        .then(None)
        .otherwise(numerator / denominator)
    )


def maker_volume_ratio(
    data: pl.DataFrame,
    window: int = 20,
    maker_volume_col: str = 'maker_volume',
    volume_col: str = 'volume',
    output_col: str = 'maker_volume_ratio',
) -> pl.DataFrame:

    '''
    Compute the rolling mean maker-volume share.

    Args:
        data (pl.DataFrame): Klines dataset with maker volume and total volume columns
        window (int): Number of periods in the rolling mean
        maker_volume_col (str): Column name for maker-side traded volume
        volume_col (str): Column name for total traded volume
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with the rolling maker-volume ratio appended
    '''

    return data.with_columns(
        _safe_divide(pl.col(maker_volume_col), pl.col(volume_col))
        .rolling_mean(window_size=window)
        .alias(output_col)
    )

import polars as pl


EPSILON = 1e-10


def _safe_divide(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(denominator.is_null() | (denominator.abs() <= EPSILON))
        .then(None)
        .otherwise(numerator / denominator)
    )


def maker_volume_share(
    data: pl.DataFrame,
    maker_volume_col: str = 'maker_volume',
    volume_col: str = 'volume',
    output_col: str = 'maker_volume_share',
) -> pl.DataFrame:

    '''
    Compute maker volume as a share of total volume.

    Args:
        data (pl.DataFrame): Klines dataset with maker volume and total volume columns
        maker_volume_col (str): Column name for maker-side traded volume
        volume_col (str): Column name for total traded volume
        output_col (str): Output column name

    Returns:
        pl.DataFrame: The input data with the maker-volume share column appended
    '''

    return data.with_columns(
        _safe_divide(pl.col(maker_volume_col), pl.col(volume_col)).alias(output_col)
    )

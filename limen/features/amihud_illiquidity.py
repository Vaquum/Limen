import polars as pl

EPSILON = 1e-10


def amihud_illiquidity(
    data: pl.DataFrame,
    price_col: str = 'close',
    volume_col: str = 'volume',
) -> pl.DataFrame:

    '''
    Compute the Amihud illiquidity proxy from price changes and dollar volume.

    Args:
        data (pl.DataFrame): Klines dataset with price and volume columns
        price_col (str): Column name used for close-to-close returns
        volume_col (str): Column name for traded volume

    Returns:
        pl.DataFrame: The input data with a new column 'amihud_illiquidity'
    '''

    dollar_volume_expr = pl.col(price_col) * pl.col(volume_col)

    return data.with_columns(
        (
            pl.col(price_col).pct_change().abs() / (dollar_volume_expr + EPSILON)
        ).alias('amihud_illiquidity')
    )

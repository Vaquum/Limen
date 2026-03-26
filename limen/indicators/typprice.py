import polars as pl


def typprice(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Typical Price.

    Args:
        data (pl.DataFrame): Dataset with high, low, and close columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'typprice'
    '''

    typ_price = (
        pl.col(high_col).cast(pl.Float64)
        + pl.col(low_col).cast(pl.Float64)
        + pl.col(close_col).cast(pl.Float64)
    ) / 3.0

    return data.with_columns(typ_price.alias('typprice'))

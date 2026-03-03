import polars as pl


<<<<<<< HEAD
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
=======
def typprice(data: pl.DataFrame,
             high_col: str = 'high',
             low_col: str = 'low',
             close_col: str = 'close') -> pl.DataFrame:

    '''
    Compute Typical Price (TYPPRICE) indicator.

    Equivalent to TA-Lib TYPPRICE: (high + low + close) / 3.

    Args:
        data (pl.DataFrame): Klines dataset with 'high', 'low', 'close' columns
>>>>>>> origin/main
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'typprice'
    '''

<<<<<<< HEAD
    typ_price = (
        pl.col(high_col).cast(pl.Float64)
        + pl.col(low_col).cast(pl.Float64)
        + pl.col(close_col).cast(pl.Float64)
    ) / 3.0

    return data.with_columns(typ_price.alias('typprice'))
=======
    return data.with_columns([
        (
            (pl.col(high_col) + pl.col(low_col) + pl.col(close_col)) / 3
        ).alias('typprice')
    ])
>>>>>>> origin/main

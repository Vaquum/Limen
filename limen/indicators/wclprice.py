import polars as pl


<<<<<<< HEAD
def wclprice(
    data: pl.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
) -> pl.DataFrame:

    '''
    Compute Weighted Close Price.

    Args:
        data (pl.DataFrame): Dataset with high, low, and close columns
=======
def wclprice(data: pl.DataFrame,
             high_col: str = 'high',
             low_col: str = 'low',
             close_col: str = 'close') -> pl.DataFrame:

    '''
    Compute Weighted Close Price (WCLPRICE) indicator.

    Equivalent to TA-Lib WCLPRICE: (high + low + 2 * close) / 4.

    Args:
        data (pl.DataFrame): Klines dataset with 'high', 'low', 'close' columns
>>>>>>> origin/main
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'wclprice'
    '''

<<<<<<< HEAD
    wcl_price = (
        pl.col(high_col).cast(pl.Float64)
        + pl.col(low_col).cast(pl.Float64)
        + (pl.col(close_col).cast(pl.Float64) * 2.0)
    ) / 4.0

    return data.with_columns(wcl_price.alias('wclprice'))
=======
    return data.with_columns([
        (
            (pl.col(high_col) + pl.col(low_col) + 2 * pl.col(close_col)) / 4
        ).alias('wclprice')
    ])
>>>>>>> origin/main

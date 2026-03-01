import polars as pl


def avgprice(data: pl.DataFrame,
             open_col: str = 'open',
             high_col: str = 'high',
             low_col: str = 'low',
             close_col: str = 'close') -> pl.DataFrame:

    '''
    Compute Average Price (AVGPRICE) indicator.

    Equivalent to TA-Lib AVGPRICE: (open + high + low + close) / 4.

    Args:
        data (pl.DataFrame): Klines dataset with OHLC columns
        open_col (str): Column name for open prices
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'avgprice'
    '''

    return data.with_columns([
        (
            (pl.col(open_col) + pl.col(high_col) + pl.col(low_col) + pl.col(close_col)) / 4
        ).alias('avgprice')
    ])

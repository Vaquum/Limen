import polars as pl


def typprice(data: pl.DataFrame,
             high_col: str = 'high',
             low_col: str = 'low',
             close_col: str = 'close') -> pl.DataFrame:

    '''
    Compute Typical Price (TYPPRICE) indicator.

    Equivalent to TA-Lib TYPPRICE: (high + low + close) / 3.

    Args:
        data (pl.DataFrame): Klines dataset with 'high', 'low', 'close' columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices

    Returns:
        pl.DataFrame: The input data with a new column 'typprice'
    '''

    return data.with_columns([
        (
            (pl.col(high_col) + pl.col(low_col) + pl.col(close_col)) / 3
        ).alias('typprice')
    ])

import polars as pl


def window_return(data: pl.DataFrame, period: int = 24) -> pl.DataFrame:

    '''
    Compute windowed return over a given period: close/close.shift(period) - 1.

    Args:
        data (pl.DataFrame): Klines dataset with 'close' column
        period (int): Period for the window return

    Returns:
        pl.DataFrame: The input data with a new column f'ret_{period}'
    '''

    col = f'ret_{period}'
    return data.with_columns(((pl.col('close') / pl.col('close').shift(period)) - 1.0).alias(col))



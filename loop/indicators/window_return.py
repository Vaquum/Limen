import polars as pl


def window_return(data: pl.DataFrame,
                  period: int = 24) -> pl.DataFrame:

    '''
    Compute windowed return close/close.shift(period) - 1 for a given period.

    Args:
        data (pl.DataFrame): Klines dataset with 'close' column
        period (int): Window length for the return

    Returns:
        pl.DataFrame: The input data with a new column named using the pattern 'ret_{period}'
    '''

    col = f'ret_{period}'
    
    return data.with_columns(((pl.col('close') / pl.col('close').shift(period)) - 1.0).alias(col))

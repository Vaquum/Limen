import polars as pl
from loop.features.lag_column import lag_column


def lag_columns(data: pl.DataFrame,
               cols: list[str],
               lag: int) -> pl.DataFrame:
    
    '''
    Compute lagged versions of multiple columns.

    Args:
        data (pl.DataFrame): Klines dataset with specified columns
        cols (list[str]): The list of column names to lag
        lag (int): The number of periods to lag

    Returns:
        pl.DataFrame: The input data with the lagged columns appended

    NOTE:
        Additional input validation is performed by lag_column for each column.
        This function will raise the same errors as lag_column if any column
        fails validation.
    '''
    
    if not cols:
        raise ValueError('cols cannot be empty')

    df = data
    for col in cols:
        df = lag_column(df, col, lag, None)
    return df
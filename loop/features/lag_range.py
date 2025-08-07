import polars as pl
from loop.features.lag_column import lag_column


def lag_range(data: pl.DataFrame,
              col: str,
              start: int,
              end: int) -> pl.DataFrame:
    
    '''
    Compute multiple lagged versions of a column over a range.

    Args:
        data (pl.DataFrame): Klines dataset with specified column
        col (str): The column name to lag
        start (int): The start of lag range (inclusive)
        end (int): The end of lag range (inclusive)

    Returns:
        pl.DataFrame: The input data with the lagged columns appended

    NOTE:
        Input validation is performed by lag_column for each lag value.
        This function will raise the same errors as lag_column if any lag
        fails validation.
    '''
    
    df = data
    for lag in range(start, end + 1):
        df = lag_column(df, col, lag, None)
    return df
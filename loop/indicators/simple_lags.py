import polars as pl
from typing import Union, List

def lag_column(data: pl.DataFrame,
              col: str,
              lag: int,
              suffix: str = None) -> pl.DataFrame:
    '''
    Create a lagged version of a column.

    Args:
        data (pl.DataFrame): The input DataFrame
        col (str): The column name to lag
        lag (int): The number of periods to lag
        suffix (str, optional): The suffix for the new column name. If None, uses f"lag_{lag}"

    Returns:
        pl.DataFrame: The input data with the lagged column appended
    Raises:
        ValueError: If lag is negative or if column doesn't exist
        TypeError: If input types are incorrect
    '''
    if not isinstance(data, pl.DataFrame):
        raise TypeError("data must be a polars DataFrame")
    if not isinstance(col, str):
        raise TypeError("col must be a string")
    if not isinstance(lag, int):
        raise TypeError("lag must be an integer")
    if suffix is not None and not isinstance(suffix, str):
        raise TypeError("suffix must be a string or None")
    if lag < 0:
        raise ValueError("lag must be non-negative")
    if col not in data.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")

    new_col = f"{col}_lag_{lag}" if suffix is None else f"{col}_{suffix}"
    return data.with_columns(pl.col(col).shift(lag).alias(new_col))

def lag_columns(data: pl.DataFrame,
               cols: list[str],
               lag: int,
               suffix: str = None) -> pl.DataFrame:
    '''
    Create lagged versions of multiple columns.

    Args:
        data (pl.DataFrame): The input DataFrame
        cols (list[str]): The list of column names to lag
        lag (int): The number of periods to lag
        suffix (str, optional): The suffix for the new column names. If None, uses f"lag_{lag}"

    Returns:
        pl.DataFrame: The input data with the lagged columns appended

    Raises:
        ValueError: If cols is empty

    Note:
        Additional input validation is performed by lag_column for each column.
        This function will raise the same errors as lag_column if any column
        fails validation.
    '''
    if not cols:
        raise ValueError("cols cannot be empty")

    df = data
    for col in cols:
        df = lag_column(df, col, lag, suffix)
    return df

def lag_range(data: pl.DataFrame,
             col: str,
             start: int,
             end: int,
             suffix: str = None) -> pl.DataFrame:
    '''
    Create multiple lagged versions of a column over a range.

    Args:
        data (pl.DataFrame): The input DataFrame
        col (str): The column name to lag
        start (int): The start of lag range (inclusive)
        end (int): The end of lag range (inclusive)
        suffix (str, optional): The suffix for the new column names. If None, uses f"lag_{lag}"

    Returns:
        pl.DataFrame: The input data with the lagged columns appended

    Note:
        Input validation is performed by lag_column for each lag value.
        This function will raise the same errors as lag_column if any lag
        fails validation.
    '''
    df = data
    for lag in range(start, end + 1):
        df = lag_column(df, col, lag, suffix)
    return df
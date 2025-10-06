import polars as pl


def lag_range_cols(data: pl.DataFrame,
        cols: list[str],
        start: int,
        end: int) -> pl.DataFrame:

    '''
    Compute multiple lagged versions of multiple columns over a range.

    Args:
        data (pl.DataFrame): Klines dataset with specified columns
        cols (list[str]): The list of column names to lag
        start (int): The start of lag range (inclusive)
        end (int): The end of lag range (inclusive)

    Returns:
        pl.DataFrame: The input data with the lagged columns appended
    '''

    if not cols:
        raise ValueError('cols cannot be empty')

    if not isinstance(start, int) or not isinstance(end, int):
        raise TypeError('start and end must be integers')

    if start < 0 or end < 0:
        raise ValueError('start and end must be non-negative')

    if start > end:
        raise ValueError('start must be less than or equal to end')

    lag_expressions = [
        pl.col(col).shift(lag).alias(f'{col}_lag_{lag}')
        for col in cols
        for lag in range(start, end + 1)
    ]

    return data.with_columns(lag_expressions)


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

    '''

    return lag_range_cols(data, [col], start, end)


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
    '''

    return lag_range_cols(data, cols, lag, lag)


def lag_column(data: pl.DataFrame,
        col: str,
        lag: int,
        alias: str = None) -> pl.DataFrame:

    '''
    Compute a lagged version of a column.

    Args:
        data (pl.DataFrame): Klines dataset with specified column
        col (str): The column name to lag
        lag (int): The number of periods to lag
        alias (str, optional): New column name. If None, uses alias f"lag_{lag}"

    Returns:
        pl.DataFrame: The input data with the lagged column appended
    '''

    if alias is not None and not isinstance(alias, str):
        raise TypeError('alias must be a string or None')

    result = lag_range_cols(data, [col], lag, lag)

    return result.rename({f'{col}_lag_{lag}': alias}) if alias is not None else result
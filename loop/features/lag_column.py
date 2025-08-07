import polars as pl


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
    
    if not isinstance(data, pl.DataFrame):
        raise TypeError('data must be a polars DataFrame')
    
    if not isinstance(col, str):
        raise TypeError('col must be a string')
    
    if not isinstance(lag, int):
        raise TypeError('lag must be an integer')
    
    if alias is not None and not isinstance(alias, str):
        raise TypeError('alias must be a string or None')
    
    if lag < 0:
        raise ValueError('lag must be non-negative')
    
    if col not in data.columns:
        raise ValueError(f'Column \'{col}\' not found in DataFrame')

    new_col = f'{col}_lag_{lag}' if alias is None else alias
    
    return data.with_columns(pl.col(col).shift(lag).alias(new_col))
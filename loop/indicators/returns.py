import polars as pl


def returns(data: pl.DataFrame,
            name: str = 'returns') -> pl.DataFrame:
    
    '''
    Compute period-over-period returns of close prices.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'close' column
        name (str): Alias name for the returns output column

    Returns:
        pl.DataFrame: The input data with a new column '{name}'
    '''

    return data.with_columns(pl.col('close').pct_change().alias(name))
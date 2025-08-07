import polars as pl


def returns(data: pl.DataFrame) -> pl.DataFrame:
    
    '''
    Compute period-over-period returns of close prices.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'close' column

    Returns:
        pl.DataFrame: The input data with a new column 'returns'
    '''

    return data.with_columns(pl.col('close').pct_change().alias('returns'))
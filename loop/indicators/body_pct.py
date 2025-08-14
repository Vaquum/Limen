import polars as pl


def body_pct(data: pl.DataFrame,
             name: str = "body_pct") -> pl.DataFrame:
    
    '''
    Compute the body percentage (candle body size relative to open).
    
    Args:
        data (pl.DataFrame): Klines dataset with 'open' and 'close' columns
        name (str): Alias name for the body percentage output column

    Returns:
        pl.DataFrame: The input data with a new column '{name}'
    '''

    return data.with_columns(((pl.col('close') - pl.col('open')) / pl.col('open')).alias(name))
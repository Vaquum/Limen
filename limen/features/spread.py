import polars as pl


def spread(df: pl.DataFrame) -> pl.DataFrame:
    '''
    Compute price spread as percentage of close price.
    
    Args:
        df (pl.DataFrame): Klines dataset with 'high', 'low', 'close' columns
        
    Returns:
        pl.DataFrame: The input data with a new column 'spread'
    '''
    
    return df.with_columns([
        ((pl.col('high') - pl.col('low')) / pl.col('close')).alias('spread')
    ])
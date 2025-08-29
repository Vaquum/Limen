import polars as pl


def position_in_range(df: pl.DataFrame) -> pl.DataFrame:
    '''
    Compute position of close within candle high-low range.
    
    Uses the following calculation:
    (close - low) / (high - low + 1e-10)
    
    Args:
        df (pl.DataFrame): Klines dataset with 'high', 'low', 'close' columns
        
    Returns:
        pl.DataFrame: The input data with a new column 'position_in_range'
    '''
    
    return df.with_columns([
        ((pl.col('close') - pl.col('low')) / (pl.col('high') - pl.col('low') + 1e-10)).alias('position_in_range')
    ])
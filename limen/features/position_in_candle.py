import polars as pl

# Constants
EPSILON = 1e-10  # Small value to prevent division by zero


def position_in_candle(data: pl.DataFrame) -> pl.DataFrame:
    
    '''
    Compute position of close within candle high-low range.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'high', 'low', 'close' columns
        
    Returns:
        pl.DataFrame: The input data with a new column 'position_in_candle'
    '''
    
    return data.with_columns([
        ((pl.col('close') - pl.col('low')) / 
         (pl.col('high') - pl.col('low') + EPSILON))
        .alias('position_in_candle')
    ])
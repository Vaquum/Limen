import polars as pl

# Constants
EPSILON = 1e-10  # Small value to prevent division by zero

def price_range_position(data: pl.DataFrame, period: int = 24) -> pl.DataFrame:
    
    '''
    Compute price position within rolling high-low range.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'high', 'low', 'close' columns
        period (int): Number of periods for rolling range calculation
        
    Returns:
        pl.DataFrame: The input data with a new column 'price_range_position'
    '''
    
    return data.with_columns([
        ((pl.col('close') - pl.col('low').rolling_min(window_size=period)) / 
         (pl.col('high').rolling_max(window_size=period) - 
          pl.col('low').rolling_min(window_size=period) + EPSILON))
        .alias('price_range_position')
    ])
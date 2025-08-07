import polars as pl


def price_range_position(data: pl.DataFrame, period: int = 24) -> pl.DataFrame:
    '''
    Calculate price position within rolling high-low range.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'high', 'low', 'close' columns
        period (int): Number of klines to use as window
        
    Returns:
        pl.DataFrame: The input data with a new column 'price_range_position'
    '''
    
    return data.with_columns([
        ((pl.col('close') - pl.col('low').rolling_min(window_size=period)) / 
         (pl.col('high').rolling_max(window_size=period) - 
          pl.col('low').rolling_min(window_size=period) + 1e-10))
        .alias("price_range_position")
    ])
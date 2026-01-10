import polars as pl


def distance_from_low(data: pl.DataFrame, period: int = 24) -> pl.DataFrame:
    
    '''
    Compute distance from rolling low as percentage.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'low', 'close' columns
        period (int): Number of periods for rolling low calculation
        
    Returns:
        pl.DataFrame: The input data with a new column 'distance_from_low'
    '''
    
    return data.with_columns([
        ((pl.col('close') - pl.col('low').rolling_min(window_size=period)) / pl.col('close'))
        .alias('distance_from_low')
    ])
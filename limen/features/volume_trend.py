import polars as pl
from limen.indicators.sma import sma


def volume_trend(data: pl.DataFrame, short_period: int = 12, long_period: int = 48) -> pl.DataFrame:
    
    '''
    Compute volume trend by comparing short-term to long-term volume averages.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'volume' column
        short_period (int): Number of periods for short-term SMA calculation
        long_period (int): Number of periods for long-term SMA calculation
        
    Returns:
        pl.DataFrame: The input data with a new column 'volume_trend'
    '''
    
    # Calculate volume SMAs if not already present
    short_sma_col = f'volume_sma_{short_period}'
    long_sma_col = f'volume_sma_{long_period}'
    
    if short_sma_col not in data.columns:
        data = sma(data, 'volume', short_period)
    
    if long_sma_col not in data.columns:
        data = sma(data, 'volume', long_period)
    
    return data.with_columns([
        (pl.col(short_sma_col) / pl.col(long_sma_col))
        .alias('volume_trend')
    ])
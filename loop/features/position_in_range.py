#!/usr/bin/env python3
'''
Position in Range Feature - Position of close within candle range

Calculates where the close price falls within the high-low range of the candle,
with 0 being at the low and 1 being at the high. Uses exact same calculation
as the original inline implementation.
'''

import polars as pl


def position_in_range(df: pl.DataFrame) -> pl.DataFrame:
    '''
    Calculate position of close within candle high-low range.
    
    Uses exact same calculation as original inline implementation:
    (close - low) / (high - low + 1e-10)
    
    Args:
        df (pl.DataFrame): DataFrame with 'high', 'low', 'close' columns
        
    Returns:
        pl.DataFrame: DataFrame with added 'position_in_range' column
    '''
    
    return df.with_columns([
        ((pl.col('close') - pl.col('low')) / (pl.col('high') - pl.col('low') + 1e-10)).alias('position_in_range')
    ])
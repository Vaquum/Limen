#!/usr/bin/env python3
'''
Spread Feature - Price spread as percentage of close price

Calculates the high-low spread normalized by close price, representing
the intrabar volatility relative to the closing price.
'''

import polars as pl


def spread(df: pl.DataFrame) -> pl.DataFrame:
    '''
    Calculate price spread as percentage of close price.
    
    Args:
        df (pl.DataFrame): DataFrame with 'high', 'low', 'close' columns
        
    Returns:
        pl.DataFrame: DataFrame with added 'spread' column
    '''
    
    return df.with_columns([
        ((pl.col('high') - pl.col('low')) / pl.col('close')).alias('spread')
    ])
#!/usr/bin/env python3
'''
Time Features - Extract hour and minute from datetime

Extracts intraday time features that can capture patterns
related to market sessions and trading hours.
'''

import polars as pl


def time_features(df: pl.DataFrame) -> pl.DataFrame:
    '''
    Extract hour and minute features from datetime column.
    
    Args:
        df (pl.DataFrame): DataFrame with 'datetime' column
        
    Returns:
        pl.DataFrame: DataFrame with added 'hour' and 'minute' columns
    '''
    
    return df.with_columns([
        pl.col('datetime').dt.hour().alias('hour'),
        pl.col('datetime').dt.minute().alias('minute')
    ])
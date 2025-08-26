#!/usr/bin/env python3
'''
1-Hour Volatility Feature - Alias for 12-period rolling volatility

Provides a semantic alias for the 12-period (1-hour for 5-minute candles)
rolling volatility calculation.
'''

import polars as pl


def volatility_1h(df: pl.DataFrame, volatility_column: str = 'returns_volatility_12') -> pl.DataFrame:
    '''
    Create volatility_1h alias from existing volatility column.
    
    Args:
        df (pl.DataFrame): DataFrame with volatility column
        volatility_column (str): Name of the source volatility column
        
    Returns:
        pl.DataFrame: DataFrame with added 'volatility_1h' column
    '''
    
    return df.with_columns([
        pl.col(volatility_column).alias('volatility_1h')
    ])
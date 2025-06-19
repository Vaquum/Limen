import polars as pl
from typing import List, Union
from simple_lags import lag_range

def _breakout_lags(data: pl.DataFrame,
                 long_col: str = 'breakout_long',
                 short_col: str = 'breakout_short',
                 lookback: int = 12,
                 horizon: int = 12) -> pl.DataFrame:
    '''
    Create lag features for breakout signals.
    
    Args:
        data (pl.DataFrame): Input DataFrame with breakout columns
        long_col (str): Name of long breakout column
        short_col (str): Name of short breakout column
        lookback (int): Number of periods to look back
        horizon (int): Number of periods to shift for known data
        
    Returns:
        pl.DataFrame: Original DataFrame with new lag columns
    '''
    # Create lag features for long breakouts
    df = lag_range(data, long_col, horizon, horizon + lookback - 1)
    
    # Create lag features for short breakouts
    df = lag_range(df, short_col, horizon, horizon + lookback - 1)
    
    # Rename columns to match original naming convention
    rename_dict = {}
    for lag in range(horizon, horizon + lookback):
        rename_dict[f"{long_col}_lag_{lag}"] = f"long_t-{lag}"
        rename_dict[f"{short_col}_lag_{lag}"] = f"short_t-{lag}"
    
    return df.rename(rename_dict)

def _breakout_stats(data: pl.DataFrame,
                  long_col: str = 'breakout_long',
                  short_col: str = 'breakout_short',
                  lookback: int = 12,
                  horizon: int = 12) -> pl.DataFrame:
    '''
    Calculate rolling statistics for breakout signals.
    
    Args:
        data (pl.DataFrame): Input DataFrame with breakout columns
        long_col (str): Name of long breakout column
        short_col (str): Name of short breakout column
        lookback (int): Window size for rolling calculations
        horizon (int): Number of periods to shift for known data
        
    Returns:
        pl.DataFrame: Original DataFrame with new statistical columns
    '''
    return data.with_columns([
        # Rolling statistics for long breakouts
        pl.col(long_col)
          .shift(horizon)  # Shift to use only "known" breakout data
          .rolling_mean(window_size=lookback)
          .alias('long_roll_mean'),
        pl.col(long_col)
          .shift(horizon)
          .rolling_std(window_size=lookback)
          .alias('long_roll_std'),
        
        # Rolling statistics for short breakouts
        pl.col(short_col)
          .shift(horizon)
          .rolling_mean(window_size=lookback)
          .alias('short_roll_mean'),
        pl.col(short_col)
          .shift(horizon)
          .rolling_std(window_size=lookback)
          .alias('short_roll_std')
    ])

def _breakout_roc(data: pl.DataFrame,
                long_col: str,
                short_col: str,
                next_long_col: str,
                next_short_col: str) -> pl.DataFrame:
    '''
    Calculate Rate of Change (ROC) for breakout signals.
    
    Args:
        data (pl.DataFrame): Input DataFrame with breakout columns
        long_col (str): Name of current long breakout column
        short_col (str): Name of current short breakout column
        next_long_col (str): Name of next long breakout column
        next_short_col (str): Name of next short breakout column
        
    Returns:
        pl.DataFrame: Original DataFrame with new ROC columns
    '''
    return data.with_columns([
        # ROC for long breakouts
        pl.when(pl.col(long_col) != 0)
          .then(((pl.col(next_long_col) - pl.col(long_col)) / pl.col(long_col)) * 100)
          .otherwise(0)
          .alias('roc_long_1'),
        
        # ROC for short breakouts
        pl.when(pl.col(short_col) != 0)
          .then(((pl.col(next_short_col) - pl.col(short_col)) / pl.col(short_col)) * 100)
          .otherwise(0)
          .alias('roc_short_1')
    ])

def breakout_features(data: pl.DataFrame,
                     long_col: str = 'breakout_long',
                     short_col: str = 'breakout_short',
                     lookback: int = 12,
                     horizon: int = 12,
                     target: str = 'breakout_pct') -> pl.DataFrame:
    '''
    Calculate all breakout-related features in one operation.
    
    Args:
        data (pl.DataFrame): Input DataFrame with breakout columns
        long_col (str): Name of long breakout column
        short_col (str): Name of short breakout column
        lookback (int): Number of periods to look back
        horizon (int): Number of periods to shift for known data
        target (str): Name of target column for null dropping
        
    Returns:
        pl.DataFrame: Original DataFrame with all breakout features
    '''
    # First create lag features
    df = _breakout_lags(data, long_col, short_col, lookback, horizon)
    
    # Then add rolling statistics
    df = _breakout_stats(df, long_col, short_col, lookback, horizon)
    
    # Calculate the column names for ROC based on horizon
    current_long_col = f"long_t-{horizon + 1}"  # e.g., long_t-13
    current_short_col = f"short_t-{horizon + 1}"  # e.g., short_t-13
    next_long_col = f"long_t-{horizon}"  # e.g., long_t-12
    next_short_col = f"short_t-{horizon}"  # e.g., short_t-12

    # Finally add ROC features with proper column names
    df = _breakout_roc(df, current_long_col, current_short_col, next_long_col, next_short_col)
    
    # Drop rows with nulls in any feature or the target (matching original behavior)
    cols = [f"long_t-{i}"  for i in range(horizon, horizon + lookback)] \
         + [f"short_t-{i}" for i in range(horizon, horizon + lookback)] \
         + ['long_roll_mean','long_roll_std', 'short_roll_mean', 'short_roll_std', 'roc_long_1', 'roc_short_1', target]
    
    return df.drop_nulls(subset=cols) 
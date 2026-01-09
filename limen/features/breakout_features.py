import polars as pl
from limen.features.lagged_features import lag_range


def _breakout_lags(data: pl.DataFrame,
                 long_col: str = 'breakout_long',
                 short_col: str = 'breakout_short',
                 lookback: int = 12,
                 horizon: int = 12) -> pl.DataFrame:

    '''
    Compute lag features for breakout signals.
    
    Args:
        data (pl.DataFrame): Input DataFrame with breakout columns
        long_col (str): Name of long breakout column
        short_col (str): Name of short breakout column
        lookback (int): Number of periods to look back
        horizon (int): Number of periods to shift for known data
        
    Returns:
        pl.DataFrame: The input data with new lag columns
    '''
    
    df = lag_range(data, long_col, horizon, horizon + lookback - 1)

    df = lag_range(df, short_col, horizon, horizon + lookback - 1)

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
    Compute rolling statistics for breakout signals.
    
    Args:
        data (pl.DataFrame): Input DataFrame with breakout columns
        long_col (str): Name of long breakout column
        short_col (str): Name of short breakout column
        lookback (int): Window size for rolling calculations
        horizon (int): Number of periods to shift for known data
        
    Returns:
        pl.DataFrame: The input data with new statistical columns
    '''

    return data.with_columns([
        pl.col(long_col)
          .shift(horizon)
          .rolling_mean(window_size=lookback)
          .alias('long_roll_mean'),
        pl.col(long_col)
          .shift(horizon)
          .rolling_std(window_size=lookback)
          .alias('long_roll_std'),
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
    Compute Rate of Change (ROC) for breakout signals.
    
    Args:
        data (pl.DataFrame): Input DataFrame with breakout columns
        long_col (str): Name of current long breakout column
        short_col (str): Name of current short breakout column
        next_long_col (str): Name of next long breakout column
        next_short_col (str): Name of next short breakout column
        
    Returns:
        pl.DataFrame: The input data with new ROC columns
    '''
    
    try:
        current_lag = int(long_col.split('-')[-1])
        next_lag = int(next_long_col.split('-')[-1])
        base_lag = min(current_lag, next_lag)
        lag_diff = abs(current_lag - next_lag)
        suffix = f"_{base_lag}_{lag_diff}"
    except (ValueError, IndexError):
        suffix = ''

    return data.with_columns([
        pl.when(pl.col(long_col) != 0)
          .then(((pl.col(next_long_col) - pl.col(long_col)) / pl.col(long_col)) * 100)
          .otherwise(0)
          .alias(f"roc_long{suffix}"),
        
        pl.when(pl.col(short_col) != 0)
          .then(((pl.col(next_short_col) - pl.col(short_col)) / pl.col(short_col)) * 100)
          .otherwise(0)
          .alias(f"roc_short{suffix}")
    ])

def breakout_features(data: pl.DataFrame,
                     long_col: str = 'breakout_long',
                     short_col: str = 'breakout_short',
                     lookback: int = 12,
                     horizon: int = 12,
                     target: str = 'breakout_pct') -> pl.DataFrame:
    
    '''
    Compute comprehensive breakout-related features including lags, stats, and ROC.
    
    Args:
        data (pl.DataFrame): Klines dataset with breakout signal columns
        long_col (str): Column name for long breakout signals
        short_col (str): Column name for short breakout signals
        lookback (int): Number of periods for feature calculation
        horizon (int): Number of periods to shift for avoiding lookahead bias
        target (str): Target column name for filtering null values

    Returns:
        pl.DataFrame: The input data with multiple breakout feature columns added
    '''
    
    df = _breakout_lags(data, long_col, short_col, lookback, horizon)
    
    df = _breakout_stats(df, long_col, short_col, lookback, horizon)
    
    current_long_col = f"long_t-{horizon + 1}"
    current_short_col = f"short_t-{horizon + 1}"
    next_long_col = f"long_t-{horizon}"
    next_short_col = f"short_t-{horizon}"

    df = _breakout_roc(df, current_long_col, current_short_col, next_long_col, next_short_col)
    
    try:
        current_lag = int(current_long_col.split('-')[-1])
        next_lag = int(next_long_col.split('-')[-1])
        base_lag = min(current_lag, next_lag)
        lag_diff = abs(current_lag - next_lag)
        roc_suffix = f"_{base_lag}_{lag_diff}"
    except (ValueError, IndexError):
        roc_suffix = ""

    cols = [f"long_t-{i}"  for i in range(horizon, horizon + lookback)] \
         + [f"short_t-{i}" for i in range(horizon, horizon + lookback)] \
         + ['long_roll_mean','long_roll_std', 'short_roll_mean', 'short_roll_std', f"roc_long{roc_suffix}", f"roc_short{roc_suffix}", target]
    
    return df.drop_nulls(subset=cols)

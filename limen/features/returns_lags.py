import polars as pl
from limen.features.lagged_features import lag_range


def returns_lags(data: pl.DataFrame, max_lag: int = 24, returns_col: str = 'returns') -> pl.DataFrame:
    
    '''
    Compute multiple lagged returns features for time series analysis.
    
    Args:
        data (pl.DataFrame): Klines dataset with returns column
        max_lag (int): Maximum number of lag periods to compute
        returns_col (str): Name of the returns column
        
    Returns:
        pl.DataFrame: The input data with new columns 'returns_lag_1', 'returns_lag_2', etc.
    '''
    
    # Ensure returns column exists
    if returns_col not in data.columns:
        data = data.with_columns([
            pl.col('close').pct_change().alias(returns_col)
        ])
    
    # Use lag_range to generate all lag features at once
    return lag_range(data, returns_col, 1, max_lag)
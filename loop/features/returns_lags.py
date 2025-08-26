import polars as pl


def returns_lags(data: pl.DataFrame, max_lag: int = 24, returns_col: str = 'returns') -> pl.DataFrame:
    
    '''
    Compute multiple lagged returns features for time series analysis.
    
    Args:
        data (pl.DataFrame): Dataset with returns column
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
    
    # Generate lag features
    lag_expressions = []
    for lag in range(1, max_lag + 1):
        lag_expressions.append(
            pl.col(returns_col).shift(lag).alias(f'returns_lag_{lag}')
        )
    
    return data.with_columns(lag_expressions)
import polars as pl


def add_breakout_ema(data: pl.DataFrame, 
                     target_col: str,
                     ema_span: int = 30,
                     breakout_delta: float = 0.2,
                     breakout_horizon: int = 3) -> pl.DataFrame:
    
    '''
    Compute breakout labels based on EMA crossover patterns.
    
    Args:
        data (pl.DataFrame): Klines dataset with specified target column
        target_col (str): Column name to compute EMA breakout labels for
        ema_span (int): Number of periods for EMA calculation
        breakout_delta (float): Percentage threshold for breakout detection
        breakout_horizon (int): Number of periods to look ahead for breakout confirmation
        
    Returns:
        pl.DataFrame: The input data with a new column 'breakout_ema'
    '''
    
    s = data[target_col].to_pandas()
    
    ema = s.ewm(span=ema_span, adjust=False).mean()
    future_val = s.shift(-breakout_horizon)
    label = ((future_val > ema * (1 + breakout_delta)).astype(int)).fillna(0)
    
    return data.with_columns(pl.Series("breakout_ema", label.values.astype(int)))[:-breakout_horizon]

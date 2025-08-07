import polars as pl


def rolling_volatility(data: pl.DataFrame, column: str, window: int) -> pl.DataFrame:
    
    '''
    Compute rolling volatility (standard deviation).
    
    Args:
        data (pl.DataFrame): Klines dataset
        column (str): Column name to calculate volatility on (typically returns)
        window (int): Rolling window size
        
    Returns:
        pl.DataFrame: The input data with a new column f'{column}_volatility_{window}'
    '''
    
    return data.with_columns([
        pl.col(column).rolling_std(window_size=window).alias(f"{column}_volatility_{window}")
    ])
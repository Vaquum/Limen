import polars as pl


def sma(data: pl.DataFrame, 
        column: str, 
        period: int) -> pl.DataFrame:
    
    '''
    Compute Simple Moving Average (SMA) indicator.
    
    Args:
        data (pl.DataFrame): Klines dataset with price column
        column (str): Column name to calculate SMA on
        period (int): Number of periods for SMA calculation
        
    Returns:
        pl.DataFrame: The input data with a new column '{column}_sma_{period}'
    '''
    
    return data.with_columns([
        pl.col(column).rolling_mean(window_size=period).alias(f"{column}_sma_{period}")
    ])

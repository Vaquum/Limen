import polars as pl


def sma(data: pl.DataFrame, column: str, period: int) -> pl.DataFrame:
    '''
    Calculate Simple Moving Average.
    
    Args:
        data (pl.DataFrame): Klines dataset
        column (str): Column name to calculate SMA on
        period (int): Number of klines to use as window
        
    Returns:
        pl.DataFrame: Data with f'{column}_sma_{period}' column appended
    '''
    
    return data.with_columns([
        pl.col(column).rolling_mean(window_size=period).alias(f"{column}_sma_{period}")
    ])
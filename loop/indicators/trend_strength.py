import polars as pl


def calculate_trend_strength(data: pl.DataFrame, fast_period: int = 20, slow_period: int = 50) -> pl.DataFrame:
    '''
    Calculate trend strength based on moving average divergence.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'close' column
        fast_period (int): Fast SMA period
        slow_period (int): Slow SMA period
        
    Returns:
        pl.DataFrame: Data with 'trend_strength' column appended
    '''
    
    return (
        data
        .with_columns([
            pl.col('close').rolling_mean(window_size=fast_period).alias("sma_fast"),
            pl.col('close').rolling_mean(window_size=slow_period).alias("sma_slow"),
        ])
        .with_columns([
            ((pl.col("sma_fast") - pl.col("sma_slow")) / pl.col("sma_slow")).alias("trend_strength")
        ])
        .drop(["sma_fast", "sma_slow"])
    )
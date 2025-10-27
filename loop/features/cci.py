import polars as pl


def cci(df: pl.DataFrame, window: int = 14) -> pl.DataFrame:
    
    '''
    Compute
    
    Args:
        df (pl.DataFrame): Klines dataset with 'high', 'low', 'close' columns
        window
        
    Returns:
        pl.DataFrame: The input data with a new column ''
    '''
    
    # Typical price
    df = df.with_columns(
        ((pl.col('high') + pl.col('low') + pl.col('close')) / 3).alias('tp')
    )
    
    # Moving average of typical price
    df = df.with_columns(
        pl.col('tp').rolling_mean(window).alias('tp_sma')
    )
    
    # Mean deviation (average absolute difference from SMA)
    df = df.with_columns(
        (pl.col('tp') - pl.col('tp_sma')).abs().rolling_mean(window).alias('mean_dev')
    )
    
    # Commodity Channel Index (scaled)
    df = df.with_columns(
        ((pl.col('tp') - pl.col('tp_sma')) / (0.015 * pl.col('mean_dev'))).alias(f"cci_{window}")
    )
    
    return df

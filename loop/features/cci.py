import polars as pl


def cci(df: pl.DataFrame, window: int = 14) -> pl.DataFrame:
    
    '''
    Compute Commodity Channel Index (CCI) using rolling mean and mean deviation.
    
    Args:
        df (pl.DataFrame): Klines dataset with 'high', 'low', 'close' columns
        window (int): Number of periods for CCI calculation
        
    Returns:
        pl.DataFrame: The input data with a new column 'cci_{window}'
    '''
    
    df = df.with_columns(
        ((pl.col('high') + pl.col('low') + pl.col('close')) / 3).alias('tp')
    )
    
    df = df.with_columns(
        pl.col('tp').rolling_mean(window).alias('tp_sma')
    )
    
    df = df.with_columns(
        (pl.col('tp') - pl.col('tp_sma')).abs().rolling_mean(window).alias('mean_dev')
    )
    
    df = df.with_columns(
        ((pl.col('tp') - pl.col('tp_sma')) / (0.015 * pl.col('mean_dev'))).alias(f"cci_{window}")
    )
    
    return df

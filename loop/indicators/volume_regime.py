import polars as pl


def volume_regime(data: pl.DataFrame, lookback: int = 48) -> pl.DataFrame:
    '''
    Calculate volume regime (current vs average volume).
    
    Args:
        data (pl.DataFrame): Klines dataset with 'volume' column
        lookback (int): Lookback period for average volume
        
    Returns:
        pl.DataFrame: Data with 'volume_regime' column appended
    '''
    
    return (
        data
        .with_columns([
            pl.col('volume').rolling_mean(window_size=lookback).alias("volume_avg"),
            pl.col('volume').rolling_mean(window_size=lookback//4).alias("volume_recent"),
        ])
        .with_columns([
            (pl.col("volume_recent") / pl.col("volume_avg")).alias("volume_regime")
        ])
        .drop(["volume_avg", "volume_recent"])
    )
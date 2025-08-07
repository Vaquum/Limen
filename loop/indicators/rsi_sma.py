import polars as pl


def rsi_sma(data: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    
    '''
    Compute RSI using Simple Moving Average (not Wilder's smoothing).
    NOTE: Different from wilder_rsi which uses EWM.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'close' column
        period (int): Number of klines to use as window
        
    Returns:
        pl.DataFrame: The input data with a new column 'rsi_sma'
    '''
    
    return (
        data
        .with_columns([
            pl.col('close').diff(1).alias("delta")
        ])
        .with_columns([
            pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0).alias("gain"),
            pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0).alias("loss"),
        ])
        .with_columns([
            pl.col("gain").rolling_mean(window_size=period).alias("avg_gain"),
            pl.col("loss").rolling_mean(window_size=period).alias("avg_loss"),
        ])
        .with_columns([
            (100 - (100 / (1 + pl.col("avg_gain") / (pl.col("avg_loss") + 1e-10)))).alias("rsi_sma")
        ])
        .drop(["delta", "gain", "loss", "avg_gain", "avg_loss"])
    )
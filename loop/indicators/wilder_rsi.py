import polars as pl


def wilder_rsi(data: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    
    '''
    Compute Wilder's RSI over `period` based on column close and append as "wilder_rsi".

    Args:
        data (pl.DataFrame) : Klines dataset
        period (int) : Number of klines to use as window

    Returns:
        pl.DataFrame
        
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
            pl.col("gain").ewm_mean(alpha=1/period, adjust=False).alias("avg_gain"),
            pl.col("loss").ewm_mean(alpha=1/period, adjust=False).alias("avg_loss"),
        ])
        .with_columns([
            (100 - 100 / (1 + pl.col("avg_gain") / pl.col("avg_loss"))).alias("wilder_rsi")
        ])
        .drop(["delta", "gain", "loss", "avg_gain", "avg_loss"])
    )

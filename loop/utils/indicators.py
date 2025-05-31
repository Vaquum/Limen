import polars as pl

def perc_based_breakout(data, round_params):

    '''Adds a boolean column based on two parameters:
       
       `breakout_threshold` and `shift` and these must
       both be present in `round_params`. 
    '''
    
    data = data.with_columns([(
        ((pl.col("high") - pl.col("low")) / pl.col("open") * 100) 
            .gt(round_params['breakout_threshold'])).cast(pl.UInt8)
         .shift(round_params['shift'])
         .alias("perc_breakout")
    ]).drop_nulls("perc_breakout")

    return data



def quantile_based_breakout(data: pl.DataFrame, round_params: dict) -> pl.DataFrame:
    """
    Adds a boolean column 'perc_breakout' based on:
      – breakout_threshold: a float in [0,1] meaning “top q-quantile”
      – shift: how many rows to shift the flag forward
    
    The threshold is interpreted as the top-q fraction of the
    (high – low) / open * 100 moves.
    """
    
    move_pct = (pl.col("high") - pl.col("low")) / pl.col("open") * 100
    q = round_params["breakout_threshold"]
    cutoff = data.select(move_pct.quantile(1.0 - q)).item()

    return (
        data
        .with_columns([
            move_pct
            .gt(cutoff)
            .cast(pl.UInt8)
            .shift(round_params["shift"])
            .alias("perc_breakout")
        ])
        .drop_nulls("perc_breakout")
    )

def wilder_rsi(data: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    
    '''
    Compute Wilder's RSI over `period` based on column close and append as "rsi".

    Args:
        data (pl.DataFrame) : Klines dataset
        period (int) : Number of klines to use as window

    Returns:
        pd.DataFrame
        
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
            (100 - 100 / (1 + pl.col("avg_gain") / pl.col("avg_loss"))).alias("rsi")
        ])
        .drop(["delta", "gain", "loss", "avg_gain", "avg_loss"])
    )

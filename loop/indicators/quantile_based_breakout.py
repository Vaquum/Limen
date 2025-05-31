import polars as pl


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

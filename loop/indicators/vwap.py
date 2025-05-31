import polars as pl

def vwap(data: pl.DataFrame,
         price_col: str = "close",
         volume_col: str = "volume") -> pl.DataFrame:
    
    '''
    Compute Volume Weighted Average Price (VWAP) for each kline over its trading day.

    VWAP is defined as the cumulative sum of (price * volume) divided by cumulative sum of volume,
    reset at the start of each calendar day.

    Args:
        data (pl.DataFrame): The input kline DataFrame. Must contain:
            • "datetime"  (DateTime) – kline timestamp
            • price_col   (Float/Float64) – closing price of the kline
            • volume_col  (Float/Float64) – volume traded in the kline
        price_col (str):   Name of the price column (default: "close")
        volume_col (str):  Name of the volume column (default: "volume")

    Returns:
        pl.DataFrame: Original DataFrame sorted by "datetime", with one new column:
            • "vwap" = VWAP value for that kline’s calendar day
    '''

    return (
        data
        .sort("datetime")
        .with_columns([
            pl.col("datetime").dt.date().alias("__date")
        ])
        .with_columns([
            (pl.col(price_col) * pl.col(volume_col))
                .cum_sum()
                .over("__date")
                .alias("__cum_pv"),
            pl.col(volume_col)
                .cum_sum()
                .over("__date")
                .alias("__cum_vol")
        ])
        .with_columns([
            (pl.col("__cum_pv") / pl.col("__cum_vol"))
                .alias("vwap")
        ])
        .drop(["__date", "__cum_pv", "__cum_vol"])
    )

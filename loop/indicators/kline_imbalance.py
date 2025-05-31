import polars as pl

def kline_imbalance(data: pl.DataFrame, window: int = 14) -> pl.DataFrame:
    """
    Compute a rolling buyer/seller imbalance over K‐lines instead of raw trades.

    For each kline (row), we define:
        #aggressive_buys  = (1 – maker_ratio) * no_of_trades
        #aggressive_sells = maker_ratio * no_of_trades
        imbalance_raw     = #aggressive_buys – #aggressive_sells
                         = (1 – 2 * maker_ratio) * no_of_trades

    Then we take a rolling sum of “imbalance_raw” over the last `window` klines.

    Args:
        data (pl.DataFrame): K‐line DataFrame. Must contain:
            • "datetime"     (DateTime or comparable) – kline timestamp
            • "maker_ratio"  (Float) – fraction of trades where buyer was maker
            • "no_of_trades" (UInt32 or Int) – total trade count in that kline
        window (int): Number of consecutive klines to include in the rolling sum.

    Returns:
        pl.DataFrame: Original DataFrame sorted by "datetime", with one new column:
            • "imbalance" = rolling sum of imbalance_raw over the last `window` rows
    """
    return (
        data
        .sort("datetime")
        .with_columns([
            # Compute per‐kline raw imbalance
            ((1 - 2 * pl.col("maker_ratio")) * pl.col("no_of_trades"))
                .alias("imbalance_raw")
        ])
        .with_columns([
            # Rolling sum of the last `window` values of "imbalance_raw"
            pl.col("imbalance_raw")
              .rolling_sum(window_size=window, min_samples=1)
              .alias("imbalance")
        ])
        .drop("imbalance_raw")
    )

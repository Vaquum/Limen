import polars as pl


def macd(data: pl.DataFrame,
         close_col: str = "close",
         fast_period: int = 12,
         slow_period: int = 26,
         signal_period: int = 9) -> pl.DataFrame:
    '''
    Compute MACD (Moving Average Convergence Divergence) from kline close prices.

    MACD line = EMA(fast_period) – EMA(slow_period)
    Signal line = EMA(signal_period) of MACD line
    Histogram = MACD line – Signal line

    Uses exponential moving averages (Wilder’s style).

    Args:
        data (pl.DataFrame): The input kline DataFrame. Must contain:
            • close_col (Float/Float64) – closing price of the kline
        close_col (str):       Name of the close price column (default: "close")
        fast_period (int):     Lookback for the fast EMA (default: 12)
        slow_period (int):     Lookback for the slow EMA (default: 26)
        signal_period (int):   Lookback for the signal‐line EMA (default: 9)

    Returns:
        pl.DataFrame: Original DataFrame, with three new columns:
            • "macd"        = EMA(fast_period) – EMA(slow_period)
            • "macd_signal" = EMA(signal_period) of the "macd" column
            • "macd_hist"   = "macd" – "macd_signal"
    '''

    alpha_fast = 2.0 / (fast_period + 1)
    alpha_slow = 2.0 / (slow_period + 1)
    alpha_signal = 2.0 / (signal_period + 1)

    return (
        data
        .with_columns([
            pl.col(close_col)
              .ewm_mean(alpha=alpha_fast, adjust=False)
              .alias("__ema_fast"),
            pl.col(close_col)
              .ewm_mean(alpha=alpha_slow, adjust=False)
              .alias("__ema_slow")
        ])
        .with_columns([
            (pl.col("__ema_fast") - pl.col("__ema_slow"))
              .alias("macd")
        ])
        .with_columns([
            pl.col("macd")
              .ewm_mean(alpha=alpha_signal, adjust=False)
              .alias("macd_signal")
        ])
        .with_columns([
            (pl.col("macd") - pl.col("macd_signal"))
              .alias("macd_hist")
        ])
        .drop(["__ema_fast", "__ema_slow"])
    )

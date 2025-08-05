import polars as pl


def macd(data: pl.DataFrame,
         close_col: str = "close",
         fast_period: int = 12,
         slow_period: int = 26,
         signal_period: int = 9) -> pl.DataFrame:
    '''
    Compute MACD (Moving Average Convergence Divergence) from kline close prices.

    Args:
        data (pl.DataFrame): The input kline DataFrame.
        close_col (str): Name of the close price column.
        fast_period (int): Lookback for the fast EMA.
        slow_period (int): Lookback for the slow EMA.
        signal_period (int): Lookback for the signal‐line EMA.

    Returns:
        pl.DataFrame: The input data with three macd columns added.
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

import polars as pl

def atr(data: pl.DataFrame,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        period: int = 14) -> pl.DataFrame:
    
    '''
    Compute Average True Range (ATR) over `period` using Wilder's smoothing (EMA).

    Args:
        data (pl.DataFrame): The input data
        high_col (str): The column name for the high prices.
        low_col (str): The column name for the low prices.
        close_col (str): The column name for the closing prices.
        period (int): The period for the ATR calculation.

    Returns:
        pl.DataFrame: The input data with the ATR column appended.

    '''
    
    prev_close = pl.col(close_col).shift(1)
    true_range = pl.max_horizontal([
        pl.col(high_col) - pl.col(low_col),
        (pl.col(high_col) - prev_close).abs(),
        (pl.col(low_col) - prev_close).abs(),
    ])

    return data.with_columns([
        true_range
            .ewm_mean(alpha=1.0 / period, adjust=False)
            .alias("atr")
    ])

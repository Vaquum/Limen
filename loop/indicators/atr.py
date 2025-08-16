import polars as pl

def atr(data: pl.DataFrame,
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        period: int = 14) -> pl.DataFrame:
    
    '''
    Compute Average True Range (ATR) using Wilder's smoothing method.

    Args:
        data (pl.DataFrame): Klines dataset with 'high', 'low', 'close' columns
        high_col (str): Column name for high prices
        low_col (str): Column name for low prices
        close_col (str): Column name for close prices
        period (int): Number of periods for ATR calculation

    Returns:
        pl.DataFrame: The input data with a new column 'atr_{period}'
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
            .alias(f"atr_{period}")
    ])

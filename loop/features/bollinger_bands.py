import polars as pl


def bollinger_bands(
    df: pl.DataFrame,
    price_col: str = "close",
    window: int = 20,
    num_std: float = 2.0
) -> pl.DataFrame:
    
    '''
    Compute Bollinger Bands using SMA.
    
    Args:
        df (pl.DataFrame): Klines dataset with 'high', 'low', 'close' columns
        price_col (str): Column to use for price.
        window (int): Rolling window size for SMA and Std Dev.
        num_std (float): Number of standard deviations for upper/lower bands.
        
    Returns:
        pl.DataFrame: The input data with a new column ''
    '''

    sma_col = f"bb_middle_{window}"
    upper_col = f"bb_upper_{window}"
    lower_col = f"bb_lower_{window}"

    return df.with_columns([
        pl.col(price_col).rolling_mean(window).alias(sma_col),
        (pl.col(price_col).rolling_mean(window)
         + num_std * pl.col(price_col).rolling_std(window)
        ).alias(upper_col),
        (pl.col(price_col).rolling_mean(window)
         - num_std * pl.col(price_col).rolling_std(window)
        ).alias(lower_col)
    ])

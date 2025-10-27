import polars as pl


def bollinger_bands(
    df: pl.DataFrame,
    price_col: str = 'close',
    window: int = 20,
    num_std: float = 2.0
) -> pl.DataFrame:
    
    '''
    Compute Bollinger Bands using Simple Moving Average (SMA).
    
    Args:
        df (pl.DataFrame): Klines dataset with 'close' column
        price_col (str): Column name used for Bollinger Band calculation
        window (int): Number of periods for SMA and standard deviation calculation
        num_std (float): Number of standard deviations for upper and lower bands
        
    Returns:
        pl.DataFrame: The input data with three new columns: 
                      'bb_middle_{window}', 'bb_upper_{window}', and 'bb_lower_{window}'
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

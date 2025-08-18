import polars as pl


def ichimoku_cloud(data: pl.DataFrame) -> pl.DataFrame:

    '''
    Compute Ichimoku Cloud components for trend and momentum analysis.
    
    Args:
        df (pl.DataFrame): Klines dataset with 'high', 'low', 'close' columns

    Returns:
        pl.DataFrame: The input data with new columns:
            'tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou'
    '''
    
    return (
        data
        .with_columns(
            ((pl.col("high").rolling_max(9) + pl.col("low").rolling_min(9)) / 2).alias("tenkan")
        )
        .with_columns(
            ((pl.col("high").rolling_max(26) + pl.col("low").rolling_min(26)) / 2).alias("kijun")
        )
        .with_columns(
            ((pl.col("tenkan") + pl.col("kijun")) / 2).shift(-26).alias("senkou_a")
        )
        .with_columns(
            ((pl.col("high").rolling_max(52) + pl.col("low").rolling_min(52)) / 2).shift(-26).alias("senkou_b")
        )
        .with_columns(
            pl.col("close").shift(26).alias("chikou")
        )
    )

import polars as pl


def wilder_rsi(data: pl.DataFrame, 
               period: int = 14,
               name: str = 'wilder_rsi') -> pl.DataFrame:
    
    '''
    Compute Wilder's RSI using exponential smoothing method.

    Args:
        data (pl.DataFrame): Klines dataset with 'close' column
        period (int): Number of periods for RSI calculation
        name (str): Alias name for the Wilder's RSI output column

    Returns:
        pl.DataFrame: The input data with a new column '{name}'
    '''
    
    return (
        data
        .with_columns([
            pl.col('close').diff(1).alias('delta')
        ])
        .with_columns([
            pl.when(pl.col('delta') > 0).then(pl.col('delta')).otherwise(0).alias('gain'),
            pl.when(pl.col('delta') < 0).then(-pl.col('delta')).otherwise(0).alias('loss'),
        ])
        .with_columns([
            pl.col('gain').ewm_mean(alpha=1/period, adjust=False).alias('avg_gain'),
            pl.col('loss').ewm_mean(alpha=1/period, adjust=False).alias('avg_loss'),
        ])
        .with_columns([
            (100 - 100 / (1 + pl.col('avg_gain') / pl.col('avg_loss'))).alias(name)
        ])
        .drop(['delta', 'gain', 'loss', 'avg_gain', 'avg_loss'])
    )

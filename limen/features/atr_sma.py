import polars as pl


def atr_sma(data: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    
    '''
    Compute Average True Range using Simple Moving Average.
    
    NOTE: Different from standard ATR which uses Wilder's smoothing.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'high', 'low', 'close' columns
        period (int): Number of periods for ATR calculation
        
    Returns:
        pl.DataFrame: The input data with a new column 'atr_sma'
    '''
    
    return (
        data
        .with_columns([
            pl.col('close').shift(1).alias('prev_close')
        ])
        .with_columns([
            (pl.col('high') - pl.col('low')).alias('high_low'),
            (pl.col('high') - pl.col('prev_close')).abs().alias('high_close'),
            (pl.col('low') - pl.col('prev_close')).abs().alias('low_close'),
        ])
        .with_columns([
            pl.max_horizontal(['high_low', 'high_close', 'low_close']).alias('true_range')
        ])
        .with_columns([
            pl.col('true_range').rolling_mean(window_size=period).alias('atr_sma')
        ])
        .drop(['prev_close', 'high_low', 'high_close', 'low_close', 'true_range'])
    )
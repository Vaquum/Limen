import polars as pl
from loop.indicators.sma import sma


def volume_weight(data: pl.DataFrame, period: int = 20) -> pl.DataFrame:
    
    '''
    Compute volume-based weighting factor with clipping.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'volume' column
        period (int): Period for volume moving average
        
    Returns:
        pl.DataFrame: The input data with new columns 'volume_ma', 'volume_weight'
    '''
    
    df = sma(data, 'volume', period)
    df = df.rename({f'volume_sma_{period}': 'volume_ma'})
    
    df = df.with_columns([
        (pl.col('volume') / pl.col('volume_ma')).clip(0.5, 2.0).alias('volume_weight')
    ])
    
    return df
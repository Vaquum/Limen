import polars as pl
from limen.indicators.sma import sma



def volume_weight(data: pl.DataFrame, 
                 period: int = 20,
                 volume_weight_min: float = 0.5,
                 volume_weight_max: float = 2.0) -> pl.DataFrame:
    
    '''
    Compute volume-based weighting factor with clipping.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'volume' column
        period (int): Period for volume moving average
        volume_weight_min (float): Minimum volume weight value
        volume_weight_max (float): Maximum volume weight value
        
    Returns:
        pl.DataFrame: The input data with new columns 'volume_ma', 'volume_weight'
    '''
    
    df = sma(data, 'volume', period)
    df = df.rename({f'volume_sma_{period}': 'volume_ma'})
    
    df = df.with_columns([
        (pl.col('volume') / pl.col('volume_ma')).clip(volume_weight_min, volume_weight_max).alias('volume_weight')
    ])
    
    return df
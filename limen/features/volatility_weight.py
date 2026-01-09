import polars as pl
from limen.indicators.rolling_volatility import rolling_volatility



def volatility_weight(data: pl.DataFrame, 
                     period: int = 20,
                     volatility_scaling_factor: float = 100,
                     volatility_weight_min: float = 0.3,
                     volatility_weight_max: float = 1.0) -> pl.DataFrame:
    
    '''
    Compute volatility-based weighting factor with inverse scaling.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'close' column
        period (int): Period for volatility calculation
        volatility_scaling_factor (float): Scaling factor for volatility normalization
        volatility_weight_min (float): Minimum volatility weight value
        volatility_weight_max (float): Maximum volatility weight value
        
    Returns:
        pl.DataFrame: The input data with new columns 'volatility', 'volatility_weight'
    '''
    
    df = data.with_columns([
        pl.col('close').pct_change().alias('returns_temp')
    ])
    
    df = rolling_volatility(df, 'returns_temp', period)
    df = df.rename({f'returns_temp_volatility_{period}': 'volatility'})
    
    df = df.with_columns([
        (2 / (1 + pl.col('volatility') * volatility_scaling_factor)).clip(volatility_weight_min, volatility_weight_max).alias('volatility_weight')
    ])
    
    return df.drop('returns_temp')
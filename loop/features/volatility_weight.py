import polars as pl
from loop.indicators.rolling_volatility import rolling_volatility

VOLATILITY_SCALING_FACTOR = 100
VOLATILITY_WEIGHT_MIN = 0.3
VOLATILITY_WEIGHT_MAX = 1.0


def volatility_weight(data: pl.DataFrame, period: int = 20) -> pl.DataFrame:
    
    '''
    Compute volatility-based weighting factor with inverse scaling.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'close' column
        period (int): Period for volatility calculation
        
    Returns:
        pl.DataFrame: The input data with new columns 'volatility', 'volatility_weight'
    '''
    
    df = data.with_columns([
        pl.col('close').pct_change().alias('returns_temp')
    ])
    
    df = rolling_volatility(df, 'returns_temp', period)
    df = df.rename({f'returns_temp_volatility_{period}': 'volatility'})
    
    df = df.with_columns([
        (2 / (1 + pl.col('volatility') * VOLATILITY_SCALING_FACTOR)).clip(VOLATILITY_WEIGHT_MIN, VOLATILITY_WEIGHT_MAX).alias('volatility_weight')
    ])
    
    return df.drop('returns_temp')
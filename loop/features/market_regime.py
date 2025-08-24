import polars as pl
from loop.indicators.sma import sma
from loop.features.trend_strength import trend_strength
from loop.features.volume_regime import volume_regime


def market_regime(df: pl.DataFrame, lookback: int = 48) -> pl.DataFrame:
    
    '''
    Compute market regime indicators including trend strength and volume regime.
    
    Args:
        df (pl.DataFrame): Klines dataset with 'close', 'volume' columns
        lookback (int): Lookback period for calculations
        
    Returns:
        pl.DataFrame: The input data with new columns 'sma_20', 'sma_50', 'trend_strength', 'volatility_ratio', 'volume_sma', 'volume_regime', 'market_favorable'
    '''
    
    df = sma(df, 'close', 20)
    df = sma(df, 'close', 50)
    df = trend_strength(df, 20, 50)
    
    df = df.rename({
        'close_sma_20': 'sma_20',
        'close_sma_50': 'sma_50'
    })
    
    df = df.with_columns([
        pl.col('close').pct_change().alias('returns_temp')
    ])
    
    df = df.with_columns([
        (pl.col('returns_temp').rolling_std(window_size=12) / 
         pl.col('returns_temp').rolling_std(window_size=48)).alias('volatility_ratio')
    ])
    
    df = sma(df, 'volume', 48)
    df = df.rename({'volume_sma_48': 'volume_sma'})
    df = volume_regime(df, 48)
    
    df = df.with_columns([
        (((pl.col('trend_strength') > -0.001).cast(pl.Int32) +
          (pl.col('volatility_ratio') < 2.0).cast(pl.Int32) +
          (pl.col('volume_regime') > 0.7).cast(pl.Int32)) / 3.0)
        .alias('market_favorable')
    ])
    
    return df.drop('returns_temp')
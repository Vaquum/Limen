import polars as pl
from limen.indicators.sma import sma
from limen.features.trend_strength import trend_strength
from limen.features.volume_regime import volume_regime


def market_regime(df: pl.DataFrame, lookback: int = 48, short_sma: int = 20, long_sma: int = 50) -> pl.DataFrame:
    
    '''
    Compute market regime indicators including trend strength and volume regime.
    
    Args:
        df (pl.DataFrame): Klines dataset with 'close', 'volume' columns
        lookback (int): Lookback period for calculations used for volatility ratio and volume regime
        short_sma (int): Period for short SMA calculation
        long_sma (int): Period for long SMA calculation
        
    Returns:
        pl.DataFrame: The input data with new columns 'sma_20', 'sma_50', 'trend_strength', 'volatility_ratio', 'volume_sma', 'volume_regime', 'market_favorable'
    '''
    
    df = sma(df, 'close', short_sma)
    df = sma(df, 'close', long_sma)
    df = trend_strength(df, short_sma, long_sma)
    
    df = df.rename({
        f'close_sma_{short_sma}': 'sma_20',
        f'close_sma_{long_sma}': 'sma_50'
    })
    
    df = df.with_columns([
        pl.col('close').pct_change().alias('returns_temp')
    ])
    
    df = df.with_columns([
        (pl.col('returns_temp').rolling_std(window_size=12) / 
         pl.col('returns_temp').rolling_std(window_size=lookback)).alias('volatility_ratio')
    ])
    
    df = sma(df, 'volume', lookback)
    df = df.rename({f'volume_sma_{lookback}': 'volume_sma'})
    df = volume_regime(df, lookback)
    
    df = df.with_columns([
        (((pl.col('trend_strength') > -0.001).cast(pl.Int32) +
          (pl.col('volatility_ratio') < 2.0).cast(pl.Int32) +
          (pl.col('volume_regime') > 0.7).cast(pl.Int32)) / 3.0)
        .alias('market_favorable')
    ])
    
    return df.drop('returns_temp')
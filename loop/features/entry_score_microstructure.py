import polars as pl
from loop.features.position_in_candle import position_in_candle
from loop.features.micro_momentum import micro_momentum
from loop.features.volume_spike import volume_spike
from loop.features.spread_percent import spread_percent


def entry_score_microstructure(data: pl.DataFrame, 
                              micro_momentum_period: int = 3,
                              volume_spike_period: int = 20,
                              spread_mean_period: int = 48) -> pl.DataFrame:
    
    '''
    Compute sophisticated entry score based on microstructure timing signals.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'high', 'low', 'close', 'volume', 'volatility_regime' columns
        micro_momentum_period (int): Period for micro momentum calculation
        volume_spike_period (int): Period for volume spike calculation
        spread_mean_period (int): Period for spread normalization
        
    Returns:
        pl.DataFrame: The input data with a new column 'entry_score'
    '''
    
    # Calculate component microstructure features
    df = position_in_candle(data)
    df = micro_momentum(df, micro_momentum_period)
    df = volume_spike(df, volume_spike_period)
    df = spread_percent(df)
    
    # Calculate base entry score with equal weights
    df = df.with_columns([
        ((1 - pl.col('position_in_candle')) * 0.25 +
         (pl.col('micro_momentum') > 0).cast(pl.Float32) * 0.25 +
         pl.col('volume_spike').clip(0.5, 1.5) / 1.5 * 0.25 +
         (1 - (pl.col('spread_percent') / pl.col('spread_percent').rolling_mean(window_size=spread_mean_period)).clip(0, 2)) * 0.25)
        .alias('entry_score_base')
    ])
    
    # Calculate regime-aware entry score
    df = df.with_columns([
        pl.when(pl.col('volatility_regime') == 'low')
            .then(
                (1 - pl.col('position_in_candle')) * 0.35 +
                (pl.col('micro_momentum') > 0).cast(pl.Float32) * 0.15 +
                pl.col('volume_spike').clip(0.5, 1.5) / 1.5 * 0.15 +
                (1 - (pl.col('spread_percent') / pl.col('spread_percent').rolling_mean(window_size=spread_mean_period)).clip(0, 2)) * 0.35
            )
            .when(pl.col('volatility_regime') == 'high')
            .then(
                (1 - pl.col('position_in_candle')) * 0.15 +
                (pl.col('micro_momentum') > 0).cast(pl.Float32) * 0.35 +
                pl.col('volume_spike').clip(0.5, 1.5) / 1.5 * 0.35 +
                (1 - (pl.col('spread_percent') / pl.col('spread_percent').rolling_mean(window_size=spread_mean_period)).clip(0, 2)) * 0.15
            )
            .otherwise(pl.col('entry_score_base'))
            .alias('entry_score')
    ])
    
    # Keep intermediate microstructure features for potential use
    # Drop only the intermediate entry_score_base
    return df.drop(['entry_score_base'])
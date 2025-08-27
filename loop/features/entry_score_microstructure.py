import polars as pl
from loop.features.position_in_candle import position_in_candle
from loop.features.micro_momentum import micro_momentum
from loop.features.volume_spike import volume_spike
from loop.features.spread_percent import spread_percent

POSITION_WEIGHT_BASE = 0.25
MOMENTUM_WEIGHT_BASE = 0.25
VOLUME_WEIGHT_BASE = 0.25
SPREAD_WEIGHT_BASE = 0.25

POSITION_WEIGHT_LOW_VOL = 0.35
MOMENTUM_WEIGHT_LOW_VOL = 0.15
VOLUME_WEIGHT_LOW_VOL = 0.15
SPREAD_WEIGHT_LOW_VOL = 0.35

POSITION_WEIGHT_HIGH_VOL = 0.15
MOMENTUM_WEIGHT_HIGH_VOL = 0.35
VOLUME_WEIGHT_HIGH_VOL = 0.35
SPREAD_WEIGHT_HIGH_VOL = 0.15

VOLUME_SPIKE_MIN = 0.5
VOLUME_SPIKE_MAX = 1.5
VOLUME_SPIKE_NORMALIZER = 1.5

SPREAD_RATIO_MIN = 0
SPREAD_RATIO_MAX = 2


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
    
    df = position_in_candle(data)
    df = micro_momentum(df, micro_momentum_period)
    df = volume_spike(df, volume_spike_period)
    df = spread_percent(df)
    
    
    df = df.with_columns([
        ((1 - pl.col('position_in_candle')) * POSITION_WEIGHT_BASE +
         (pl.col('micro_momentum') > 0).cast(pl.Float32) * MOMENTUM_WEIGHT_BASE +
         pl.col('volume_spike').clip(VOLUME_SPIKE_MIN, VOLUME_SPIKE_MAX) / VOLUME_SPIKE_NORMALIZER * VOLUME_WEIGHT_BASE +
         (1 - (pl.col('spread_percent') / pl.col('spread_percent').rolling_mean(window_size=spread_mean_period)).clip(SPREAD_RATIO_MIN, SPREAD_RATIO_MAX)) * SPREAD_WEIGHT_BASE)
        .alias('entry_score_base')
    ])
    
    
    df = df.with_columns([
        pl.when(pl.col('volatility_regime') == 'low')
            .then(
                (1 - pl.col('position_in_candle')) * POSITION_WEIGHT_LOW_VOL +
                (pl.col('micro_momentum') > 0).cast(pl.Float32) * MOMENTUM_WEIGHT_LOW_VOL +
                pl.col('volume_spike').clip(VOLUME_SPIKE_MIN, VOLUME_SPIKE_MAX) / VOLUME_SPIKE_NORMALIZER * VOLUME_WEIGHT_LOW_VOL +
                (1 - (pl.col('spread_percent') / pl.col('spread_percent').rolling_mean(window_size=spread_mean_period)).clip(SPREAD_RATIO_MIN, SPREAD_RATIO_MAX)) * SPREAD_WEIGHT_LOW_VOL
            )
            .when(pl.col('volatility_regime') == 'high')
            .then(
                (1 - pl.col('position_in_candle')) * POSITION_WEIGHT_HIGH_VOL +
                (pl.col('micro_momentum') > 0).cast(pl.Float32) * MOMENTUM_WEIGHT_HIGH_VOL +
                pl.col('volume_spike').clip(VOLUME_SPIKE_MIN, VOLUME_SPIKE_MAX) / VOLUME_SPIKE_NORMALIZER * VOLUME_WEIGHT_HIGH_VOL +
                (1 - (pl.col('spread_percent') / pl.col('spread_percent').rolling_mean(window_size=spread_mean_period)).clip(SPREAD_RATIO_MIN, SPREAD_RATIO_MAX)) * SPREAD_WEIGHT_HIGH_VOL
            )
            .otherwise(pl.col('entry_score_base'))
            .alias('entry_score')
    ])
    
    return df.drop(['entry_score_base'])
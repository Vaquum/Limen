import polars as pl
from limen.features.position_in_candle import position_in_candle
from limen.features.micro_momentum import micro_momentum
from limen.features.volume_spike import volume_spike
from limen.features.spread_percent import spread_percent


def entry_score_microstructure(data: pl.DataFrame, 
                              micro_momentum_period: int = 3,
                              volume_spike_period: int = 20,
                              spread_mean_period: int = 48,
                              entry_position_weight_base: float = 0.25,
                              entry_momentum_weight_base: float = 0.25,
                              entry_volume_weight_base: float = 0.25,
                              entry_spread_weight_base: float = 0.25,
                              entry_position_weight_low_vol: float = 0.35,
                              entry_momentum_weight_low_vol: float = 0.15,
                              entry_volume_weight_low_vol: float = 0.15,
                              entry_spread_weight_low_vol: float = 0.35,
                              entry_position_weight_high_vol: float = 0.15,
                              entry_momentum_weight_high_vol: float = 0.35,
                              entry_volume_weight_high_vol: float = 0.35,
                              entry_spread_weight_high_vol: float = 0.15,
                              entry_volume_spike_min: float = 0.5,
                              entry_volume_spike_max: float = 1.5,
                              entry_volume_spike_normalizer: float = 1.5,
                              entry_spread_ratio_min: float = 0,
                              entry_spread_ratio_max: float = 2) -> pl.DataFrame:
    
    '''
    Compute sophisticated entry score based on microstructure timing signals.
    
    Args:
        data (pl.DataFrame): Klines dataset with 'high', 'low', 'close', 'volume', 'volatility_regime' columns
        micro_momentum_period (int): Period for micro momentum calculation
        volume_spike_period (int): Period for volume spike calculation
        spread_mean_period (int): Period for spread normalization
        entry_position_weight_base (float): Base weight for position-in-candle factor
        entry_momentum_weight_base (float): Base weight for momentum factor
        entry_volume_weight_base (float): Base weight for volume factor
        entry_spread_weight_base (float): Base weight for spread factor
        entry_position_weight_low_vol (float): Position weight in low volatility regime
        entry_momentum_weight_low_vol (float): Momentum weight in low volatility regime
        entry_volume_weight_low_vol (float): Volume weight in low volatility regime
        entry_spread_weight_low_vol (float): Spread weight in low volatility regime
        entry_position_weight_high_vol (float): Position weight in high volatility regime
        entry_momentum_weight_high_vol (float): Momentum weight in high volatility regime
        entry_volume_weight_high_vol (float): Volume weight in high volatility regime
        entry_spread_weight_high_vol (float): Spread weight in high volatility regime
        entry_volume_spike_min (float): Minimum volume spike value for clipping
        entry_volume_spike_max (float): Maximum volume spike value for clipping
        entry_volume_spike_normalizer (float): Normalizer for volume spike scaling
        entry_spread_ratio_min (float): Minimum spread ratio for clipping
        entry_spread_ratio_max (float): Maximum spread ratio for clipping
        
    Returns:
        pl.DataFrame: The input data with a new column 'entry_score'
    '''
    
    df = position_in_candle(data)
    df = micro_momentum(df, micro_momentum_period)
    df = volume_spike(df, volume_spike_period)
    df = spread_percent(df)
    
    
    df = df.with_columns([
        ((1 - pl.col('position_in_candle')) * entry_position_weight_base +
         (pl.col('micro_momentum') > 0).cast(pl.Float32) * entry_momentum_weight_base +
         pl.col('volume_spike').clip(entry_volume_spike_min, entry_volume_spike_max) / entry_volume_spike_normalizer * entry_volume_weight_base +
         (1 - (pl.col('spread_percent') / pl.col('spread_percent').rolling_mean(window_size=spread_mean_period)).clip(entry_spread_ratio_min, entry_spread_ratio_max)) * entry_spread_weight_base)
        .alias('entry_score_base')
    ])
    
    
    df = df.with_columns([
        pl.when(pl.col('volatility_regime') == 'low')
            .then(
                (1 - pl.col('position_in_candle')) * entry_position_weight_low_vol +
                (pl.col('micro_momentum') > 0).cast(pl.Float32) * entry_momentum_weight_low_vol +
                pl.col('volume_spike').clip(entry_volume_spike_min, entry_volume_spike_max) / entry_volume_spike_normalizer * entry_volume_weight_low_vol +
                (1 - (pl.col('spread_percent') / pl.col('spread_percent').rolling_mean(window_size=spread_mean_period)).clip(entry_spread_ratio_min, entry_spread_ratio_max)) * entry_spread_weight_low_vol
            )
            .when(pl.col('volatility_regime') == 'high')
            .then(
                (1 - pl.col('position_in_candle')) * entry_position_weight_high_vol +
                (pl.col('micro_momentum') > 0).cast(pl.Float32) * entry_momentum_weight_high_vol +
                pl.col('volume_spike').clip(entry_volume_spike_min, entry_volume_spike_max) / entry_volume_spike_normalizer * entry_volume_weight_high_vol +
                (1 - (pl.col('spread_percent') / pl.col('spread_percent').rolling_mean(window_size=spread_mean_period)).clip(entry_spread_ratio_min, entry_spread_ratio_max)) * entry_spread_weight_high_vol
            )
            .otherwise(pl.col('entry_score_base'))
            .alias('entry_score')
    ])
    
    return df.drop(['entry_score_base'])
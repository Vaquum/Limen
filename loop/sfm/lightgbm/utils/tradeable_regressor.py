#!/usr/bin/env python3
'''
Feature Engineering Functions for LightGBM Tradeable Regressor
'''

import polars as pl
from loop.indicators.rsi_sma import rsi_sma
from loop.indicators.sma import sma
from loop.indicators.rolling_volatility import rolling_volatility
from loop.features.atr_sma import atr_sma
from loop.features.atr_percent_sma import atr_percent_sma
from loop.features.market_regime import market_regime
from loop.features.momentum_confirmation import momentum_confirmation
from loop.utils.time_decay import time_decay

def calculate_market_regime(df: pl.DataFrame, lookback: int = 48) -> pl.DataFrame:
    return market_regime(df, lookback)

def calculate_dynamic_parameters(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    '''
    Compute dynamic targets and stop losses based on market volatility conditions.
    
    Args:
        df (pl.DataFrame): Klines dataset with 'high', 'low', 'close' columns
        config (dict): Configuration dictionary with dynamic parameter settings
    
    Returns:
        pl.DataFrame: The input data with new columns 'rolling_volatility', 'atr', 'atr_pct', 'dynamic_target', 'dynamic_stop_loss'
    '''
    
    df = df.with_columns([
        pl.col('close').pct_change().alias('returns')
    ])
    
    df = rolling_volatility(df, 'returns', config['volatility_lookback'])
    df = df.rename({f"returns_volatility_{config['volatility_lookback']}": 'rolling_volatility'})
    
    df = atr_sma(df, config['volatility_lookback'])
    df = df.rename({'atr_sma': 'atr'})
    
    df = atr_percent_sma(df, config['volatility_lookback'])
    df = df.rename({'atr_percent_sma': 'atr_pct'})
    
    df = df.with_columns([
        pl.col('close').shift(1).alias('prev_close')
    ])
    
    if config['dynamic_targets']:
        df = df.with_columns([
            ((pl.col('rolling_volatility') + pl.col('atr_pct')) / 2).alias('volatility_measure')
        ])
        
        df = df.with_columns([
            pl.when(pl.col('volatility_regime') == 'low')
                .then(pl.lit(0.8))
                .when(pl.col('volatility_regime') == 'high')
                .then(pl.lit(1.2))
                .otherwise(pl.lit(1.0))
                .alias('regime_multiplier')
        ])
        
        df = df.with_columns([
            (pl.col('volatility_measure') * config['target_volatility_multiplier'] * pl.col('regime_multiplier'))
                .clip(config['base_min_breakout'] * 0.6, config['base_min_breakout'] * 1.4)
                .alias('dynamic_target')
        ])
    else:
        df = df.with_columns([
            pl.lit(config['base_min_breakout']).alias('dynamic_target')
        ])
    
    if config['volatility_adjusted_stops']:
        df = df.with_columns([
            (pl.col('volatility_measure') * config['stop_volatility_multiplier'] * pl.col('regime_multiplier'))
                .clip(config['base_stop_loss'] * 0.7, config['base_stop_loss'] * 1.4)
                .alias('dynamic_stop_loss')
        ])
    else:
        df = df.with_columns([
            pl.lit(config['base_stop_loss']).alias('dynamic_stop_loss')
        ])
    
    return df.drop('prev_close')

def calculate_microstructure_features(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    '''
    Compute microstructure features for better entry timing including position in candle and volume spikes.
    
    Args:
        df (pl.DataFrame): Klines dataset with 'high', 'low', 'close', 'volume' columns
        config (dict): Configuration dictionary with microstructure settings
    
    Returns:
        pl.DataFrame: The input data with new columns 'position_in_candle', 'micro_momentum', 'volume_spike', 'spread_pct', 'entry_score'
    '''
    
    if config['microstructure_timing']:
        df = df.with_columns([
            ((pl.col('close') - pl.col('low')) / (pl.col('high') - pl.col('low') + 1e-10))
                .alias('position_in_candle')
        ])
        
        df = df.with_columns([
            pl.col('close').pct_change(3).alias('micro_momentum')
        ])
        
        df = df.with_columns([
            (pl.col('volume') / pl.col('volume').rolling_mean(window_size=20))
                .alias('volume_spike')
        ])
        
        df = df.with_columns([
            ((pl.col('high') - pl.col('low')) / pl.col('close')).alias('spread_pct')
        ])
        
        df = df.with_columns([
            ((1 - pl.col('position_in_candle')) * 0.25 +
             (pl.col('micro_momentum') > 0).cast(pl.Float32) * 0.25 +
             pl.col('volume_spike').clip(0.5, 1.5) / 1.5 * 0.25 +
             (1 - (pl.col('spread_pct') / pl.col('spread_pct').rolling_mean(window_size=48)).clip(0, 2)) * 0.25)
            .alias('entry_score_base')
        ])
        
        df = df.with_columns([
            pl.when(pl.col('volatility_regime') == 'low')
                .then(
                    (1 - pl.col('position_in_candle')) * 0.35 +
                    (pl.col('micro_momentum') > 0).cast(pl.Float32) * 0.15 +
                    pl.col('volume_spike').clip(0.5, 1.5) / 1.5 * 0.15 +
                    (1 - (pl.col('spread_pct') / pl.col('spread_pct').rolling_mean(window_size=48)).clip(0, 2)) * 0.35
                )
                .when(pl.col('volatility_regime') == 'high')
                .then(
                    (1 - pl.col('position_in_candle')) * 0.15 +
                    (pl.col('micro_momentum') > 0).cast(pl.Float32) * 0.35 +
                    pl.col('volume_spike').clip(0.5, 1.5) / 1.5 * 0.35 +
                    (1 - (pl.col('spread_pct') / pl.col('spread_pct').rolling_mean(window_size=48)).clip(0, 2)) * 0.15
                )
                .otherwise(pl.col('entry_score_base'))
                .alias('entry_score')
        ])
        
        df = df.drop('entry_score_base')
    else:
        df = df.with_columns([
            pl.lit(1.0).alias('entry_score')
        ])
    
    return df

def calculate_simple_momentum_confirmation(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    if config['simple_momentum_confirmation']:
        return momentum_confirmation(df, short_period=1, long_period=3, short_weight=0.5)
    else:
        return df.with_columns([pl.lit(1.0).alias('momentum_score')])

def simulate_exit_reality(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    '''
    Simulate actual trading scenarios to get realistic P&L outcomes with stops and targets.
    
    Args:
        df (pl.DataFrame): Input DataFrame with OHLCV data and dynamic parameters
        config (dict): Configuration dictionary with trading simulation settings
    
    Returns:
        pl.DataFrame: DataFrame with exit reality simulation results added
    '''
    
    lookahead_candles = config['lookahead_minutes'] // 5
    
    df = df.with_columns([
        pl.lit(None).alias('exit_gross_return'),
        pl.lit(None).alias('exit_net_return'),
        pl.lit('').alias('exit_reason'),
        pl.lit(None).alias('exit_bars'),
        pl.lit(None).alias('exit_max_return'),
        pl.lit(None).alias('exit_min_return')
    ])
    
    sample_indices = list(range(0, len(df) - lookahead_candles, 5))
    
    df_pd = df.to_pandas()
    
    for i in sample_indices:
        entry_price = df_pd.iloc[i]['close']
        dynamic_target = df_pd.iloc[i]['dynamic_target']
        dynamic_stop = df_pd.iloc[i]['dynamic_stop_loss']
        
        max_return = 0
        min_return = 0
        exit_return = 0
        exit_reason = 'timeout'
        bars_to_exit = lookahead_candles
        
        for j in range(1, lookahead_candles + 1):
            current_idx = i + j
            if current_idx >= len(df_pd):
                break
                
            high_price = df_pd.iloc[current_idx]['high']
            low_price = df_pd.iloc[current_idx]['low']
            
            high_return = (high_price - entry_price) / entry_price
            low_return = (low_price - entry_price) / entry_price
            max_return = max(max_return, high_return)
            min_return = min(min_return, low_return)
            
            if low_return <= -dynamic_stop:
                exit_return = -dynamic_stop
                exit_reason = 'stop_loss'
                bars_to_exit = j
                break
            
            if high_return >= dynamic_target:
                exit_return = dynamic_target
                exit_reason = 'target_hit'
                bars_to_exit = j
                break
            
            if config['trailing_stop'] and max_return > 0:
                trailing_level = max_return - config['trailing_stop_distance']
                if low_return <= trailing_level:
                    exit_return = trailing_level
                    exit_reason = 'trailing_stop'
                    bars_to_exit = j
                    break
        
        if exit_reason == 'timeout':
            exit_return = (df_pd.iloc[min(i + lookahead_candles, len(df_pd) - 1)]['close'] - entry_price) / entry_price
        
        gross_return = exit_return
        net_return = gross_return - config['commission_rate']
        
        df_pd.at[i, 'exit_gross_return'] = gross_return
        df_pd.at[i, 'exit_net_return'] = net_return
        df_pd.at[i, 'exit_reason'] = exit_reason
        df_pd.at[i, 'exit_bars'] = bars_to_exit
        df_pd.at[i, 'exit_max_return'] = max_return
        df_pd.at[i, 'exit_min_return'] = min_return
    
    numeric_exit_cols = ['exit_gross_return', 'exit_net_return', 
                        'exit_bars', 'exit_max_return', 'exit_min_return']
    for col in numeric_exit_cols:
        df_pd[col] = df_pd[col].astype('float64')
    df_pd[numeric_exit_cols] = df_pd[numeric_exit_cols].ffill()
    
    df_pd['exit_reason'] = df_pd['exit_reason'].fillna('').ffill()
    
    return pl.from_pandas(df_pd)

def calculate_time_decay_factor(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    return time_decay(df, 'exit_bars', config['time_decay_halflife'], time_units=5, output_column='time_decay_factor')

def create_tradeable_labels(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    '''
    Compute comprehensive tradeable labels combining exit reality, time decay, and market conditions.
    
    Args:
        df (pl.DataFrame): Klines dataset with required feature columns
        config (dict): Configuration dictionary with labeling settings
    
    Returns:
        pl.DataFrame: DataFrame with final tradeable labels and scores
    '''
    
    df = df.with_columns([
        pl.col('close').ewm_mean(span=21, adjust=False).alias('ema')
    ])
    
    lookahead_candles = config['lookahead_minutes'] // 5
    
    df = df.with_columns([
        pl.col('high').rolling_max(window_size=lookahead_candles).shift(-lookahead_candles).alias('future_high'),
        pl.col('low').rolling_min(window_size=lookahead_candles).shift(-lookahead_candles).alias('future_low')
    ])
    
    df = df.with_columns([
        ((pl.col('future_high') - pl.col('close')) / pl.col('close')).alias('capturable_breakout'),
        ((pl.col('future_low') - pl.col('close')) / pl.col('close')).alias('max_drawdown')
    ])
    
    df = df.with_columns([
        (1 - (pl.col('close') - pl.col('ema')).abs() / pl.col('ema'))
            .clip(0, 1)
            .pow(config['ema_weight_power'])
            .alias('ema_alignment')
    ])
    
    if config['volume_weight_enabled']:
        df = sma(df, 'volume', 20)
        df = df.rename({'volume_sma_20': 'volume_ma'})
        
        df = df.with_columns([
            (pl.col('volume') / pl.col('volume_ma')).clip(0.5, 2.0).alias('volume_weight')
        ])
    else:
        df = df.with_columns([
            pl.lit(1.0).alias('volume_weight')
        ])
    
    df = df.with_columns([
        pl.col('close').pct_change().alias('returns_temp')
    ])
    df = rolling_volatility(df, 'returns_temp', 20)
    df = df.rename({f'returns_temp_volatility_20': 'volatility'})
    
    df = df.with_columns([
        (2 / (1 + pl.col('volatility') * 100)).clip(0.3, 1.0).alias('volatility_weight')
    ])
    
    df = df.with_columns([
        ((pl.col('close').pct_change(12) > 0).cast(pl.Float32) * 0.5 + 0.5).alias('momentum_weight')
    ])
    
    if config['market_regime_filter']:
        regime_weight_col = 'market_favorable'
    else:
        df = df.with_columns([
            pl.lit(1.0).alias('market_favorable')
        ])
        regime_weight_col = 'market_favorable'
    
    df = df.with_columns([
        (pl.col('capturable_breakout') * 
         pl.col('ema_alignment') * 
         pl.col('volume_weight') * 
         pl.col('volatility_weight') *
         pl.col('momentum_weight') *
         pl.col(regime_weight_col) *
         pl.col('entry_score') *
         pl.col('momentum_score'))
        .alias('tradeable_breakout')
    ])
    
    df = df.with_columns([
        (pl.col('capturable_breakout') / (pl.col('max_drawdown').abs() + 0.001)).alias('risk_reward_ratio')
    ])
    
    df = df.with_columns([
        (pl.col('tradeable_breakout') * pl.col('risk_reward_ratio').clip(0, 3)).alias('tradeable_score_base')
    ])
    
    df = df.with_columns([
        pl.when(pl.col('exit_net_return').is_not_null())
            .then(pl.col('exit_net_return').clip(-0.01, 0.02))
            .otherwise(pl.lit(0))
            .alias('exit_reality_score')
    ])
    
    df = df.with_columns([
        pl.when((pl.col('exit_reason').is_in(['target_hit', 'trailing_stop'])) & (pl.col('exit_net_return') > 0))
            .then(pl.lit(1.0))
            .when((pl.col('exit_reason') == 'stop_loss') | ((pl.col('exit_reason') == 'timeout') & (pl.col('exit_net_return') < 0)))
            .then(pl.lit(0.2))
            .otherwise(pl.lit(0.5))
            .alias('exit_quality')
    ])
    
    df = df.with_columns([
        (pl.col('exit_reality_score') * pl.col('time_decay_factor')).alias('exit_reality_time_decayed')
    ])
    
    df = df.with_columns([
        ((1 - config['exit_reality_blend'] - config['time_decay_blend']) * pl.col('tradeable_score_base') +
         config['exit_reality_blend'] * pl.col('exit_reality_score') * pl.col('exit_quality') +
         config['time_decay_blend'] * pl.col('exit_reality_time_decayed') * pl.col('exit_quality'))
        .alias('tradeable_score')
    ])
    
    df = df.with_columns([
        ((pl.col('exit_reason') == 'target_hit') | 
         (pl.col('capturable_breakout') >= pl.col('dynamic_target')))
        .alias('achieves_dynamic_target')
    ])
    
    return df.drop('returns_temp')

def prepare_features_5m(df: pl.DataFrame, lookback: int = 48, config: dict = None) -> pl.DataFrame:
    '''
    Compute comprehensive feature set for 5-minute trading including momentum, volatility, and volume features.
    
    Args:
        df (pl.DataFrame): Klines dataset with 'open', 'high', 'low', 'close', 'volume' columns
        lookback (int): Lookback period for feature calculations
        config (dict): Optional configuration dictionary
    
    Returns:
        pl.DataFrame: The input data with complete feature set for trading models
    '''
    
    df = df.with_columns([
        pl.col('close').pct_change().alias('returns'),
        (pl.col('close') / pl.col('close').shift(1)).log().alias('log_returns')
    ])
    
    for period in [12, 24, 48]:
        df = df.with_columns([
            pl.col('close').pct_change(period).alias(f'momentum_{period}')
        ])
        df = rsi_sma(df, period)
        df = df.rename({f"rsi_sma_{period}": f"rsi_{period}"})
    
    df = rolling_volatility(df, 'returns', 12)
    df = df.rename({'returns_volatility_12': 'volatility_5m'})
    
    df = df.with_columns([
        pl.col('volatility_5m').alias('volatility_1h')
    ])
    
    df = sma(df, 'volume', 20)
    df = df.with_columns([
        (pl.col('volume') / pl.col('volume_sma_20')).alias('volume_ratio')
    ])
    
    df = sma(df, 'volume', 12)
    df = sma(df, 'volume', 48)
    df = df.with_columns([
        (pl.col('volume_sma_12') / pl.col('volume_sma_48')).alias('volume_trend')
    ])
    
    df = df.with_columns([
        ((pl.col('high') - pl.col('low')) / pl.col('close')).alias('spread'),
        ((pl.col('close') - pl.col('low')) / (pl.col('high') - pl.col('low') + 1e-10)).alias('position_in_range')
    ])
    
    df = df.with_columns([
        pl.col('datetime').dt.hour().alias('hour'),
        pl.col('datetime').dt.minute().alias('minute')
    ])
    
    for lag in range(1, min(lookback + 1, 25)):
        df = df.with_columns([
            pl.col('returns').shift(lag).alias(f'returns_lag_{lag}')
        ])
    
    base_min_breakout = config.get('base_min_breakout', 0.005) if config else 0.005
    volatility_regime_enabled = config.get('volatility_regime_enabled', True) if config else True
    
    df = df.with_columns([
        pl.col('dynamic_target').fill_null(base_min_breakout).alias('dynamic_target_feature'),
        pl.col('entry_score').fill_null(1.0).alias('entry_score_feature'),
        pl.col('momentum_score').fill_null(1.0).alias('momentum_score_feature')
    ])
    
    if volatility_regime_enabled:
        df = df.with_columns([
            pl.col('vol_60h').fill_null(0).alias('vol_60h_feature'),
            pl.col('vol_percentile').fill_null(50).alias('vol_percentile_feature'),
            pl.col('regime_low').fill_null(0).alias('regime_low_feature'),
            pl.col('regime_normal').fill_null(1).alias('regime_normal_feature'),
            pl.col('regime_high').fill_null(0).alias('regime_high_feature')
        ])
    
    df = df.with_columns([
        ((pl.col('close') - pl.col('high')) / pl.col('high')).alias('close_to_high'),
        ((pl.col('close') - pl.col('low')) / pl.col('low')).alias('close_to_low')
    ])
    
    for period in [5, 10, 20, 50]:
        df = sma(df, 'close', period)
        df = df.with_columns([
            (pl.col('close') / pl.col(f'close_sma_{period}')).alias(f'sma_{period}_ratio')
        ])
    
    return df
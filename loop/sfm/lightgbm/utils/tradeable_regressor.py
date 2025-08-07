#!/usr/bin/env python3
"""
Feature Engineering Functions for LightGBM Tradeable Regressor
"""

import polars as pl
from loop.indicators.rsi_sma import rsi_sma
from loop.indicators.sma import sma
from loop.features.trend_strength import trend_strength
from loop.features.volume_regime import volume_regime
from loop.indicators.rolling_volatility import rolling_volatility
from loop.features.atr_sma import atr_sma
from loop.features.atr_percent_sma import atr_percent_sma

# Constants for volatility regime values
VOL_REGIME_LOW_VALUE = 10
VOL_REGIME_HIGH_VALUE = 90
VOL_REGIME_MID_VALUE = 50

def calculate_volatility_regime(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Calculate volatility regime for each row"""
    
    # Calculate rolling volatility (60-hour window)
    lookback = config['vol_regime_lookback']
    
    # First calculate returns
    df = df.with_columns([
        pl.col('close').pct_change().alias('returns_temp')
    ])
    
    # Then calculate rolling volatility
    df = rolling_volatility(df, 'returns_temp', lookback)
    df = df.with_columns([
        pl.col(f'returns_temp_volatility_{lookback}').alias('vol_60h')
    ]).drop(f'returns_temp_volatility_{lookback}')
    
    # Calculate rolling percentiles of volatility
    window_size = lookback * 2
    
    df = df.with_columns([
        (pl.col('vol_60h')
         .rolling_quantile(window_size=window_size, quantile=0.2)
         .alias('vol_p20')),
        (pl.col('vol_60h')
         .rolling_quantile(window_size=window_size, quantile=0.8)
         .alias('vol_p80'))
    ])
    
    df = df.with_columns([
        pl.when(pl.col('vol_60h') <= pl.col('vol_p20'))
        .then(VOL_REGIME_LOW_VALUE)
        .when(pl.col('vol_60h') >= pl.col('vol_p80'))
        .then(VOL_REGIME_HIGH_VALUE)
        .otherwise(VOL_REGIME_MID_VALUE)
        .alias('vol_percentile')
    ])
    
    df = df.drop(['vol_p20', 'vol_p80'])
    
    # Classify into regimes
    df = df.with_columns([
        pl.when(pl.col('vol_percentile') <= config['vol_low_percentile'])
            .then(pl.lit('low'))
            .when(pl.col('vol_percentile') >= config['vol_high_percentile'])
            .then(pl.lit('high'))
            .otherwise(pl.lit('normal'))
            .alias('volatility_regime')
    ])
    
    # Add regime-specific features
    df = df.with_columns([
        (pl.col('volatility_regime') == 'low').cast(pl.Int32).alias('regime_low'),
        (pl.col('volatility_regime') == 'normal').cast(pl.Int32).alias('regime_normal'),
        (pl.col('volatility_regime') == 'high').cast(pl.Int32).alias('regime_high')
    ])
    
    return df.drop('returns_temp')

def calculate_market_regime(df: pl.DataFrame, lookback: int = 48) -> pl.DataFrame:
    """Calculate market regime indicators"""
    
    # Trend strength using existing indicators
    df = sma(df, 'close', 20)
    df = sma(df, 'close', 50)
    df = trend_strength(df, 20, 50)
    
    # Rename the SMA columns to expected names
    df = df.rename({
        'close_sma_20': 'sma_20',
        'close_sma_50': 'sma_50'
    })
    
    # Volatility regime (short-term)
    df = df.with_columns([
        pl.col('close').pct_change().alias('returns_temp')
    ])
    
    df = df.with_columns([
        (pl.col('returns_temp').rolling_std(window_size=12) / 
         pl.col('returns_temp').rolling_std(window_size=48)).alias('volatility_ratio')
    ])
    
    # Volume regime
    df = sma(df, 'volume', 48)
    df = df.rename({'volume_sma_48': 'volume_sma'})
    df = volume_regime(df, 48)
    
    # Market favorability score
    df = df.with_columns([
        (((pl.col('trend_strength') > -0.001).cast(pl.Int32) +
          (pl.col('volatility_ratio') < 2.0).cast(pl.Int32) +
          (pl.col('volume_regime') > 0.7).cast(pl.Int32)) / 3.0)
        .alias('market_favorable')
    ])
    
    return df.drop('returns_temp')

def calculate_dynamic_parameters(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Calculate dynamic targets and stops based on market conditions"""
    
    # Calculate rolling volatility
    df = df.with_columns([
        pl.col('close').pct_change().alias('returns')
    ])
    
    df = rolling_volatility(df, 'returns', config['volatility_lookback'])
    df = df.rename({f"returns_volatility_{config['volatility_lookback']}": 'rolling_volatility'})
    
    # Calculate ATR
    df = atr_sma(df, config['volatility_lookback'])
    df = df.rename({'atr_sma': 'atr'})
    
    df = atr_percent_sma(df, config['volatility_lookback'])
    df = df.rename({'atr_percent_sma': 'atr_pct'})
    
    # Calculate true range components for other uses
    df = df.with_columns([
        pl.col('close').shift(1).alias('prev_close')
    ])
    
    df = df.with_columns([
        (pl.col('high') - pl.col('low')).alias('high_low'),
        (pl.col('high') - pl.col('prev_close')).abs().alias('high_close'),
        (pl.col('low') - pl.col('prev_close')).abs().alias('low_close')
    ])
    
    df = df.with_columns([
        pl.max_horizontal(['high_low', 'high_close', 'low_close']).alias('true_range')
    ])
    
    # Dynamic target - adjust based on volatility regime
    if config['dynamic_targets']:
        df = df.with_columns([
            ((pl.col('rolling_volatility') + pl.col('atr_pct')) / 2).alias('volatility_measure')
        ])
        
        # Regime-specific adjustments
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
    
    # Dynamic stop loss - also regime-adjusted
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
    """Calculate microstructure features for better entry timing"""
    
    if config['microstructure_timing']:
        # Position within candle
        df = df.with_columns([
            ((pl.col('close') - pl.col('low')) / (pl.col('high') - pl.col('low') + 1e-10))
                .alias('position_in_candle')
        ])
        
        # Recent price momentum
        df = df.with_columns([
            pl.col('close').pct_change(3).alias('micro_momentum')
        ])
        
        # Volume spike indicator
        df = df.with_columns([
            (pl.col('volume') / pl.col('volume').rolling_mean(window_size=20))
                .alias('volume_spike')
        ])
        
        # Spread tightness
        df = df.with_columns([
            ((pl.col('high') - pl.col('low')) / pl.col('close')).alias('spread_pct')
        ])
        
        # Base entry score
        df = df.with_columns([
            ((1 - pl.col('position_in_candle')) * 0.25 +
             (pl.col('micro_momentum') > 0).cast(pl.Float32) * 0.25 +
             pl.col('volume_spike').clip(0.5, 1.5) / 1.5 * 0.25 +
             (1 - (pl.col('spread_pct') / pl.col('spread_pct').rolling_mean(window_size=48)).clip(0, 2)) * 0.25)
            .alias('entry_score_base')
        ])
        
        # Low vol: weight spread and position more
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
    """Simple momentum confirmation"""
    
    if config['simple_momentum_confirmation']:
        df = df.with_columns([
            pl.col('close').pct_change(1).alias('momentum_1'),
            pl.col('close').pct_change(3).alias('momentum_3')
        ])
        
        df = df.with_columns([
            ((pl.col('momentum_1') > 0).cast(pl.Float32) * 0.5 +
             (pl.col('momentum_3') > 0).cast(pl.Float32) * 0.5)
            .alias('momentum_score')
        ])
    else:
        df = df.with_columns([
            pl.lit(1.0).alias('momentum_score')
        ])
    
    return df

def simulate_exit_reality(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Simulate actual trading to get real P&L outcomes"""
    
    lookahead_candles = config['lookahead_minutes'] // 5
    
    # For Polars, we need to work differently than pandas
    # We'll create the exit reality columns first
    df = df.with_columns([
        pl.lit(None).alias('exit_gross_return'),
        pl.lit(None).alias('exit_net_return'),
        pl.lit('').alias('exit_reason'),
        pl.lit(None).alias('exit_bars'),
        pl.lit(None).alias('exit_max_return'),
        pl.lit(None).alias('exit_min_return')
    ])
    
    # Sample every 5th row indices
    sample_indices = list(range(0, len(df) - lookahead_candles, 5))
    
    # Convert to pandas for the simulation (temporary)
    # This is necessary because the forward-looking logic is very difficult in pure Polars
    df_pd = df.to_pandas()
    
    # For each sampled potential entry point
    for i in sample_indices:
        entry_price = df_pd.iloc[i]['close']
        dynamic_target = df_pd.iloc[i]['dynamic_target']
        dynamic_stop = df_pd.iloc[i]['dynamic_stop_loss']
        
        # Simulate the trade
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
            
            # Track extremes
            high_return = (high_price - entry_price) / entry_price
            low_return = (low_price - entry_price) / entry_price
            max_return = max(max_return, high_return)
            min_return = min(min_return, low_return)
            
            # Check stop loss
            if low_return <= -dynamic_stop:
                exit_return = -dynamic_stop
                exit_reason = 'stop_loss'
                bars_to_exit = j
                break
            
            # Check target
            if high_return >= dynamic_target:
                exit_return = dynamic_target
                exit_reason = 'target_hit'
                bars_to_exit = j
                break
            
            # Trailing stop logic
            if config['trailing_stop'] and max_return > 0:
                trailing_level = max_return - config['trailing_stop_distance']
                if low_return <= trailing_level:
                    exit_return = trailing_level
                    exit_reason = 'trailing_stop'
                    bars_to_exit = j
                    break
        
        # If no exit triggered, use close at timeout
        if exit_reason == 'timeout':
            exit_return = (df_pd.iloc[min(i + lookahead_candles, len(df_pd) - 1)]['close'] - entry_price) / entry_price
        
        # Calculate P&L after fees
        gross_return = exit_return
        net_return = gross_return - config['commission_rate']
        
        # Update the pandas dataframe
        df_pd.at[i, 'exit_gross_return'] = gross_return
        df_pd.at[i, 'exit_net_return'] = net_return
        df_pd.at[i, 'exit_reason'] = exit_reason
        df_pd.at[i, 'exit_bars'] = bars_to_exit
        df_pd.at[i, 'exit_max_return'] = max_return
        df_pd.at[i, 'exit_min_return'] = min_return
    
    # Forward fill the exit reality data
    # Convert numeric columns to float to avoid downcasting warning
    numeric_exit_cols = ['exit_gross_return', 'exit_net_return', 
                        'exit_bars', 'exit_max_return', 'exit_min_return']
    for col in numeric_exit_cols:
        df_pd[col] = df_pd[col].astype('float64')
    df_pd[numeric_exit_cols] = df_pd[numeric_exit_cols].ffill()
    
    # String column
    df_pd['exit_reason'] = df_pd['exit_reason'].fillna('').ffill()
    
    # Convert back to Polars
    return pl.from_pandas(df_pd)

def calculate_time_decay_factor(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Calculate time decay factor for exit reality scores"""
    
    # Calculate decay factor based on bars to exit
    halflife_bars = config['time_decay_halflife'] / 5  # Convert minutes to 5-min bars
    
    df = df.with_columns([
        pl.when(pl.col('exit_bars').is_not_null())
            .then(pl.lit(-0.693).mul(pl.col('exit_bars')).truediv(halflife_bars).exp())
            .otherwise(pl.lit(0.5))
            .alias('time_decay_factor')
    ])
    
    return df

def create_tradeable_labels(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Create tradeable labels with exit reality and time decay"""
    
    # Calculate EMA
    df = df.with_columns([
        pl.col('close').ewm_mean(span=21, adjust=False).alias('ema')
    ])
    
    # Look ahead for actual high/low
    lookahead_candles = config['lookahead_minutes'] // 5
    
    df = df.with_columns([
        pl.col('high').rolling_max(window_size=lookahead_candles).shift(-lookahead_candles).alias('future_high'),
        pl.col('low').rolling_min(window_size=lookahead_candles).shift(-lookahead_candles).alias('future_low')
    ])
    
    # Calculate what's capturable from opening price
    df = df.with_columns([
        ((pl.col('future_high') - pl.col('close')) / pl.col('close')).alias('capturable_breakout'),
        ((pl.col('future_low') - pl.col('close')) / pl.col('close')).alias('max_drawdown')
    ])
    
    # EMA alignment weight
    df = df.with_columns([
        (1 - (pl.col('close') - pl.col('ema')).abs() / pl.col('ema'))
            .clip(0, 1)
            .pow(config['ema_weight_power'])
            .alias('ema_alignment')
    ])
    
    # Volume confirmation weight
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
    
    # Volatility weight
    df = df.with_columns([
        pl.col('close').pct_change().alias('returns_temp')
    ])
    df = rolling_volatility(df, 'returns_temp', 20)
    df = df.rename({f'returns_temp_volatility_20': 'volatility'})
    
    df = df.with_columns([
        (2 / (1 + pl.col('volatility') * 100)).clip(0.3, 1.0).alias('volatility_weight')
    ])
    
    # Momentum weight
    df = df.with_columns([
        ((pl.col('close').pct_change(12) > 0).cast(pl.Float32) * 0.5 + 0.5).alias('momentum_weight')
    ])
    
    # Market regime weight
    if config['market_regime_filter']:
        regime_weight_col = 'market_favorable'
    else:
        df = df.with_columns([
            pl.lit(1.0).alias('market_favorable')
        ])
        regime_weight_col = 'market_favorable'
    
    # Combined tradeable breakout
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
    
    # Risk-adjusted score
    df = df.with_columns([
        (pl.col('capturable_breakout') / (pl.col('max_drawdown').abs() + 0.001)).alias('risk_reward_ratio')
    ])
    
    df = df.with_columns([
        (pl.col('tradeable_breakout') * pl.col('risk_reward_ratio').clip(0, 3)).alias('tradeable_score_base')
    ])
    
    # EXIT REALITY ENHANCEMENT
    df = df.with_columns([
        pl.when(pl.col('exit_net_return').is_not_null())
            .then(pl.col('exit_net_return').clip(-0.01, 0.02))
            .otherwise(pl.lit(0))
            .alias('exit_reality_score')
    ])
    
    # Quality factor
    df = df.with_columns([
        pl.when((pl.col('exit_reason').is_in(['target_hit', 'trailing_stop'])) & (pl.col('exit_net_return') > 0))
            .then(pl.lit(1.0))
            .when((pl.col('exit_reason') == 'stop_loss') | ((pl.col('exit_reason') == 'timeout') & (pl.col('exit_net_return') < 0)))
            .then(pl.lit(0.2))
            .otherwise(pl.lit(0.5))
            .alias('exit_quality')
    ])
    
    # TIME DECAY ENHANCEMENT
    df = df.with_columns([
        (pl.col('exit_reality_score') * pl.col('time_decay_factor')).alias('exit_reality_time_decayed')
    ])
    
    # BLEND ALL THREE COMPONENTS
    df = df.with_columns([
        ((1 - config['exit_reality_blend'] - config['time_decay_blend']) * pl.col('tradeable_score_base') +
         config['exit_reality_blend'] * pl.col('exit_reality_score') * pl.col('exit_quality') +
         config['time_decay_blend'] * pl.col('exit_reality_time_decayed') * pl.col('exit_quality'))
        .alias('tradeable_score')
    ])
    
    # Add flag for trades that achieve dynamic target
    df = df.with_columns([
        ((pl.col('exit_reason') == 'target_hit') | 
         (pl.col('capturable_breakout') >= pl.col('dynamic_target')))
        .alias('achieves_dynamic_target')
    ])
    
    return df.drop('returns_temp')

def prepare_features_5m(df: pl.DataFrame, lookback: int = 48, config: dict = None) -> pl.DataFrame:
    """Prepare features for 5-minute trading"""
    
    # Price-based features
    df = df.with_columns([
        pl.col('close').pct_change().alias('returns'),
        (pl.col('close') / pl.col('close').shift(1)).log().alias('log_returns')
    ])
    
    # Momentum indicators
    for period in [12, 24, 48]:
        df = df.with_columns([
            pl.col('close').pct_change(period).alias(f'momentum_{period}')
        ])
        df = rsi_sma(df, period)
        df = df.rename({f'rsi_sma': f'rsi_{period}'})
    
    # Volatility
    df = rolling_volatility(df, 'returns', 12)
    df = df.rename({'returns_volatility_12': 'volatility_5m'})
    
    # Duplicate for 1h (same calculation for now)
    df = df.with_columns([
        pl.col('volatility_5m').alias('volatility_1h')
    ])
    
    # Volume features
    df = sma(df, 'volume', 20)
    df = df.with_columns([
        (pl.col('volume') / pl.col('volume_sma_20')).alias('volume_ratio')
    ])
    
    df = sma(df, 'volume', 12)
    df = sma(df, 'volume', 48)
    df = df.with_columns([
        (pl.col('volume_sma_12') / pl.col('volume_sma_48')).alias('volume_trend')
    ])
    
    # Microstructure
    df = df.with_columns([
        ((pl.col('high') - pl.col('low')) / pl.col('close')).alias('spread'),
        ((pl.col('close') - pl.col('low')) / (pl.col('high') - pl.col('low') + 1e-10)).alias('position_in_range')
    ])
    
    # Time features
    df = df.with_columns([
        pl.col('datetime').dt.hour().alias('hour'),
        pl.col('datetime').dt.minute().alias('minute')
    ])
    
    # Lagged features
    for lag in range(1, min(lookback + 1, 25)):
        df = df.with_columns([
            pl.col('returns').shift(lag).alias(f'returns_lag_{lag}')
        ])
    
    # Use default values if config not provided
    base_min_breakout = config.get('base_min_breakout', 0.005) if config else 0.005
    volatility_regime_enabled = config.get('volatility_regime_enabled', True) if config else True
    
    # Add dynamic features
    df = df.with_columns([
        pl.col('dynamic_target').fill_null(base_min_breakout).alias('dynamic_target_feature'),
        pl.col('entry_score').fill_null(1.0).alias('entry_score_feature'),
        pl.col('momentum_score').fill_null(1.0).alias('momentum_score_feature')
    ])
    
    # Add volatility regime features
    if volatility_regime_enabled:
        df = df.with_columns([
            pl.col('vol_60h').fill_null(0).alias('vol_60h_feature'),
            pl.col('vol_percentile').fill_null(50).alias('vol_percentile_feature'),
            pl.col('regime_low').fill_null(0).alias('regime_low_feature'),
            pl.col('regime_normal').fill_null(1).alias('regime_normal_feature'),
            pl.col('regime_high').fill_null(0).alias('regime_high_feature')
        ])
    
    # Additional calculated features
    df = df.with_columns([
        ((pl.col('close') - pl.col('high')) / pl.col('high')).alias('close_to_high'),
        ((pl.col('close') - pl.col('low')) / pl.col('low')).alias('close_to_low')
    ])
    
    # SMAs for additional features
    for period in [5, 10, 20, 50]:
        df = sma(df, 'close', period)
        df = df.with_columns([
            (pl.col('close') / pl.col(f'close_sma_{period}')).alias(f'sma_{period}_ratio')
        ])
    
    return df
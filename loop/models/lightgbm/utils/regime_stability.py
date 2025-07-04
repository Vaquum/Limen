'''
Utility functions for regime stability feature engineering
Location: loop/models/lightgbm/utils/regime_stability.py
'''

import polars as pl

# Feature configuration constants
VOLATILITY_WINDOWS = [6, 12, 24]
ACTIVITY_WINDOWS = [6, 12]
MOMENTUM_WINDOWS = [6, 12]
STABILITY_ROLLING_MEAN_WINDOW = 24
EMA_WINDOW = 5


def add_stability_features(df, leakage_shift=12):
    '''
    Add features for regime stability prediction.
    
    These features capture market volatility, regime change patterns,
    momentum consistency, and alignment indicators that help predict
    whether a regime will persist.
    
    Args:
        df (pl.DataFrame): DataFrame with base features including long_base, short_base
        leakage_shift (int): Number of periods to shift for future regime (default: 12)
        
    Returns:
        pl.DataFrame: DataFrame with additional stability features and regime_persists label
    '''
    
    # Ensure we have price columns
    if 'long_close' not in df.columns:
        if 'close' in df.columns:
            df = df.with_columns([
                pl.col('close').alias('long_close'),
                pl.col('close').alias('short_close'),
            ])
        else:
            raise ValueError("Need 'close' or 'long_close' column for stability features")
    
    # Add EMA if not present
    if 'long_ema_5' not in df.columns:
        df = df.with_columns([
            pl.col('long_close').ewm_mean(span=EMA_WINDOW).alias('long_ema_5'),
        ])
    
    # 1. Volatility measures
    for window in VOLATILITY_WINDOWS:
        df = df.with_columns([
            pl.col('long_close').rolling_std(window).alias(f'price_vol_{window}h'),
            (pl.col('long_close').rolling_max(window) - pl.col('long_close').rolling_min(window)).alias(f'price_range_{window}h'),
        ])
    
    # 2. Regime activity (instability indicators)
    for lookback in ACTIVITY_WINDOWS:
        df = df.with_columns([
            (pl.col('long_base').rolling_sum(lookback) + pl.col('short_base').rolling_sum(lookback)).alias(f'regime_activity_{lookback}h'),
            pl.col('net_bias_12').rolling_std(lookback).alias(f'bias_vol_{lookback}h'),
        ])
    
    # 3. Momentum consistency
    for window in MOMENTUM_WINDOWS:
        df = df.with_columns([
            (pl.col('long_ema_5') - pl.col('long_ema_5').shift(window)).alias(f'ema_slope_{window}h'),
        ])
        df = df.with_columns([
            pl.col(f'ema_slope_{window}h').rolling_std(window).alias(f'slope_consistency_{window}h'),
        ])
    
    # 4. Alignment features
    df = df.with_columns([
        (pl.col('net_bias_12').sign() == pl.col('net_bias_12').rolling_mean(STABILITY_ROLLING_MEAN_WINDOW).sign()).cast(pl.Int32).alias('bias_alignment'),
        (pl.col('long_close') > pl.col('long_ema_5')).cast(pl.Int32).alias('price_above_ema'),
        pl.min_horizontal(['last_long_age', 'last_short_age']).alias('min_regime_age'),
        (pl.col('last_long_age') / (pl.col('last_short_age') + 1)).alias('regime_age_ratio'),
    ])
    
    # 5. Create stability label (will regime persist?)
    df = df.with_columns([
        pl.when((pl.col('long_base') == 1) & (pl.col('short_base') == 0)).then(1)
          .when((pl.col('short_base') == 1) & (pl.col('long_base') == 0)).then(-1)
          .otherwise(0).alias('current_regime_sign')
    ])
    
    df = df.with_columns([
        pl.col('current_regime_sign').shift(-leakage_shift).alias('future_regime_sign')
    ])
    
    df = df.with_columns([
        pl.when((pl.col('current_regime_sign') != 0) & 
                (pl.col('current_regime_sign') == pl.col('future_regime_sign'))).then(1)
          .otherwise(0).alias('regime_persists')
    ])
    
    return df


def get_stability_features(df):
    '''
    Get list of stability feature column names from a dataframe.
    
    Args:
        df (pl.DataFrame): DataFrame with stability features
        
    Returns:
        list: List of stability feature column names
    '''
    stability_feature_patterns = ['vol_', 'range_', 'activity_', 'bias_vol_', 'slope_', 
                                 'consistency_', 'alignment', 'regime_age', 'min_regime_age']
    
    return [c for c in df.columns if any(pattern in c for pattern in stability_feature_patterns)]


def calculate_regime_quality_score(df, lookback=24):
    '''
    Calculate a composite regime quality score based on stability indicators.
    
    Higher scores indicate more stable/reliable regimes.
    
    Args:
        df (pl.DataFrame): DataFrame with stability features
        lookback (int): Lookback window for score calculation
        
    Returns:
        pl.DataFrame: DataFrame with added 'regime_quality_score' column
    '''
    # Normalize each component to 0-1 range
    df = df.with_columns([
        # Lower volatility = higher quality
        (1 - (pl.col(f'price_vol_{lookback}h') / pl.col(f'price_vol_{lookback}h').max())).alias('vol_score'),
        
        # Lower regime activity = higher quality
        (1 - (pl.col(f'regime_activity_{ACTIVITY_WINDOWS[-1]}h') / pl.col(f'regime_activity_{ACTIVITY_WINDOWS[-1]}h').max())).alias('activity_score'),
        
        # Better alignment = higher quality
        pl.col('bias_alignment').alias('alignment_score'),
        
        # Older regime = higher quality (more established)
        (pl.col('min_regime_age') / pl.col('min_regime_age').max()).alias('age_score'),
    ])
    
    # Composite score (equal weighting)
    df = df.with_columns([
        ((pl.col('vol_score') + pl.col('activity_score') + pl.col('alignment_score') + pl.col('age_score')) / 4).alias('regime_quality_score')
    ])
    
    # Drop intermediate scores
    df = df.drop(['vol_score', 'activity_score', 'alignment_score', 'age_score'])
    
    return df
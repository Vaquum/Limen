#!/usr/bin/env python3
'''
Feature Aliases - Create feature aliases with null filling

Creates aliased versions of dynamic features with appropriate default values
for cases where the original features might have null values.
'''

import polars as pl


def feature_aliases(df: pl.DataFrame, 
                   base_min_breakout: float = 0.005,
                   volatility_regime_enabled: bool = True) -> pl.DataFrame:
    '''
    Create feature aliases with null filling for dynamic and regime features.
    
    Args:
        df (pl.DataFrame): DataFrame with dynamic features
        base_min_breakout (float): Default value for dynamic_target
        volatility_regime_enabled (bool): Whether to create regime feature aliases
        
    Returns:
        pl.DataFrame: DataFrame with feature aliases
    '''
    
    # Create basic feature aliases
    df = df.with_columns([
        pl.col('dynamic_target').fill_null(base_min_breakout).alias('dynamic_target_feature'),
        pl.col('entry_score').fill_null(1.0).alias('entry_score_feature'),
        pl.col('momentum_score').fill_null(1.0).alias('momentum_score_feature')
    ])
    
    # Create volatility regime aliases if enabled
    if volatility_regime_enabled:
        df = df.with_columns([
            pl.col('vol_60h').fill_null(0).alias('vol_60h_feature'),
            pl.col('vol_percentile').fill_null(50).alias('vol_percentile_feature'),
            pl.col('regime_low').fill_null(0).alias('regime_low_feature'),
            pl.col('regime_normal').fill_null(1).alias('regime_normal_feature'),
            pl.col('regime_high').fill_null(0).alias('regime_high_feature')
        ])
    
    return df
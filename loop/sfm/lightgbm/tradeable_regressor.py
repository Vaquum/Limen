#!/usr/bin/env python3
'''
LightGBM Tradeable Regressor V3 Ultra - UEL Single File Model format
Ultra-simplified V3 with maximum code reduction while maintaining equivalent performance
'''

from loop.historical_data import HistoricalData
from loop.tests.utils.get_data import get_klines_data_large
from loop.features.market_regime import market_regime
from loop.features.momentum_confirmation import momentum_confirmation
from loop.indicators.returns import returns
from loop.manifest import Manifest
from loop.sfm.lightgbm.utils.tradeable_regressor import add_momentum_score_column
from loop.sfm.lightgbm.utils.tradeable_regressor import add_volatility_regime_columns
from loop.sfm.lightgbm.utils.tradeable_regressor import calculate_dynamic_parameters
from loop.sfm.lightgbm.utils.tradeable_regressor import calculate_microstructure_features
from loop.sfm.lightgbm.utils.tradeable_regressor import create_tradeable_labels
from loop.sfm.lightgbm.utils.tradeable_regressor import extend_data_dict
from loop.sfm.lightgbm.utils.tradeable_regressor import prepare_features_5m
from loop.sfm.lightgbm.utils.tradeable_regressor import simulate_exit_reality
from loop.sfm.model.lgb_tradeable_regression import lgb_tradeable_regression
from loop.utils.time_decay import time_decay


TRAIN_SPLIT = 70
VAL_SPLIT = 15
TEST_SPLIT = 15

CONFIG = {
    'kline_size': 300,
    'lookahead_minutes': 90,
    'base_min_breakout': 0.005,
    'max_positions': 1,
    'min_position_size': 20000,
    'prediction_threshold_percentile': 97,
    'ema_weight_power': 2.0,
    'volume_weight_enabled': True,
    'exit_on_target': True,
    'base_stop_loss': 0.0035,
    'trailing_stop': True,
    'trailing_stop_distance': 0.0025,
    'market_regime_filter': True,
    'position_sizing': 0.95,
    'dynamic_targets': True,
    'volatility_adjusted_stops': True,
    'microstructure_timing': True,
    'volatility_lookback': 48,
    'target_volatility_multiplier': 2.5,
    'stop_volatility_multiplier': 1.5,
    'target_clip_lower': 0.6,
    'target_clip_upper': 1.4,
    'stop_loss_clip_lower': 0.7,
    'stop_loss_clip_upper': 1.4,
    'entry_position_weight_base': 0.25,
    'entry_momentum_weight_base': 0.25,
    'entry_volume_weight_base': 0.25,
    'entry_spread_weight_base': 0.25,
    'entry_position_weight_low_vol': 0.35,
    'entry_momentum_weight_low_vol': 0.15,
    'entry_volume_weight_low_vol': 0.15,
    'entry_spread_weight_low_vol': 0.35,
    'entry_position_weight_high_vol': 0.15,
    'entry_momentum_weight_high_vol': 0.35,
    'entry_volume_weight_high_vol': 0.35,
    'entry_spread_weight_high_vol': 0.15,
    'entry_volume_spike_min': 0.5,
    'entry_volume_spike_max': 1.5,
    'entry_volume_spike_normalizer': 1.5,
    'entry_spread_ratio_min': 0,
    'entry_spread_ratio_max': 2,
    'regime_low_volatility_multiplier': 0.8,
    'regime_normal_volatility_multiplier': 1.0,
    'regime_high_volatility_multiplier': 1.2,
    'exit_quality_high': 1.0,
    'exit_quality_low': 0.2,
    'exit_quality_medium': 0.5,
    'volatility_scaling_factor': 100,
    'volatility_weight_min': 0.3,
    'volatility_weight_max': 1.0,
    'volume_weight_min': 0.5,
    'volume_weight_max': 2.0,
    'weight_target_achieved': 20,
    'weight_quick_target': 30,
    'weight_high_score_p90': 20,
    'weight_high_score_p95': 50,
    'weight_high_score_p99': 100,
    'weight_profitable_multiplier': 1.5,
    'volatility_lookback_candles': 720,
    'default_vol_percentile': 50.0,
    'default_volatility_regime': 'normal',
    'default_regime_low': 0,
    'default_regime_normal': 1,
    'default_regime_high': 0,
    'microstructure_volume_spike_period': 20,
    'microstructure_spread_mean_period': 48,
    'volume_weight_period': 20,
    'volatility_weight_period': 20,
    'momentum_weight_period': 12,
    'exit_reality_score_clip_min': -0.01,
    'exit_reality_score_clip_max': 0.02,
    'volume_ratio_period': 20,
    'volume_trend_short_period': 12,
    'volume_trend_long_period': 48,
    'feature_lookback_period': 48,
    'momentum_periods': [12, 24, 48],
    'rsi_periods': [12, 24, 48],
    'simple_momentum_confirmation': True,
    'exit_reality_blend': 0.25,
    'time_decay_blend': 0.25,
    'time_decay_halflife': 30,
    'commission_rate': 0.002,
    'num_boost_round': 300,
    'early_stopping_rounds': 30,
}


def params():
    p = {
        'objective': ['regression'],
        'metric': ['rmse'],
        'boosting_type': ['gbdt'],
        'num_leaves': [31, 35],
        'learning_rate': [0.05],
        'feature_fraction': [0.8],
        'bagging_fraction': [0.8],
        'bagging_freq': [5],
        'num_iterations': [100],
        'force_col_wise': [True],
    }
    return p


def manifest():

    return (Manifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'start_date_limit': '2025-01-01'}
        )
        .set_test_data_source(method=get_klines_data_large)
        .set_split_config(TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)
        .add_indicator(returns)
        .add_feature(add_volatility_regime_columns, config=CONFIG)
        .add_feature(market_regime, lookback=48)
        .add_feature(calculate_dynamic_parameters, config=CONFIG)
        .add_feature(calculate_microstructure_features, config=CONFIG)
        .add_feature(momentum_confirmation, short_period=1, long_period=3, short_weight=0.5)
        .add_feature(add_momentum_score_column, config=CONFIG)
        .with_target('tradeable_score')
            .add_transform(simulate_exit_reality, config=CONFIG)
            .add_transform(time_decay, time_column='exit_bars', halflife=CONFIG['time_decay_halflife'],
                          time_units=5, output_column='time_decay_factor')
            .add_transform(create_tradeable_labels, config=CONFIG)
            .add_transform(prepare_features_5m, config=CONFIG)
            .done()
        .add_to_data_dict(extend_data_dict)
        .with_model(lgb_tradeable_regression)
    )

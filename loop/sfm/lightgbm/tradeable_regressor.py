#!/usr/bin/env python3
'''
LightGBM Tradeable Regressor V3 Ultra - UEL Single File Model format
Ultra-simplified V3 with maximum code reduction while maintaining equivalent performance
'''

import lightgbm as lgb
import polars as pl
import numpy as np

from loop.utils.splits import split_sequential, split_data_to_prep_output
from loop.utils.calculate_returns import calculate_returns_if_missing
from loop.utils.standardize_datetime import standardize_datetime_column
from loop.utils.numeric_features import get_numeric_feature_columns

from loop.features.market_regime import market_regime
from loop.features.momentum_confirmation import momentum_confirmation
from loop.utils.time_decay import time_decay
from loop.sfm.lightgbm.utils.tradeable_regressor import (
    calculate_dynamic_parameters,
    calculate_microstructure_features,
    simulate_exit_reality,
    create_tradeable_labels,
    prepare_features_5m
)
from loop.sfm.lightgbm.utils.param_filtering import filter_lgb_params


TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


EXCLUDE_CATEGORIES = {
    'basic': ['datetime', 'open', 'high', 'low', 'close', 'volume'],
    'targets': ['tradeable_breakout', 'tradeable_score', 'tradeable_score_base', 'capturable_breakout'],
    'indicators': ['max_drawdown', 'ema', 'future_high', 'future_low', 'ema_alignment', 'volume_weight', 'volatility_weight'],
    'features': ['momentum_weight', 'market_favorable', 'risk_reward_ratio', 'sma_20', 'sma_50', 'trend_strength'],
    'volatility': ['volatility_ratio', 'volume_sma', 'volume_regime', 'rolling_volatility', 'atr', 'atr_pct', 'volatility_measure'],
    'dynamics': ['dynamic_target', 'dynamic_stop_loss', 'entry_score', 'position_in_candle'],
    'micro': ['micro_momentum', 'volume_spike', 'spread_pct', 'achieves_dynamic_target'],
    'range': ['high_low', 'high_close', 'low_close', 'true_range'],
    'momentum': ['momentum_1', 'momentum_3', 'momentum_score', 'volume_ma', 'volatility'],
    'exit': ['exit_gross_return', 'exit_net_return', 'exit_reason', 'exit_bars', 'exit_max_return', 'exit_min_return'],
    'reality': ['time_decay_factor', 'exit_reality_score', 'exit_quality', 'exit_reality_time_decayed', 'exit_on_prediction_drop'],
    'position': ['close_to_high', 'close_to_low']
}

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
        'verbose': [-1],
        'num_iterations': [100],
        'force_col_wise': [True],
    }
    return p


def prep(data, round_params=None):

    all_datetimes = data['datetime'].to_list()
    
    if not isinstance(data, pl.DataFrame):
        raise ValueError("Data must be a Polars DataFrame")
    
    df = data.clone()
    
    df = standardize_datetime_column(df)
    
    df = calculate_returns_if_missing(df)
    
    df = df.with_columns([
        pl.col('returns').rolling_std(CONFIG['volatility_lookback_candles'], min_samples=1).alias('vol_60h'),
        pl.lit(CONFIG['default_vol_percentile']).alias('vol_percentile'),
        pl.lit(CONFIG['default_volatility_regime']).alias('volatility_regime'),
        pl.lit(CONFIG['default_regime_low']).alias('regime_low'),
        pl.lit(CONFIG['default_regime_normal']).alias('regime_normal'),
        pl.lit(CONFIG['default_regime_high']).alias('regime_high')
    ])
    
    df = market_regime(df, 48)
    df = calculate_dynamic_parameters(df, CONFIG)
    df = calculate_microstructure_features(df, CONFIG)
    
    if CONFIG['simple_momentum_confirmation']:
        df = momentum_confirmation(df, short_period=1, long_period=3, short_weight=0.5)
    else:
        df = df.with_columns([pl.lit(1.0).alias('momentum_score')])
    
    df = simulate_exit_reality(df, CONFIG)
    df = time_decay(df, 'exit_bars', CONFIG['time_decay_halflife'], time_units=5, output_column='time_decay_factor')
    df = create_tradeable_labels(df, CONFIG)
    
    df = prepare_features_5m(df, config=CONFIG)
    
    df_clean = df.drop_nulls()
    
    if len(df_clean) == 0:
        raise ValueError("No data left after cleaning (dropna)")
    
    exclude_cols = [col for category in EXCLUDE_CATEGORIES.values() for col in category]
    
    numeric_features = get_numeric_feature_columns(df_clean, exclude_cols)
    
    cols = ['datetime'] + numeric_features + ['tradeable_score']
    
    split_data = split_sequential(df_clean, ratios=(TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT))
    
    exit_reality_cols = ['exit_gross_return', 'exit_net_return', 'exit_reason', 
                        'exit_bars', 'exit_max_return', 'exit_min_return',
                        'time_decay_factor', 'exit_reality_score', 'exit_quality',
                        'exit_reality_time_decayed', 'achieves_dynamic_target', 'tradeable_score']
    
    cols_to_drop = [col for col in exit_reality_cols if col in split_data[2].columns]
    test_tradeable_scores = split_data[2].select('tradeable_score').to_numpy().flatten()
    test_clean_for_backtest = split_data[2].drop(cols_to_drop).with_columns(pl.lit(0.0).alias('tradeable_score'))
    split_data[2] = split_data[2].drop(cols_to_drop).with_columns(pl.lit(0.0).alias('tradeable_score'))
    
    data_dict = split_data_to_prep_output(split_data, cols, all_datetimes)
    
    data_dict['_feature_names'] = numeric_features
    data_dict['_numeric_features'] = numeric_features
    data_dict['_train_clean'] = split_data[0]
    data_dict['_val_clean'] = split_data[1]
    data_dict['_test_clean'] = test_clean_for_backtest
    data_dict['_test_tradeable_scores'] = test_tradeable_scores
    
    data_dict['dtrain'] = lgb.Dataset(data_dict['x_train'], label=data_dict['y_train'])
    data_dict['dval'] = lgb.Dataset(data_dict['x_val'], label=data_dict['y_val'], reference=data_dict['dtrain'])
    
    return data_dict


def model(data, round_params):
    lgb_params = filter_lgb_params(round_params)
    lgb_params['verbose'] = -1
    
    train_data = data['_train_clean']
    arrays = train_data.select(['achieves_dynamic_target', 'exit_bars', 'tradeable_score', 'exit_net_return']).to_numpy()
    achieved, exit_bars, y_values, net_returns = arrays[:, 0].astype(bool), arrays[:, 1], arrays[:, 2], arrays[:, 3]
    profitable = net_returns > 0.001
    
    weights = np.ones(len(train_data))
    p90, p95, p99 = np.percentile(y_values, [90, 95, 99])
    
    weights[achieved] = CONFIG['weight_target_achieved']
    weights[achieved & (exit_bars <= 6)] = CONFIG['weight_quick_target']
    weights[y_values > p90] = CONFIG['weight_high_score_p90']
    weights[y_values > p95] = CONFIG['weight_high_score_p95']
    weights[y_values > p99] = CONFIG['weight_high_score_p99']
    weights[profitable] *= CONFIG['weight_profitable_multiplier']
    
    numeric_features = data['_numeric_features']
    X_train = train_data.select(numeric_features).to_numpy()
    X_val = data['_val_clean'].select(numeric_features).to_numpy()
    y_val = data['_val_clean'].select('tradeable_score').to_numpy().flatten()
    
    data['dtrain'] = lgb.Dataset(X_train, label=y_values, weight=weights)
    data['dval'] = lgb.Dataset(X_val, label=y_val, reference=data['dtrain'])
    
    evals_result = {}
    model = lgb.train(
        params=lgb_params,
        train_set=data['dtrain'],
        num_boost_round=CONFIG['num_boost_round'],
        valid_sets=[data['dtrain'], data['dval']],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(stopping_rounds=CONFIG['early_stopping_rounds'], verbose=False),
                   lgb.record_evaluation(evals_result)]
    )
    
    test_clean = data['_test_clean'] if '_test_clean' in data else None
    y_pred = model.predict(test_clean.select(numeric_features).to_numpy()) if test_clean is not None else model.predict(data['x_test'])
    data['_preds'] = y_pred
    
    val_rmse = float(evals_result['val']['rmse'][-1])
    n_samples = len(data['x_train'])
    
    return {
        'models': [model], 'val_rmse': val_rmse, 'n_regimes_trained': 0, '_preds': y_pred,
        'universal_val_rmse': val_rmse, 'universal_samples': n_samples,
        'extras': {
            'regime_models': {'universal': model}, 'test_predictions': y_pred,
            'test_clean': data['_test_clean'], 'test_tradeable_scores': data.get('_test_tradeable_scores', None),
            'numeric_features': numeric_features,
            'regime_metrics': {'universal': {'samples': n_samples, 'final_val_rmse': val_rmse}}
        }
    }
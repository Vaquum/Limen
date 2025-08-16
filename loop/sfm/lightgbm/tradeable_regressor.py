#!/usr/bin/env python3
'''
LightGBM Tradeable Regressor - UEL Single File Model format
Implements regime-aware LightGBM with tradeable label creation
'''

import lightgbm as lgb
import polars as pl
import numpy as np

from loop.utils.splits import split_sequential, split_data_to_prep_output

from loop.sfm.lightgbm.utils.tradeable_regressor import (
    calculate_volatility_regime,
    calculate_market_regime,
    calculate_dynamic_parameters,
    calculate_microstructure_features,
    calculate_simple_momentum_confirmation,
    simulate_exit_reality,
    calculate_time_decay_factor,
    create_tradeable_labels,
    prepare_features_5m
)

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

WEIGHT_TARGET_ACHIEVED = 20
WEIGHT_QUICK_TARGET = 30
WEIGHT_HIGH_SCORE_P90 = 20
WEIGHT_HIGH_SCORE_P95 = 50
WEIGHT_HIGH_SCORE_P99 = 100
WEIGHT_PROFITABLE_MULTIPLIER = 1.5

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
    'simple_momentum_confirmation': True,
    'exit_reality_blend': 0.3,
    'time_decay_blend': 0.3,
    'time_decay_halflife': 30,
    'volatility_regime_enabled': True,
    'vol_regime_lookback': 720,
    'vol_low_percentile': 20,
    'vol_high_percentile': 80,
    'commission_rate': 0.0015,
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
        'train_regime_models': [True],
    }
    return p


def prep(data, round_params=None):

    all_datetimes = data['datetime'].to_list()
    
    # Ensure we're working with Polars
    if not isinstance(data, pl.DataFrame):
        raise ValueError("Data must be a Polars DataFrame")
    
    df = data.clone()
    
    if 'Date' in df.columns:
        df = df.rename({'Date': 'datetime'})
    
    if df.schema['datetime'] != pl.Datetime:
        df = df.with_columns(pl.col('datetime').str.to_datetime())
    
    
    df = calculate_volatility_regime(df, CONFIG)
    df = calculate_market_regime(df)
    df = calculate_dynamic_parameters(df, CONFIG)
    df = calculate_microstructure_features(df, CONFIG)
    df = calculate_simple_momentum_confirmation(df, CONFIG)
    
    df = simulate_exit_reality(df, CONFIG)
    df = calculate_time_decay_factor(df, CONFIG)
    df = create_tradeable_labels(df, CONFIG)
    
    df = prepare_features_5m(df, config=CONFIG)
    
    df_clean = df.drop_nulls()
    
    if len(df_clean) == 0:
        raise ValueError("No data left after cleaning (dropna)")
    
    exclude_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume',
                    'tradeable_breakout', 'tradeable_score', 'tradeable_score_v6',
                    'tradeable_score_base', 'capturable_breakout', 'max_drawdown', 'ema', 'future_high', 'future_low',
                    'ema_alignment', 'volume_weight', 'volatility_weight',
                    'momentum_weight', 'market_favorable', 'risk_reward_ratio',
                    'sma_20', 'sma_50', 'trend_strength', 'volatility_ratio',
                    'volume_sma', 'volume_regime', 'dynamic_target',
                    'dynamic_stop_loss', 'entry_score', 'position_in_candle',
                    'micro_momentum', 'volume_spike', 'spread_pct', 'rolling_volatility',
                    'atr', 'atr_pct', 'volatility_measure', 'achieves_dynamic_target',
                    'high_low', 'high_close', 'low_close', 'true_range',
                    'momentum_1', 'momentum_3', 'momentum_score', 'volume_ma', 'volatility',
                    'exit_gross_return', 'exit_net_return', 'exit_reason', 'exit_bars',
                    'exit_max_return', 'exit_min_return', 'time_decay_factor',
                    'exit_reality_score', 'exit_quality', 'exit_reality_time_decayed',
                    'exit_on_prediction_drop', 'vol_60h', 'vol_percentile',
                    'spread', 'position_in_range', 'close_to_high', 'close_to_low']
    
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    numeric_features = [col for col in feature_cols 
                       if df_clean.schema[col] in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]]
    
    
    # Create column list with target as LAST column
    cols = ['datetime'] + numeric_features + ['tradeable_score']
    
    split_data = split_sequential(df_clean, ratios=(TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT))
    
    test_tradeable_scores = split_data[2].select('tradeable_score').to_numpy().flatten()
    
    exit_reality_cols = ['exit_gross_return', 'exit_net_return', 'exit_reason', 
                        'exit_bars', 'exit_max_return', 'exit_min_return',
                        'time_decay_factor', 'exit_reality_score', 'exit_quality',
                        'exit_reality_time_decayed', 'achieves_dynamic_target', 'tradeable_score']
    
    test_clean_for_backtest = split_data[2].drop([col for col in exit_reality_cols if col in split_data[2].columns])
    test_clean_for_backtest = test_clean_for_backtest.with_columns(pl.lit(0.0).alias('tradeable_score'))
    
    split_data[2] = split_data[2].drop([col for col in exit_reality_cols if col in split_data[2].columns])
    
    split_data[2] = split_data[2].with_columns(pl.lit(0.0).alias('tradeable_score'))
    
    # Use split_data_to_prep_output
    data_dict = split_data_to_prep_output(split_data, cols, all_datetimes)
    
    data_dict['_feature_names'] = numeric_features
    data_dict['_numeric_features'] = numeric_features
    data_dict['_train_clean'] = split_data[0]
    data_dict['_val_clean'] = split_data[1]
    data_dict['_test_clean'] = test_clean_for_backtest
    data_dict['_test_tradeable_scores'] = test_tradeable_scores
    
    data_dict['_train_regimes'] = split_data[0].select('volatility_regime').to_numpy().flatten()
    data_dict['_val_regimes'] = split_data[1].select('volatility_regime').to_numpy().flatten()
    
    data_dict['dtrain'] = lgb.Dataset(data_dict['x_train'], label=data_dict['y_train'])
    data_dict['dval'] = lgb.Dataset(data_dict['x_val'], label=data_dict['y_val'], reference=data_dict['dtrain'])
    
    return data_dict


def model(data, round_params):
    round_params = round_params.copy()
    
    train_regime_models = round_params.pop('train_regime_models', True)
    
    round_params.pop('_experiment_details', None)
    lgb_params = {k: v for k, v in round_params.items() 
                  if k in ['objective', 'metric', 'boosting_type', 'num_leaves',
                          'learning_rate', 'feature_fraction', 'bagging_fraction',
                          'bagging_freq', 'verbose', 'num_iterations', 'force_col_wise']}
    
    if train_regime_models:
        models = {}
        regime_metrics = {}
        
        train_clean = data['_train_clean']
        val_clean = data['_val_clean']
        numeric_features = data['_numeric_features']
        for regime in ['low', 'normal', 'high']:
            regime_data = train_clean.filter(pl.col('volatility_regime') == regime)
            
            regime_val_data = val_clean.filter(pl.col('volatility_regime') == regime)
            
            if len(regime_data) > 1000 and len(regime_val_data) > 100:
                    
                regime_train_set = regime_data
                regime_val_set = regime_val_data
                
                X_train = regime_train_set.select(numeric_features).to_numpy()
                y_train = regime_train_set.select('tradeable_score').to_numpy().flatten()
                X_val = regime_val_set.select(numeric_features).to_numpy()
                y_val = regime_val_set.select('tradeable_score').to_numpy().flatten()
                
                weights_train = np.ones(len(y_train))
                
                achieved_target = regime_train_set.select('achieves_dynamic_target').to_numpy().flatten()
                weights_train[achieved_target] = WEIGHT_TARGET_ACHIEVED
                
                exit_bars = regime_train_set.select('exit_bars').to_numpy().flatten()
                quick_targets = achieved_target & (exit_bars <= 6)
                weights_train[quick_targets] = WEIGHT_QUICK_TARGET
                
                weights_train[y_train > np.percentile(y_train, 90)] = WEIGHT_HIGH_SCORE_P90
                weights_train[y_train > np.percentile(y_train, 95)] = WEIGHT_HIGH_SCORE_P95
                weights_train[y_train > np.percentile(y_train, 99)] = WEIGHT_HIGH_SCORE_P99
                
                profitable_exits = (regime_train_set.select('exit_net_return').to_numpy().flatten() > 0.001)
                weights_train[profitable_exits] *= WEIGHT_PROFITABLE_MULTIPLIER
                
                train_data = lgb.Dataset(X_train, label=y_train, weight=weights_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                lgb_params = lgb_params.copy()
                lgb_params.update({
                    'verbose': -1,
                })

                evals_result = {}
                model = lgb.train(
                    params=lgb_params,
                    train_set=train_data,
                    num_boost_round=CONFIG['num_boost_round'],
                    valid_sets=[train_data, val_data],
                    valid_names=['train', 'val'],
                    callbacks=[lgb.early_stopping(stopping_rounds=CONFIG['early_stopping_rounds'], verbose=False),
                               lgb.record_evaluation(evals_result)]
                )
                
                models[regime] = model
                metric_name = list(evals_result['train'].keys())[0] if evals_result['train'] else 'rmse'
                regime_metrics[regime] = {
                    'samples': len(regime_data),
                    'final_train_rmse': float(evals_result['train'][metric_name][-1]),
                    'final_val_rmse': float(evals_result['val'][metric_name][-1])
                }
                        
        train_set = train_clean
        val_set = val_clean
        
        weights_train_universal = np.ones(len(train_set))
        
        achieved_target_universal = train_set.select('achieves_dynamic_target').to_numpy().flatten()
        weights_train_universal[achieved_target_universal] = WEIGHT_TARGET_ACHIEVED
        
        exit_bars = train_set.select('exit_bars').to_numpy().flatten()
        quick_targets = achieved_target_universal & (exit_bars <= 6)
        weights_train_universal[quick_targets] = WEIGHT_QUICK_TARGET
        
        y_train_universal = train_set.select('tradeable_score').to_numpy().flatten()
        weights_train_universal[y_train_universal > np.percentile(y_train_universal, 90)] = WEIGHT_HIGH_SCORE_P90
        weights_train_universal[y_train_universal > np.percentile(y_train_universal, 95)] = WEIGHT_HIGH_SCORE_P95
        weights_train_universal[y_train_universal > np.percentile(y_train_universal, 99)] = WEIGHT_HIGH_SCORE_P99
        
        profitable_exits = (train_set.select('exit_net_return').to_numpy().flatten() > 0.001)
        weights_train_universal[profitable_exits] *= WEIGHT_PROFITABLE_MULTIPLIER
        X_train_universal = train_set.select(numeric_features).to_numpy()
        y_train_universal = train_set.select('tradeable_score').to_numpy().flatten()
        X_val_universal = val_set.select(numeric_features).to_numpy()
        y_val_universal = val_set.select('tradeable_score').to_numpy().flatten()
        
        dtrain_universal = lgb.Dataset(X_train_universal, label=y_train_universal, weight=weights_train_universal)
        dval_universal = lgb.Dataset(X_val_universal, label=y_val_universal, reference=dtrain_universal)
        
        lgb_params = lgb_params.copy()
        lgb_params.update({
            'verbose': -1,
        })
        
        evals_result_universal = {}
        universal_model = lgb.train(
            params=lgb_params,
            train_set=dtrain_universal,
            num_boost_round=CONFIG['num_boost_round'],
            valid_sets=[dtrain_universal, dval_universal],
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(stopping_rounds=CONFIG['early_stopping_rounds'], verbose=False),
                       lgb.record_evaluation(evals_result_universal)]
        )
        
        models['universal'] = universal_model
        metric_name_universal = list(evals_result_universal['train'].keys())[0] if evals_result_universal['train'] else 'rmse'
        regime_metrics['universal'] = {
            'samples': len(train_set),
            'final_train_rmse': float(evals_result_universal['train'][metric_name_universal][-1]),
            'final_val_rmse': float(evals_result_universal['val'][metric_name_universal][-1])
        }
        
        test_clean = data['_test_clean']
        
        if 'volatility_regime' in test_clean.columns:
            y_pred = np.zeros(len(test_clean))
            for regime in ['low', 'normal', 'high']:
                regime_mask = test_clean.get_column('volatility_regime') == regime
                regime_indices = [i for i, mask_val in enumerate(regime_mask) if mask_val]
                
                if len(regime_indices) > 0:
                    if regime in models:
                        model = models[regime]
                    elif 'universal' in models:
                        model = models['universal']
                    else:
                        continue
                    
                    regime_data = test_clean.filter(pl.col('volatility_regime') == regime)
                    X_regime = regime_data.select(numeric_features).to_numpy()
                    
                    regime_preds = model.predict(X_regime)
                    
                    for idx, pred in zip(regime_indices, regime_preds):
                        y_pred[idx] = pred
        else:
            X_test = test_clean.select(numeric_features).to_numpy()
            y_pred = models['universal'].predict(X_test)
        
    else:
        numeric_features = data['_numeric_features']
        
        train_clean = data['_train_clean']
        train_set = train_clean
        
        weights_train = np.ones(len(train_set))
        
        achieved_target = train_set.select('achieves_dynamic_target').to_numpy().flatten()
        weights_train[achieved_target] = WEIGHT_TARGET_ACHIEVED
        
        exit_bars = train_set.select('exit_bars').to_numpy().flatten()
        quick_targets = achieved_target & (exit_bars <= 6)
        weights_train[quick_targets] = WEIGHT_QUICK_TARGET
        
        y_train = train_set.select('tradeable_score').to_numpy().flatten()
        weights_train[y_train > np.percentile(y_train, 90)] = WEIGHT_HIGH_SCORE_P90
        weights_train[y_train > np.percentile(y_train, 95)] = WEIGHT_HIGH_SCORE_P95
        weights_train[y_train > np.percentile(y_train, 99)] = WEIGHT_HIGH_SCORE_P99
        
        profitable_exits = (train_set.select('exit_net_return').to_numpy().flatten() > 0.001)
        weights_train[profitable_exits] *= WEIGHT_PROFITABLE_MULTIPLIER
        data['dtrain'] = lgb.Dataset(data['x_train'], label=data['y_train'], weight=weights_train)
        data['dval'] = lgb.Dataset(data['x_val'], label=data['y_val'], reference=data['dtrain'])
        
        lgb_params = lgb_params.copy()
        lgb_params.update({
            'verbose': -1,
        })
        
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
        
        if '_test_clean' in data:
            test_clean = data['_test_clean']
            X_test = test_clean.select(numeric_features).to_numpy()
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(data['x_test'])
        metric_name = list(evals_result['train'].keys())[0] if evals_result['train'] else 'rmse'
        regime_metrics = {
            'universal': {
                'samples': len(data['x_train']),
                'final_train_rmse': float(evals_result['train'][metric_name][-1]),
                'final_val_rmse': float(evals_result['val'][metric_name][-1])
            }
        }
    
    data['_preds'] = y_pred
    
    if 'universal' in regime_metrics:
        val_rmse = regime_metrics['universal']['final_val_rmse']
    elif regime_metrics:
        val_rmse = list(regime_metrics.values())[0]['final_val_rmse']
    else:
        raise ValueError("No models were trained - regime_metrics is empty")
    
    round_results = {
        'models': [models if train_regime_models else model],
        'val_rmse': val_rmse,
        'n_regimes_trained': len([r for r in regime_metrics if r != 'universal']),
    }
    
    for regime in ['low', 'normal', 'high', 'universal']:
        if regime in regime_metrics:
            round_results[f'{regime}_val_rmse'] = regime_metrics[regime]['final_val_rmse']
            round_results[f'{regime}_samples'] = regime_metrics[regime]['samples']
        else:
            round_results[f'{regime}_val_rmse'] = None
            round_results[f'{regime}_samples'] = 0
    
    round_results['extras'] = {
        'regime_models': models if train_regime_models else {'universal': model},
        'test_predictions': y_pred,
        'test_clean': data['_test_clean'],
        'test_tradeable_scores': data.get('_test_tradeable_scores', None),
        'numeric_features': data['_numeric_features'],
        'regime_metrics': regime_metrics
    }
    
    if train_regime_models:
        regime_predictions = {}
        test_clean = data['_test_clean']
        
        for regime in ['low', 'normal', 'high']:
            regime_test = test_clean.filter(pl.col('volatility_regime') == regime)
            if len(regime_test) > 0 and regime in models:
                X_regime = regime_test.select(numeric_features).to_numpy()
                regime_preds = models[regime].predict(X_regime)
                regime_mask = test_clean.get_column('volatility_regime') == regime
                indices = [i for i, val in enumerate(regime_mask) if val]
                regime_scores = None
                if data.get('_test_tradeable_scores') is not None:
                    regime_scores = np.array([data['_test_tradeable_scores'][i] for i in indices])
                
                regime_predictions[regime] = {
                    'predictions': regime_preds,
                    'indices': indices,
                    'scores': regime_scores
                }
            elif len(regime_test) > 0 and 'universal' in models:
                X_regime = regime_test.select(numeric_features).to_numpy()
                regime_preds = models['universal'].predict(X_regime)
                regime_mask = test_clean.get_column('volatility_regime') == regime
                indices = [i for i, val in enumerate(regime_mask) if val]
                regime_scores = None
                if data.get('_test_tradeable_scores') is not None:
                    regime_scores = np.array([data['_test_tradeable_scores'][i] for i in indices])
                
                regime_predictions[regime] = {
                    'predictions': regime_preds,
                    'indices': indices,
                    'scores': regime_scores
                }
        
        round_results['extras']['regime_predictions'] = regime_predictions
    
    return round_results
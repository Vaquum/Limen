'''
Tradeline Long Binary - Long-Only Binary Trading System
Implements line-based binary long-only trading predictions with simplified exit strategy
Integrates ATR-based risk management for single position long-only trades
'''
from typing import Dict, List, Any, Optional

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import classification_report

from loop.utils.splits import split_sequential, split_data_to_prep_output
from loop.metrics.binary_metrics import binary_metrics
from loop.sfm.lightgbm.utils.tradeline_long_binary import (
    find_price_lines,
    filter_lines_by_quantile,
    compute_line_features,
    compute_temporal_features,
    compute_price_features,
    create_binary_labels,
    apply_class_weights,
    compute_quantile_line_features,
    calculate_atr,
    apply_long_only_exit_strategy
)

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

EXCLUDE_CATEGORIES = {
    'basic': ['datetime', 'open', 'high', 'low', 'close', 'volume'],
    'targets': ['label'],
    'intermediate': ['hl', 'hpc', 'lpc', 'tr', 'atr']
}

CONFIG = {
    'min_height_pct_default': 0.003,
    'max_duration_hours_default': 48,
    'lookahead_hours_default': 48,
    'long_threshold_percentile_default': 75,
    'quantile_threshold_default': 0.75,
    'default_threshold': 0.034,
    'atr_period': 24,
    'confidence_threshold': 0.45,
    'position_size': 0.199,
    'max_positions': 1,
    'min_stop_loss': 0.01,
    'max_stop_loss': 0.04,
    'atr_stop_multiplier': 1.5,
    'trailing_activation': 0.02,
    'trailing_distance': 0.5,
    'loser_timeout_hours': 24,
    'max_hold_hours': 48,
    'initial_capital': 100000.0,
    'default_atr_pct': 0.015,
    'batch_size': 100,
    'max_hours_since_line': 48,
    'momentum_lookback_hours': 6,
    'density_lookback_hours': 48,
    'height_lookback_hours': 24,
    'early_stopping_rounds': 50,
    'log_evaluation_period': 0,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'random_state': 42,
    'use_calibration_default': True,
    'calibration_method_default': 'isotonic',
    'calibration_cv_default': 3
}


def params() -> dict:
    p = {
        'quantile_threshold': [0.60, 0.70, 0.75, 0.80, 0.85],
        'min_height_pct': [0.001, 0.002, 0.003, 0.004, 0.005],
        'max_duration_hours': [24, 48, 72, 96],
        'lookahead_hours': [24, 48, 72],
        'long_threshold_percentile': [65, 75, 85],
        'loser_timeout_hours': [12, 24, 48],
        'max_hold_hours': [24, 48, 72, 96],
        'confidence_threshold': [0.40, 0.50, 0.60],
        'position_size': [0.10, 0.20, 0.30],
        'min_stop_loss': [0.005, 0.010, 0.020],
        'max_stop_loss': [0.030, 0.040, 0.050],
        'atr_stop_multiplier': [1.0, 1.5, 2.0],
        'trailing_activation': [0.01, 0.02, 0.03],
        'trailing_distance': [0.3, 0.5, 0.7],
        'default_atr_pct': [0.010, 0.015, 0.020],
        'max_hours_since_line': [24, 48, 72],
        'num_leaves': [31, 63, 127],  
        'learning_rate': [0.05, 0.1],  
        'feature_fraction': [0.9],  
        'bagging_fraction': [0.8, 0.9],  
        'bagging_freq': [5],  
        'min_child_samples': [10, 20],  
        'lambda_l1': [0, 0.1],  
        'lambda_l2': [0, 0.1],  
        'n_estimators': [500]
    }
    return p


def prep(data: pl.DataFrame, round_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    all_datetimes = data['datetime'].to_list()
    
    if not isinstance(data, pl.DataFrame):
        raise ValueError("Data must be a Polars DataFrame")
    
    df = data.clone()
    
    if round_params is None:
        round_params = {}
    
    quantile_threshold = round_params.get('quantile_threshold', CONFIG['quantile_threshold_default'])
    min_height_pct = round_params.get('min_height_pct', CONFIG['min_height_pct_default'])
    max_duration_hours = round_params.get('max_duration_hours', CONFIG['max_duration_hours_default'])
    lookahead_hours = round_params.get('lookahead_hours', CONFIG['lookahead_hours_default'])
    long_threshold_percentile = round_params.get('long_threshold_percentile', CONFIG['long_threshold_percentile_default'])
    long_lines, short_lines = find_price_lines(df, max_duration_hours, min_height_pct)
    
    long_lines_filtered = filter_lines_by_quantile(long_lines, quantile_threshold)
    short_lines_filtered = filter_lines_by_quantile(short_lines, quantile_threshold)
    
    if long_lines:
        long_heights = [abs(line['height_pct']) for line in long_lines]
        long_threshold = np.percentile(long_heights, long_threshold_percentile)
    else:
        long_threshold = CONFIG['default_threshold']
    df = compute_temporal_features(df)
    
    df = compute_price_features(df)
    
    df = compute_line_features(df, long_lines_filtered, short_lines_filtered)
    
    df = compute_quantile_line_features(df, long_lines_filtered, short_lines_filtered, quantile_threshold)
    
    df = calculate_atr(df, period=CONFIG['atr_period'])
    
    df = create_binary_labels(df, long_threshold, lookahead_hours)
    
    df_clean = df.drop_nulls()
    
    if len(df_clean) == 0:
        raise ValueError("No data left after cleaning")
    exclude_cols = []
    for category in EXCLUDE_CATEGORIES.values():
        exclude_cols.extend(category)
    
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    numeric_features = []
    for col in feature_cols:
        if df_clean.schema[col] in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]:
            numeric_features.append(col)
    
    cols = ['datetime'] + numeric_features + ['label']
    
    split_data = split_sequential(df_clean, ratios=[int(TRAIN_SPLIT * 100), int(VAL_SPLIT * 100), int(TEST_SPLIT * 100)])
    
    data_dict = split_data_to_prep_output(split_data, cols, all_datetimes)
    data_dict['x_train'] = data_dict['x_train'].to_numpy()
    data_dict['y_train'] = data_dict['y_train'].to_numpy()
    data_dict['x_val'] = data_dict['x_val'].to_numpy()
    data_dict['y_val'] = data_dict['y_val'].to_numpy()
    data_dict['x_test'] = data_dict['x_test'].to_numpy()
    data_dict['y_test'] = data_dict['y_test'].to_numpy()
    
    data_dict['_feature_names'] = numeric_features
    data_dict['_quantile_threshold'] = quantile_threshold
    data_dict['_long_threshold'] = long_threshold
    data_dict['_long_threshold_percentile'] = long_threshold_percentile
    
    data_dict['_long_lines_filtered'] = long_lines_filtered
    data_dict['_short_lines_filtered'] = short_lines_filtered
    scaler = StandardScaler()
    data_dict['x_train'] = scaler.fit_transform(data_dict['x_train'])
    data_dict['x_val'] = scaler.transform(data_dict['x_val'])
    data_dict['x_test'] = scaler.transform(data_dict['x_test'])
    data_dict['_scaler'] = scaler
    
    full_df_for_trading = df_clean.select(['datetime', 'open', 'high', 'low', 'close', 'volume', 'atr_pct', 'label'])
    data_dict['_original_df'] = full_df_for_trading
    
    sample_weights = apply_class_weights(data_dict['y_train'])
    
    data_dict['dtrain'] = lgb.Dataset(data_dict['x_train'], label=data_dict['y_train'], weight=sample_weights)
    data_dict['dval'] = lgb.Dataset(data_dict['x_val'], label=data_dict['y_val'], reference=data_dict['dtrain'])
    
    return data_dict


def model(data: Dict[str, Any], round_params: Dict[str, Any]) -> Dict[str, Any]:
    use_calibration = round_params.get('use_calibration', CONFIG['use_calibration_default'])
    n_estimators = round_params.get('n_estimators', 500)
    decision_threshold = round_params.get('decision_threshold', 0.45)
    
    lgb_params = {
        'objective': CONFIG['objective'],
        'metric': CONFIG['metric'],
        'boosting_type': CONFIG['boosting_type'],
        'num_leaves': round_params.get('num_leaves', 31),
        'learning_rate': round_params.get('learning_rate', 0.05),
        'feature_fraction': round_params.get('feature_fraction', 0.9),
        'bagging_fraction': round_params.get('bagging_fraction', 0.8),
        'bagging_freq': round_params.get('bagging_freq', 5),
        'min_child_samples': round_params.get('min_child_samples', 20),
        'lambda_l1': round_params.get('lambda_l1', 0),
        'lambda_l2': round_params.get('lambda_l2', 0),
        'verbose': CONFIG['verbose'],
        'random_state': CONFIG['random_state']
    }
    X_test = data['x_test']
    X_val = data['x_val']
    y_test = data['y_test']
    y_val = data['y_val']
    y_train = data['y_train']
    
    train_data = data['dtrain']
    val_data = data['dval']
    evals_result: Dict[str, Dict[str, List[float]]] = {}
    
    lgb_model = lgb.train(
        params=lgb_params,
        train_set=train_data,
        num_boost_round=n_estimators,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=CONFIG['early_stopping_rounds'], verbose=False),
            lgb.record_evaluation(evals_result),
            lgb.log_evaluation(period=CONFIG['log_evaluation_period'])
        ]
    )
    
    final_val_loss = evals_result['val']['binary_logloss'][-1]
    
    y_pred_proba = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    y_pred = (y_pred_proba > decision_threshold).astype(int)
    if use_calibration:
        val_pred_proba = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(val_pred_proba, y_val)
        
        calibrated_proba = iso_reg.transform(y_pred_proba)
        
        y_proba = calibrated_proba
        y_pred = (calibrated_proba > decision_threshold).astype(int)
        
        final_model = lgb_model
    else:
        y_proba = y_pred_proba
        y_pred = (y_pred_proba > decision_threshold).astype(int)
        final_model = lgb_model
    metrics = binary_metrics(data, y_pred, y_proba)
    
    metrics['val_loss'] = float(final_val_loss)
    
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    feature_importance = None
    if not use_calibration and hasattr(lgb_model, 'feature_importance'):
        feature_names = data.get('_feature_names', [])
        importance = lgb_model.feature_importance(importance_type='gain')
        if len(feature_names) == len(importance):
            feature_importance = dict(zip(feature_names, importance))
            
    extras = {
        'quantile_threshold': data.get('_quantile_threshold', 0.75),
        'long_threshold': data.get('_long_threshold', 0.034),
        'long_threshold_percentile': data.get('_long_threshold_percentile', 75),
        'n_long_lines': len(data.get('_long_lines_filtered', [])),
        'n_short_lines': len(data.get('_short_lines_filtered', [])),
        'class_distribution': {
            'train': dict(zip(*np.unique(y_train, return_counts=True))),
            'test': dict(zip(*np.unique(y_test, return_counts=True)))
        },
        'per_class_metrics': {
            'no_trade': class_report.get('0', {}),
            'long': class_report.get('1', {})
        },
        'feature_importance': feature_importance,
        'best_iteration': lgb_model.best_iteration if hasattr(lgb_model, 'best_iteration') else None,
        'calibration_used': use_calibration
    }
    original_df = data.get('_original_df', None)
    if original_df is not None and 'atr_pct' in original_df.columns:
        long_threshold = data.get('_long_threshold', CONFIG['default_threshold'])
        
        # Allow exit-logic overrides from round params for sweeps
        exit_config = dict(CONFIG)
        for k in [
            'loser_timeout_hours',
            'max_hold_hours',
            'confidence_threshold',
            'position_size',
            'min_stop_loss',
            'max_stop_loss',
            'atr_stop_multiplier',
            'trailing_activation',
            'trailing_distance',
            'default_atr_pct',
            'max_hours_since_line'
        ]:
            if k in round_params:
                exit_config[k] = round_params[k]
        
        _, trading_results = apply_long_only_exit_strategy(
            original_df, y_pred, y_proba, long_threshold, exit_config
        )
        
        metrics.update({
            'trading_return_net_pct': float(trading_results['total_return_net_pct']),
            'trading_win_rate_pct': float(trading_results['trade_win_rate_pct']),
            'trading_trades_count': float(trading_results['trades_count']),
            'trading_max_drawdown_pct': float(trading_results['max_drawdown_pct']),
            'trading_avg_win': float(trading_results['avg_win']),
            'trading_avg_loss': float(trading_results['avg_loss'])
        })
        
        extras['trading_results'] = trading_results
        extras['complete_exit_strategy_applied'] = True
    else:
        raise ValueError("Could not apply complete exit strategy - missing original dataframe or ATR")
    round_results = metrics
    round_results['models'] = [final_model]
    round_results['extras'] = extras
    round_results['_scaler'] = data.get('_scaler')
    round_results['_preds'] = y_pred
    
    return round_results
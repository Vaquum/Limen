#!/usr/bin/env python3
'''
Tradeline Multiclass - UEL Single File Model format
Implements line-based multiclass trading predictions with parameterized quantile thresholds
'''

import numpy as np
import polars as pl
import lightgbm as lgb

from loop.utils.splits import split_sequential, split_data_to_prep_output
from loop.metrics.multiclass_metrics import multiclass_metrics

from loop.sfm.lightgbm.utils.tradeline_multiclass import (
    find_price_lines,
    filter_lines_by_quantile,
    compute_line_features,
    compute_temporal_features,
    compute_price_features,
    create_multiclass_labels,
    apply_class_weights
)

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


def params():
    '''
    Define parameter space for Tradeline Multiclass model optimization.
    
    Returns:
        dict: Dictionary containing parameter ranges for line computation, LightGBM hyperparameters, and trading thresholds
    '''
    p = {
        'quantile_threshold': [0.60, 0.70, 0.75, 0.80, 0.85],
        'min_height_pct': [0.003],
        'max_duration_hours': [48],
        
        'objective': ['multiclass'],
        'num_class': [3],
        'metric': ['multi_logloss'],
        'boosting_type': ['gbdt'],
        'num_leaves': [31, 63],
        'learning_rate': [0.05, 0.1],
        'feature_fraction': [0.9],
        'bagging_fraction': [0.8],
        'bagging_freq': [5],
        'min_child_samples': [20, 40],
        'lambda_l1': [0, 0.1],
        'lambda_l2': [0, 0.1],
        'verbose': [-1],
        'n_estimators': [500],
        
        'lookahead_hours': [48],
        
        'long_threshold_percentile': [65, 75, 85],
        'short_threshold_percentile': [65, 75, 85],
        
        'use_calibration': [True],
        'calibration_method': ['isotonic'],
        'calibration_cv': [3]
    }
    return p


def prep(data, round_params=None):
    '''
    Prepare data for Tradeline Multiclass model.
    
    This function performs the following steps:
        1. Computes price lines on the entire dataset.
        2. Filters lines by quantile threshold.
        3. Engineers features based on lines.
        4. Creates multiclass labels.
        5. Splits data sequentially.
        6. Returns a formatted data dictionary.
    
    Args:
        data (pl.DataFrame): Input data containing price and datetime columns.
        round_params (dict, optional): Dictionary of parameters for line computation and feature engineering.
        
    Returns:
        dict: Dictionary containing prepared training, validation, and test datasets, features, and labels.
    '''
    if not isinstance(data, pl.DataFrame):
        raise ValueError("Data must be a Polars DataFrame")
    
    df = data.clone()
    
    if 'Date' in df.columns:
        df = df.rename({'Date': 'datetime'})
    
    if 'time' in df.columns and 'datetime' not in df.columns:
        df = df.rename({'time': 'datetime'})
    
    if df.schema['datetime'] != pl.Datetime:
        try:
            df = df.with_columns(pl.col('datetime').str.to_datetime())
        except:
            df = df.with_columns(pl.from_epoch(pl.col('datetime'), time_unit='s').alias('datetime'))
    
    if round_params is None:
        round_params = {}
    
    quantile_threshold = round_params.get('quantile_threshold', 0.75)
    min_height_pct = round_params.get('min_height_pct', 0.003)
    max_duration_hours = round_params.get('max_duration_hours', 48)
    lookahead_hours = round_params.get('lookahead_hours', 48)
    long_threshold_percentile = round_params.get('long_threshold_percentile', 75)
    short_threshold_percentile = round_params.get('short_threshold_percentile', 75)
    
    
    long_lines, short_lines = find_price_lines(df, max_duration_hours, min_height_pct)
    
    long_lines_filtered = filter_lines_by_quantile(long_lines, quantile_threshold)
    short_lines_filtered = filter_lines_by_quantile(short_lines, quantile_threshold)
    
    if long_lines:
        long_heights = [abs(line['height_pct']) for line in long_lines]
        long_threshold = np.percentile(long_heights, long_threshold_percentile)
    else:
        long_threshold = 0.034
        
    if short_lines:
        short_heights = [abs(line['height_pct']) for line in short_lines]
        short_threshold = np.percentile(short_heights, short_threshold_percentile)
    else:
        short_threshold = 0.034
    
    
    df = compute_temporal_features(df)
    
    df = compute_price_features(df)
    
    df = compute_line_features(df, long_lines_filtered, short_lines_filtered)
    
    df = create_multiclass_labels(df, long_threshold, short_threshold, lookahead_hours)
    
    df_clean = df.drop_nulls()
    
    if len(df_clean) == 0:
        raise ValueError("No data left after cleaning")
    
    label_counts = df_clean.group_by('label').count().sort('label')
    
    exclude_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'label']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    numeric_features = []
    for col in feature_cols:
        if df_clean.schema[col] in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]:
            numeric_features.append(col)
    
    
    cols = numeric_features + ['label']
    
    split_data = split_sequential(df_clean, ratios=(TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT))
    
    data_dict = split_data_to_prep_output(split_data, cols)
    
    data_dict['x_train'] = data_dict['x_train'].to_numpy()
    data_dict['y_train'] = data_dict['y_train'].to_numpy()
    data_dict['x_val'] = data_dict['x_val'].to_numpy()
    data_dict['y_val'] = data_dict['y_val'].to_numpy()
    data_dict['x_test'] = data_dict['x_test'].to_numpy()
    data_dict['y_test'] = data_dict['y_test'].to_numpy()
    
    data_dict['_feature_names'] = numeric_features
    data_dict['_quantile_threshold'] = quantile_threshold
    data_dict['_long_threshold'] = long_threshold
    data_dict['_short_threshold'] = short_threshold
    data_dict['_long_threshold_percentile'] = long_threshold_percentile
    data_dict['_short_threshold_percentile'] = short_threshold_percentile
    
    data_dict['_long_lines_filtered'] = long_lines_filtered
    data_dict['_short_lines_filtered'] = short_lines_filtered
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_dict['x_train'] = scaler.fit_transform(data_dict['x_train'])
    data_dict['x_val'] = scaler.transform(data_dict['x_val'])
    data_dict['x_test'] = scaler.transform(data_dict['x_test'])
    data_dict['_scaler'] = scaler
    
    
    return data_dict


def model(data, round_params):
    '''
    Train Tradeline Multiclass model with LightGBM and optional calibration.
    
    Returns results compatible with UEL including:
    - Multiclass metrics (precision, recall, AUC, accuracy)
    - Model stored in 'models' key
    - Additional metrics in 'extras' key
    '''
    use_calibration = round_params.get('use_calibration', True)
    calibration_method = round_params.get('calibration_method', 'isotonic')
    calibration_cv = round_params.get('calibration_cv', 3)
    n_estimators = round_params.get('n_estimators', 500)
    
    lgb_params = {
        'objective': round_params.get('objective', 'multiclass'),
        'num_class': round_params.get('num_class', 3),
        'metric': round_params.get('metric', 'multi_logloss'),
        'boosting_type': round_params.get('boosting_type', 'gbdt'),
        'num_leaves': round_params.get('num_leaves', 31),
        'learning_rate': round_params.get('learning_rate', 0.05),
        'feature_fraction': round_params.get('feature_fraction', 0.9),
        'bagging_fraction': round_params.get('bagging_fraction', 0.8),
        'bagging_freq': round_params.get('bagging_freq', 5),
        'min_child_samples': round_params.get('min_child_samples', 20),
        'lambda_l1': round_params.get('lambda_l1', 0),
        'lambda_l2': round_params.get('lambda_l2', 0),
        'verbose': round_params.get('verbose', -1),
        'random_state': 42
    }
    
    X_train = data['x_train']
    y_train = data['y_train']
    X_val = data['x_val']
    y_val = data['y_val']
    X_test = data['x_test']
    y_test = data['y_test']
    
    sample_weights = apply_class_weights(y_train)
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    
    lgb_params = lgb_params.copy()
    lgb_params.update({
        'verbose': -1,
    })
    
    evals_result = {}
    lgb_model = lgb.train(
        params=lgb_params,
        train_set=train_data,
        num_boost_round=n_estimators,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.record_evaluation(evals_result),
            lgb.log_evaluation(period=0)
        ]
    )
    
    final_train_loss = evals_result['train']['multi_logloss'][-1]
    final_val_loss = evals_result['val']['multi_logloss'][-1]
    
    y_pred_proba = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    if use_calibration:
        
        val_pred_proba = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        
        from sklearn.isotonic import IsotonicRegression
        calibrators = []
        
        for i in range(3):
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(val_pred_proba[:, i], y_val == i)
            calibrators.append(iso_reg)
        
        calibrated_proba = np.zeros_like(y_pred_proba)
        for i in range(3):
            calibrated_proba[:, i] = calibrators[i].transform(y_pred_proba[:, i])
        
        calibrated_proba = calibrated_proba / calibrated_proba.sum(axis=1, keepdims=True)
        
        y_proba = calibrated_proba
        y_pred = np.argmax(calibrated_proba, axis=1)
        
        final_model = lgb_model
    else:
        y_proba = y_pred_proba
        final_model = lgb_model
    
    metrics = multiclass_metrics(data, y_pred, y_proba)
    
    metrics['val_loss'] = float(final_val_loss)
    
    from sklearn.metrics import classification_report
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
        'short_threshold': data.get('_short_threshold', 0.034),
        'long_threshold_percentile': data.get('_long_threshold_percentile', 75),
        'short_threshold_percentile': data.get('_short_threshold_percentile', 75),
        'n_long_lines': len(data.get('_long_lines_filtered', [])),
        'n_short_lines': len(data.get('_short_lines_filtered', [])),
        'class_distribution': {
            'train': dict(zip(*np.unique(y_train, return_counts=True))),
            'test': dict(zip(*np.unique(y_test, return_counts=True)))
        },
        'per_class_metrics': {
            'no_trade': class_report.get('0', {}),
            'long': class_report.get('1', {}),
            'short': class_report.get('2', {})
        },
        'feature_importance': feature_importance,
        'best_iteration': lgb_model.best_iteration if hasattr(lgb_model, 'best_iteration') else None,
        'calibration_used': use_calibration
    }
    
    round_results = metrics
    round_results['models'] = [final_model]
    round_results['extras'] = extras
    round_results['_scaler'] = data.get('_scaler')
    round_results['_preds'] = y_pred
    
    return round_results
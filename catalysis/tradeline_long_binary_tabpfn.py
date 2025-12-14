'''
Tradeline Long Binary TabPFN - Long-Only Binary Trading System
Forked from tradeline_long_binary_fixed.py
Uses TabPFN (Tabular Prior-data Fitted Network) instead of LightGBM
Based on Nature paper: "Accurate predictions on small data with a tabular foundation model"
'''
from typing import Dict, List, Any, Optional

import numpy as np
import polars as pl
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from loop.utils.splits import split_sequential, split_data_to_prep_output
from loop.metrics.binary_metrics import binary_metrics
from catalysis.utils.tradeline_long_binary import (
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



EXCLUDE_CATEGORIES = {
    'basic': ['datetime', 'open', 'high', 'low', 'close', 'volume'],
    'targets': ['label'],
    'intermediate': ['hl', 'hpc', 'lpc', 'tr', 'atr']
}

CONFIG = {
    'min_height_pct_default': 0.003,
    'max_duration_hours_default': 48,
    'lookahead_hours_default': 24,
    'long_threshold_percentile_default': 75,
    'quantile_threshold_default': 0.75,
    'default_threshold': 0.034,
    'atr_period': 24,
    'max_positions': 1,
    'initial_capital': 100000.0,
    'density_lookback_hours': 48,
    'big_move_lookback_hours': 168,
    'random_state': 42,
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15
}


def params() -> dict:
    '''
    TabPFN has minimal hyperparameters compared to LightGBM.
    Focus on trading/signal parameters + TabPFN ensemble settings.
    '''
    p = {
        # Signal detection params
        'quantile_threshold': [0.60, 0.70, 0.75, 0.80, 0.85],
        'min_height_pct': [0.001, 0.002, 0.003, 0.004, 0.005],
        'max_duration_hours': [24, 48, 72, 96],
        'lookahead_hours': [24, 48, 72],
        'long_threshold_percentile': [65, 75, 85],

        # Trading params
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

        # TabPFN params (minimal - it works well out-of-box)
        'n_estimators': [4, 8],  # Ensemble size (default 8)
        'softmax_temperature': [0.7, 0.9, 1.0],  # Prediction confidence
    }
    return p


def prep(data: pl.DataFrame, round_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    '''
    Data preparation - same as fixed version but without LightGBM Dataset creation.
    TabPFN works directly with numpy arrays.
    '''
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

    df = compute_line_features(
        df,
        long_lines_filtered,
        short_lines_filtered,
        big_move_lookback_hours=round_params.get('big_move_lookback_hours', CONFIG['big_move_lookback_hours'])
    )

    df = compute_quantile_line_features(
        df,
        long_lines_filtered,
        short_lines_filtered,
        density_lookback_hours=round_params.get('density_lookback_hours', CONFIG['density_lookback_hours'])
    )

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

    train_split = CONFIG.get('train_split', 0.7)
    val_split = CONFIG.get('val_split', 0.15)
    test_split = CONFIG.get('test_split', 0.15)
    split_data = split_sequential(df_clean, ratios=[int(train_split * 100), int(val_split * 100), int(test_split * 100)])

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

    # StandardScaler for TabPFN (helps with numeric stability)
    scaler = StandardScaler()
    data_dict['x_train'] = scaler.fit_transform(data_dict['x_train'])
    data_dict['x_val'] = scaler.transform(data_dict['x_val'])
    data_dict['x_test'] = scaler.transform(data_dict['x_test'])
    data_dict['_scaler'] = scaler

    # FIX: Extract only test split portion instead of full dataset
    # Calculate test split start index (train 70% + val 15% = 85%)
    total_rows = len(df_clean)
    test_start_idx = int(total_rows * (train_split + val_split))
    test_df = df_clean[test_start_idx:]

    test_df_for_trading = test_df.select(['datetime', 'open', 'high', 'low', 'close', 'volume', 'atr_pct', 'label'])
    data_dict['_original_df'] = test_df_for_trading

    # No LightGBM Dataset needed - TabPFN works with raw numpy arrays

    return data_dict


def model(data: Dict[str, Any], round_params: Dict[str, Any]) -> Dict[str, Any]:
    '''
    TabPFN model training and evaluation.
    Much simpler than LightGBM - no early stopping, no hyperparameter tuning needed.
    '''
    # TabPFN params
    n_estimators = round_params.get('n_estimators', 8)
    softmax_temperature = round_params.get('softmax_temperature', 0.9)
    decision_threshold = round_params.get('decision_threshold', 0.45)

    X_train = data['x_train']
    X_val = data['x_val']
    X_test = data['x_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']

    # Combine train and val for TabPFN (it doesn't need validation for early stopping)
    # TabPFN uses in-context learning - more training data is better
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    # Initialize TabPFN V2 (open model, no auth required)
    # V2.5 requires HuggingFace authentication
    tabpfn_model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        n_estimators=n_estimators,
        softmax_temperature=softmax_temperature,
        random_state=CONFIG['random_state'],
        device='cpu',  # Use CPU for stability
        ignore_pretraining_limits=True,  # Allow >1000 samples on CPU
    )

    # Fit - TabPFN's "training" is just storing the context
    # The actual learning happens during the forward pass
    tabpfn_model.fit(X_train_full, y_train_full)

    # Predict
    y_pred_proba = tabpfn_model.predict_proba(X_test)[:, 1]  # Probability of class 1
    y_proba = y_pred_proba
    y_pred = (y_pred_proba > decision_threshold).astype(int)

    # Compute metrics
    metrics = binary_metrics(data, y_pred, y_proba)

    class_report = classification_report(y_test, y_pred, output_dict=True)

    extras = {
        'quantile_threshold': data.get('_quantile_threshold', 0.75),
        'long_threshold': data.get('_long_threshold', 0.034),
        'long_threshold_percentile': data.get('_long_threshold_percentile', 75),
        'n_long_lines': len(data.get('_long_lines_filtered', [])),
        'n_short_lines': len(data.get('_short_lines_filtered', [])),
        'class_distribution': {
            'train': dict(zip(*np.unique(y_train_full, return_counts=True))),
            'test': dict(zip(*np.unique(y_test, return_counts=True)))
        },
        'per_class_metrics': {
            'no_trade': class_report.get('0', {}),
            'long': class_report.get('1', {})
        },
        'model_type': 'TabPFN',
        'n_estimators': n_estimators,
        'softmax_temperature': softmax_temperature,
    }

    # Trading simulation
    original_df = data['_original_df']
    long_threshold = data.get('_long_threshold', CONFIG['default_threshold'])

    exit_config = {
        'confidence_threshold': round_params['confidence_threshold'],
        'position_size': round_params['position_size'],
        'min_stop_loss': round_params['min_stop_loss'],
        'max_stop_loss': round_params['max_stop_loss'],
        'atr_stop_multiplier': round_params['atr_stop_multiplier'],
        'trailing_activation': round_params['trailing_activation'],
        'trailing_distance': round_params['trailing_distance'],
        'loser_timeout_hours': round_params['loser_timeout_hours'],
        'max_hold_hours': round_params['max_hold_hours'],
        'default_atr_pct': round_params['default_atr_pct'],
        'initial_capital': CONFIG['initial_capital']
    }

    _, trading_results = apply_long_only_exit_strategy(
        original_df, y_pred, y_proba, long_threshold, exit_config
    )

    metrics.update({
        'trading_return_net_pct': float(trading_results['total_return_net_pct']),
        'trading_win_rate_pct': float(trading_results['trade_win_rate_pct']),
        'trading_trades_count': float(trading_results['trades_count']),
        'trading_avg_win': float(trading_results['avg_win']),
        'trading_avg_loss': float(trading_results['avg_loss'])
    })

    extras['trading_results'] = trading_results
    extras['complete_exit_strategy_applied'] = True

    round_results = metrics
    round_results['models'] = [tabpfn_model]
    round_results['extras'] = extras
    round_results['_scaler'] = data.get('_scaler')
    round_results['_preds'] = y_pred

    return round_results

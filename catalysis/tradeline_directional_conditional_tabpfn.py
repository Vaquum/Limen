'''
Tradeline Directional Conditional TabPFN - 4-Model Conditional Probability Trading System
Forked from loop/sfm/lightgbm/tradeline_directional_conditional.py
Uses TabPFN instead of LightGBM for all 4 binary classifiers

Model Architecture:
    Trains 4 TabPFN binary classifiers:
    1. LONG: P(price rises >= threshold%)
    2. SHORT: P(price falls >= threshold%)
    3. MOVEMENT: P(|price change| >= threshold%)
    4. BOTH: P(LONG and SHORT both occur)

    Then computes conditional probabilities:
    - P(LONG | movement) = P(LONG) / P(MOVEMENT)
    - P(SHORT | movement) = P(SHORT) / P(MOVEMENT)
    - P(BOTH | movement) = P(BOTH) / P(MOVEMENT)

    And safer directional probabilities:
    - P(LONG only | movement) = P(LONG | movement) - P(BOTH | movement)
    - P(SHORT only | movement) = P(SHORT | movement) - P(BOTH | movement)

Signal Generation:
    Binary signals generated when:
    - P(directional | movement) > conditional_threshold AND
    - P(movement) > movement_threshold
'''
from typing import Dict, Any, Optional

import numpy as np
import polars as pl
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

from loop.utils.splits import split_sequential, split_data_to_prep_output
from loop.metrics.binary_metrics import binary_metrics
from loop.sfm.lightgbm.utils.tradeline_long_binary import (
    find_price_lines,
    filter_lines_by_quantile,
    compute_line_features,
    compute_temporal_features,
    compute_price_features,
    apply_class_weights,
    compute_quantile_line_features,
    calculate_atr
)
from loop.sfm.lightgbm.utils.tradeline_directional_conditional import (
    create_quad_labels
)


EXCLUDE_CATEGORIES = {
    'basic': ['datetime', 'open', 'high', 'low', 'close', 'volume'],
    'targets': ['label_long', 'label_short', 'label_movement', 'label_both'],
    'intermediate': ['hl', 'hpc', 'lpc', 'tr', 'atr']
}

CONFIG = {
    'min_height_pct_default': 0.003,
    'max_duration_hours_default': 48,
    'lookahead_hours_default': 48,
    'quantile_threshold_default': 0.75,
    'threshold_default': 0.01,
    'atr_period': 24,
    'density_lookback_hours': 48,
    'big_move_lookback_hours': 168,
    'random_state': 42,
    'use_calibration': True,
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'conditional_threshold': 0.7,
    'movement_threshold': 0.3,
    'use_safer': False
}


def params() -> dict:
    '''
    TabPFN has minimal hyperparameters.
    Focus on signal detection and conditional probability thresholds.
    '''
    p = {
        # Signal detection params
        'threshold_pct': [0.005, 0.010, 0.015, 0.020],
        'lookahead_hours': [24, 48, 72],
        'quantile_threshold': [0.70, 0.75, 0.80],
        'min_height_pct': [0.002, 0.003, 0.005],
        'max_duration_hours': [24, 48, 72],

        # Conditional probability thresholds
        'conditional_threshold': [0.6, 0.7, 0.8],
        'movement_threshold': [0.2, 0.3, 0.4],
        'use_safer': [False, True],

        # TabPFN params (minimal)
        'n_estimators': [4, 8],
        'softmax_temperature': [0.7, 0.9, 1.0],
    }
    return p


def prep(data: pl.DataFrame, round_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    '''
    Data preparation - creates 4 task datasets for the quad-model approach.
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
    threshold_pct = round_params.get('threshold_pct', CONFIG['threshold_default'])

    long_lines, short_lines = find_price_lines(df, max_duration_hours, min_height_pct)
    long_lines_filtered = filter_lines_by_quantile(long_lines, quantile_threshold)
    short_lines_filtered = filter_lines_by_quantile(short_lines, quantile_threshold)

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

    df = create_quad_labels(df, threshold_pct, lookahead_hours)

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

    train_split = CONFIG.get('train_split', 0.7)
    val_split = CONFIG.get('val_split', 0.15)
    test_split = CONFIG.get('test_split', 0.15)
    split_ratios = [int(train_split * 100), int(val_split * 100), int(test_split * 100)]

    data_dict = {
        '_feature_names': numeric_features,
        '_threshold_pct': threshold_pct,
        '_all_datetimes': all_datetimes
    }

    # Create task datasets for each of the 4 models
    for task in ['long', 'short', 'movement', 'both']:
        cols = ['datetime'] + numeric_features + [f'label_{task}']
        split_data = split_sequential(df_clean, ratios=split_ratios)
        task_dict = split_data_to_prep_output(split_data, cols, all_datetimes)

        task_dict['x_train'] = task_dict['x_train'].to_numpy()
        task_dict['y_train'] = task_dict['y_train'].to_numpy()
        task_dict['x_val'] = task_dict['x_val'].to_numpy()
        task_dict['y_val'] = task_dict['y_val'].to_numpy()
        task_dict['x_test'] = task_dict['x_test'].to_numpy()
        task_dict['y_test'] = task_dict['y_test'].to_numpy()

        data_dict[f'task_{task}'] = task_dict

    data_dict['_alignment'] = data_dict['task_long']['_alignment']

    # Shared scaler across all tasks
    scaler = StandardScaler()
    scaler.fit(data_dict['task_long']['x_train'])

    for task in ['long', 'short', 'movement', 'both']:
        td = data_dict[f'task_{task}']
        td['x_train'] = scaler.transform(td['x_train'])
        td['x_val'] = scaler.transform(td['x_val'])
        td['x_test'] = scaler.transform(td['x_test'])

    data_dict['_scaler'] = scaler

    # Copy long task data to top level for binary_metrics compatibility
    long_task = data_dict['task_long']
    data_dict['x_train'] = long_task['x_train']
    data_dict['y_train'] = long_task['y_train']
    data_dict['x_val'] = long_task['x_val']
    data_dict['y_val'] = long_task['y_val']
    data_dict['x_test'] = long_task['x_test']
    data_dict['y_test'] = long_task['y_test']

    # No LGB Datasets needed - TabPFN uses raw numpy arrays

    return data_dict


def model(data: Dict[str, Any], round_params: Dict[str, Any]) -> Dict[str, Any]:
    '''
    Train 4 TabPFN models and compute conditional probabilities.
    '''
    use_calibration = round_params.get('use_calibration', CONFIG['use_calibration'])
    n_estimators = round_params.get('n_estimators', 8)
    softmax_temperature = round_params.get('softmax_temperature', 0.9)

    models = {}
    calibrators = {}
    test_probabilities = {}

    for task in ['long', 'short', 'movement', 'both']:
        td = data[f'task_{task}']

        # Combine train and val for TabPFN (more context = better)
        X_train_full = np.vstack([td['x_train'], td['x_val']])
        y_train_full = np.concatenate([td['y_train'], td['y_val']])

        # Check for single-class issue
        unique_classes = np.unique(y_train_full)
        if len(unique_classes) < 2:
            # Skip this task - assign default probabilities
            test_probabilities[task] = np.full(len(td['x_test']), 0.5)
            models[task] = None
            continue

        # Initialize TabPFN
        tabpfn_model = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2,
            n_estimators=n_estimators,
            softmax_temperature=softmax_temperature,
            random_state=CONFIG['random_state'],
            device='cpu',
            ignore_pretraining_limits=True,
        )

        # Fit
        tabpfn_model.fit(X_train_full, y_train_full)

        # Predict probabilities
        y_pred_proba_raw = tabpfn_model.predict_proba(td['x_test'])

        # Handle potential single-class prediction output
        if y_pred_proba_raw.shape[1] == 1:
            y_pred_proba = np.full(len(td['x_test']), 0.5)
        else:
            y_pred_proba = y_pred_proba_raw[:, 1]

        # Isotonic calibration
        if use_calibration:
            val_pred_proba_raw = tabpfn_model.predict_proba(td['x_val'])
            if val_pred_proba_raw.shape[1] == 1:
                val_pred_proba = np.full(len(td['x_val']), 0.5)
            else:
                val_pred_proba = val_pred_proba_raw[:, 1]

            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(val_pred_proba, td['y_val'])
            calibrated_proba = iso_reg.transform(y_pred_proba)
            test_probabilities[task] = calibrated_proba
            calibrators[task] = iso_reg
        else:
            test_probabilities[task] = y_pred_proba

        models[task] = tabpfn_model

    # Compute conditional probabilities
    p_long = test_probabilities['long']
    p_short = test_probabilities['short']
    p_movement = test_probabilities['movement']
    p_both = test_probabilities['both']

    p_movement_safe = np.maximum(p_movement, 1e-6)

    p_long_given_movement = p_long / p_movement_safe
    p_short_given_movement = p_short / p_movement_safe
    p_both_given_movement = p_both / p_movement_safe

    p_long_given_movement = np.clip(p_long_given_movement, 0, 1)
    p_short_given_movement = np.clip(p_short_given_movement, 0, 1)
    p_both_given_movement = np.clip(p_both_given_movement, 0, 1)

    p_safe_long_given_movement = p_long_given_movement - p_both_given_movement
    p_safe_short_given_movement = p_short_given_movement - p_both_given_movement

    p_safe_long_given_movement = np.clip(p_safe_long_given_movement, 0, 1)
    p_safe_short_given_movement = np.clip(p_safe_short_given_movement, 0, 1)

    # Generate signals
    conditional_threshold = round_params.get('conditional_threshold', CONFIG['conditional_threshold'])
    movement_threshold = round_params.get('movement_threshold', CONFIG['movement_threshold'])
    use_safer = round_params.get('use_safer', CONFIG['use_safer'])

    if use_safer:
        signal_prob = p_safe_long_given_movement
    else:
        signal_prob = p_long_given_movement

    y_pred = ((signal_prob > conditional_threshold) & (p_movement > movement_threshold)).astype(int)

    metrics = binary_metrics(data, y_pred, signal_prob)

    round_results = metrics
    round_results['models'] = [models.get('long'), models.get('short'), models.get('movement'), models.get('both')]
    round_results['_preds'] = y_pred
    round_results['extras'] = {
        'threshold_pct': data['_threshold_pct'],
        'model_type': 'TabPFN',
        'n_estimators': n_estimators,
        'softmax_temperature': softmax_temperature,
        'probabilities': {
            'long': p_long,
            'short': p_short,
            'movement': p_movement,
            'both': p_both,
            'long_given_movement': p_long_given_movement,
            'short_given_movement': p_short_given_movement,
            'both_given_movement': p_both_given_movement,
            'safe_long_given_movement': p_safe_long_given_movement,
            'safe_short_given_movement': p_safe_short_given_movement
        },
        'signal_config': {
            'use_safer': use_safer,
            'conditional_threshold': conditional_threshold,
            'movement_threshold': movement_threshold,
            'signals_generated': int(np.sum(y_pred)),
            'signal_rate_pct': float(np.sum(y_pred)/len(y_pred)*100) if len(y_pred) > 0 else 0.0
        },
        'calibration_used': use_calibration,
        'calibrators': calibrators if use_calibration else None,
        'class_distributions': {
            'long': dict(zip(*np.unique(data['task_long']['y_test'], return_counts=True))),
            'short': dict(zip(*np.unique(data['task_short']['y_test'], return_counts=True))),
            'movement': dict(zip(*np.unique(data['task_movement']['y_test'], return_counts=True))),
            'both': dict(zip(*np.unique(data['task_both']['y_test'], return_counts=True)))
        }
    }
    round_results['_scaler'] = data.get('_scaler')

    return round_results

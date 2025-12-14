"""
Multi-Threshold Resistance Finder

Fork of tradeline_long_binary that trains multiple models for different breakout thresholds.
Analyzes probability gaps between thresholds to identify ML-predicted resistance/support levels.

Concept:
- Train binary classifiers for thresholds: [0.5%, 1.0%, 1.5%, 2.0%, 2.5%, 3.0%, 3.5%, 4.0%, 4.5%, 5.0%]
- For each timestamp, get P(breakout > threshold) for all thresholds
- Calculate probability deltas: ΔP = P(>T1) - P(>T2)
- Large ΔP indicates resistance between T1 and T2
- For shorts, this same analysis finds support levels
"""

from typing import Dict, List, Any, Optional
import numpy as np
import polars as pl
import lightgbm as lgb
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
    create_binary_labels,
    apply_class_weights,
    compute_quantile_line_features,
    calculate_atr
)

# Test multiple breakout thresholds (in decimal form, e.g., 0.01 = 1%)
THRESHOLDS = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050]

EXCLUDE_CATEGORIES = {
    'basic': ['datetime', 'open', 'high', 'low', 'close', 'volume'],
    'targets': ['label'],
    'intermediate': ['hl', 'hpc', 'lpc', 'tr', 'atr']
}

CONFIG = {
    'min_height_pct_default': 0.003,
    'max_duration_hours_default': 48,
    'lookahead_hours_default': 48,
    'quantile_threshold_default': 0.75,
    'atr_period': 24,
    'density_lookback_hours': 48,
    'big_move_lookback_hours': 168,
    'early_stopping_rounds': 50,
    'log_evaluation_period': 0,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'random_state': 42,
    'use_calibration': True,
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15
}


def params() -> dict:
    """Simplified params for multi-threshold training"""
    p = {
        'quantile_threshold': [0.75],
        'min_height_pct': [0.003],
        'max_duration_hours': [48],
        'lookahead_hours': [48],
        'num_leaves': [31],
        'learning_rate': [0.05],
        'feature_fraction': [0.9],
        'bagging_fraction': [0.8],
        'bagging_freq': [5],
        'min_child_samples': [20],
        'lambda_l1': [0],
        'lambda_l2': [0],
        'n_estimators': [300]
    }
    return p


def prep(data: pl.DataFrame, round_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Prepare data with features but create labels for ALL thresholds.
    Returns a data dict with separate labels for each threshold.
    """

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

    # Find price lines
    long_lines, short_lines = find_price_lines(df, max_duration_hours, min_height_pct)
    long_lines_filtered = filter_lines_by_quantile(long_lines, quantile_threshold)
    short_lines_filtered = filter_lines_by_quantile(short_lines, quantile_threshold)

    # Compute features
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

    # Create labels for ALL thresholds
    df_with_labels = df.clone()

    for threshold in THRESHOLDS:
        df_with_labels = create_binary_labels(df_with_labels, threshold, lookahead_hours)
        # Rename the label column to include threshold (use 10x to avoid duplicates)
        threshold_key = int(threshold * 1000)  # e.g., 0.005 -> 5, 0.010 -> 10
        df_with_labels = df_with_labels.rename({'label': f'label_{threshold_key}'})

    df_clean = df_with_labels.drop_nulls()

    if len(df_clean) == 0:
        raise ValueError("No data left after cleaning")

    # Identify feature columns
    exclude_cols = []
    for category in EXCLUDE_CATEGORIES.values():
        exclude_cols.extend(category)

    # Also exclude all the label columns from features
    label_cols = [f'label_{int(t*1000)}' for t in THRESHOLDS]
    exclude_cols.extend(label_cols)

    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

    numeric_features = []
    for col in feature_cols:
        if df_clean.schema[col] in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]:
            numeric_features.append(col)

    # Split data
    train_split = CONFIG.get('train_split', 0.7)
    val_split = CONFIG.get('val_split', 0.15)
    test_split = CONFIG.get('test_split', 0.15)
    split_ratios = [int(train_split * 100), int(val_split * 100), int(test_split * 100)]

    # Create separate data dicts for each threshold
    data_dict = {
        '_feature_names': numeric_features,
        '_thresholds': THRESHOLDS,
        '_all_datetimes': all_datetimes,
        'threshold_data': {}
    }

    for threshold in THRESHOLDS:
        threshold_key = int(threshold * 1000)
        label_col = f'label_{threshold_key}'

        cols = ['datetime'] + numeric_features + [label_col]
        split_data = split_sequential(df_clean, ratios=split_ratios)

        threshold_dict = split_data_to_prep_output(split_data, cols, all_datetimes)

        # Convert to numpy
        threshold_dict['x_train'] = threshold_dict['x_train'].to_numpy()
        threshold_dict['y_train'] = threshold_dict['y_train'].to_numpy()
        threshold_dict['x_val'] = threshold_dict['x_val'].to_numpy()
        threshold_dict['y_val'] = threshold_dict['y_val'].to_numpy()
        threshold_dict['x_test'] = threshold_dict['x_test'].to_numpy()
        threshold_dict['y_test'] = threshold_dict['y_test'].to_numpy()

        data_dict['threshold_data'][threshold] = threshold_dict

    # Use first threshold's alignment for top-level alignment (required by UEL)
    first_threshold = THRESHOLDS[0]
    data_dict['_alignment'] = data_dict['threshold_data'][first_threshold]['_alignment']

    # Scale features (use first threshold's data for fitting scaler)
    scaler = StandardScaler()
    scaler.fit(data_dict['threshold_data'][first_threshold]['x_train'])

    # Apply scaler to all thresholds
    for threshold in THRESHOLDS:
        td = data_dict['threshold_data'][threshold]
        td['x_train'] = scaler.transform(td['x_train'])
        td['x_val'] = scaler.transform(td['x_val'])
        td['x_test'] = scaler.transform(td['x_test'])

    data_dict['_scaler'] = scaler

    # Add top-level keys for UEL compatibility (use first threshold's data)
    first_td = data_dict['threshold_data'][first_threshold]
    data_dict['x_train'] = first_td['x_train']
    data_dict['y_train'] = first_td['y_train']
    data_dict['x_val'] = first_td['x_val']
    data_dict['y_val'] = first_td['y_val']
    data_dict['x_test'] = first_td['x_test']
    data_dict['y_test'] = first_td['y_test']

    # Create LightGBM datasets for all thresholds
    for threshold in THRESHOLDS:
        td = data_dict['threshold_data'][threshold]
        sample_weights = apply_class_weights(td['y_train'])
        td['dtrain'] = lgb.Dataset(td['x_train'], label=td['y_train'], weight=sample_weights)
        td['dval'] = lgb.Dataset(td['x_val'], label=td['y_val'], reference=td['dtrain'])

    return data_dict


def model(data: Dict[str, Any], round_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train separate models for each threshold.
    Return probabilities for all thresholds on test data.
    """

    use_calibration = round_params.get('use_calibration', CONFIG['use_calibration'])
    n_estimators = round_params.get('n_estimators', 300)

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

    models = {}
    calibrators = {}
    test_probabilities = {}

    print(f'\nTraining {len(THRESHOLDS)} models for thresholds: {[f"{t*100:.1f}%" for t in THRESHOLDS]}')

    for threshold in THRESHOLDS:
        threshold_pct = int(threshold * 100)
        print(f'  Training model for {threshold*100:.1f}% threshold...')

        td = data['threshold_data'][threshold]

        # Train model
        evals_result = {}
        lgb_model = lgb.train(
            params=lgb_params,
            train_set=td['dtrain'],
            num_boost_round=n_estimators,
            valid_sets=[td['dtrain'], td['dval']],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=CONFIG['early_stopping_rounds'], verbose=False),
                lgb.record_evaluation(evals_result),
                lgb.log_evaluation(period=CONFIG['log_evaluation_period'])
            ]
        )

        # Get predictions
        y_pred_proba = lgb_model.predict(td['x_test'], num_iteration=lgb_model.best_iteration)

        # Calibrate if needed
        if use_calibration:
            val_pred_proba = lgb_model.predict(td['x_val'], num_iteration=lgb_model.best_iteration)
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(val_pred_proba, td['y_val'])
            calibrated_proba = iso_reg.transform(y_pred_proba)
            test_probabilities[threshold] = calibrated_proba
            calibrators[threshold] = iso_reg
        else:
            test_probabilities[threshold] = y_pred_proba

        models[threshold] = lgb_model

    print(f'\nAll {len(THRESHOLDS)} models trained successfully')

    # Analyze probability gradients to find resistance levels
    resistance_analysis = analyze_resistance_levels(test_probabilities, THRESHOLDS)

    # Compute basic metrics for first threshold (for compatibility)
    first_threshold = THRESHOLDS[0]
    td = data['threshold_data'][first_threshold]
    y_pred = (test_probabilities[first_threshold] > 0.5).astype(int)
    metrics = binary_metrics(td, y_pred, test_probabilities[first_threshold])

    # Package results
    round_results = metrics
    round_results['models'] = models
    round_results['_preds'] = y_pred  # Required by UEL
    round_results['extras'] = {
        'thresholds': THRESHOLDS,
        'test_probabilities': test_probabilities,
        'resistance_analysis': resistance_analysis,
        'calibration_used': use_calibration,
        'calibrators': calibrators if use_calibration else None
    }
    round_results['_scaler'] = data.get('_scaler')

    return round_results


def analyze_resistance_levels(probabilities: Dict[float, np.ndarray], thresholds: List[float]) -> Dict[str, Any]:
    """
    Analyze probability gradients to identify resistance levels.

    For each timestamp, calculate ΔP between consecutive thresholds.
    Large ΔP indicates resistance.
    """

    n_samples = len(probabilities[thresholds[0]])

    # Calculate probability deltas for each sample
    sample_deltas = []

    for i in range(n_samples):
        deltas = []
        probs = []

        for threshold in thresholds:
            probs.append(probabilities[threshold][i])

        # Calculate deltas between consecutive thresholds
        for j in range(len(thresholds) - 1):
            delta = probs[j] - probs[j + 1]
            deltas.append({
                'threshold_low': thresholds[j],
                'threshold_high': thresholds[j + 1],
                'prob_low': probs[j],
                'prob_high': probs[j + 1],
                'delta': delta
            })

        sample_deltas.append(deltas)

    # Find average deltas across all samples
    avg_deltas = []
    for j in range(len(thresholds) - 1):
        delta_values = [sample_deltas[i][j]['delta'] for i in range(n_samples)]
        avg_deltas.append({
            'threshold_low': thresholds[j],
            'threshold_high': thresholds[j + 1],
            'avg_delta': np.mean(delta_values),
            'max_delta': np.max(delta_values),
            'std_delta': np.std(delta_values)
        })

    # Identify resistance zones (where avg_delta is large)
    resistance_threshold = np.mean([d['avg_delta'] for d in avg_deltas]) + np.std([d['avg_delta'] for d in avg_deltas])

    resistance_zones = [d for d in avg_deltas if d['avg_delta'] > resistance_threshold]

    analysis = {
        'sample_deltas': sample_deltas,
        'avg_deltas': avg_deltas,
        'resistance_zones': resistance_zones,
        'resistance_threshold': resistance_threshold
    }

    return analysis

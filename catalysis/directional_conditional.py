"""
Directional Conditional Probability Model

Trains 4 binary classifiers:
1. LONG: P(price rises ≥threshold%)
2. SHORT: P(price falls ≥threshold%)
3. MOVEMENT: P(|price change| ≥threshold%)
4. BOTH: P(LONG and SHORT both occur)

Then computes conditional probabilities:
- P(LONG | movement) = P(LONG) / P(MOVEMENT)
- P(SHORT | movement) = P(SHORT) / P(MOVEMENT)
- P(BOTH | movement) = P(BOTH) / P(MOVEMENT)

And safer directional probabilities:
- P(LONG and not SHORT | movement) = P(LONG | movement) - P(BOTH | movement)
- P(SHORT and not LONG | movement) = P(SHORT | movement) - P(BOTH | movement)
"""

from typing import Dict, Any, Optional
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
    apply_class_weights,
    compute_quantile_line_features,
    calculate_atr
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
    'threshold_default': 0.01,  # 1% default breakout threshold
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
    'test_split': 0.15,
    'conditional_threshold': 0.7,  # Threshold for P(LONG | movement)
    'movement_threshold': 0.3,  # Threshold for P(movement)
    'use_safer': False  # Use P(LONG only | movement) instead of P(LONG | movement)
}


def params() -> dict:
    """Parameter grid for UEL - sweep over MODEL LOGIC parameters"""
    p = {
        # MODEL LOGIC PARAMETERS - control the trading strategy behavior
        'threshold_pct': [0.005, 0.010, 0.015, 0.020],  # Breakout threshold: 0.5%, 1%, 1.5%, 2%
        'lookahead_hours': [24, 48, 72],  # How far ahead to look: 1 day, 2 days, 3 days
        'quantile_threshold': [0.70, 0.75, 0.80],  # Quantile for filtering price lines
        'min_height_pct': [0.002, 0.003, 0.005],  # Minimum price line height
        'max_duration_hours': [24, 48, 72],  # Max duration for price lines

        # CONDITIONAL PROBABILITY THRESHOLDS - signal generation logic
        'conditional_threshold': [0.6, 0.7, 0.8],  # Threshold for P(LONG | movement)
        'movement_threshold': [0.2, 0.3, 0.4],  # Threshold for P(movement)
        'use_safer': [False, True],  # Test both regular and safer signals

        # LightGBM model parameters - kept simpler to focus on logic params
        'n_estimators': [200, 300],  # Number of boosting rounds
        'num_leaves': [31, 63],  # Model complexity
        'learning_rate': [0.05, 0.1],  # Step size
        'max_depth': [-1],  # Tree depth (-1 = no limit)
        'min_child_samples': [20],  # Min samples per leaf
        'feature_fraction': [0.9],  # Column sampling
        'bagging_fraction': [0.8],  # Row sampling
        'bagging_freq': [5],  # Frequency of bagging
        'lambda_l1': [0],  # L1 regularization
        'lambda_l2': [0],  # L2 regularization
    }
    return p


def create_quad_labels(df: pl.DataFrame, threshold: float, lookahead_hours: int) -> pl.DataFrame:
    """
    Create 4 binary labels:
    - label_long: 1 if max future return ≥ threshold
    - label_short: 1 if min future return ≤ -threshold
    - label_movement: 1 if max |future return| ≥ threshold
    - label_both: 1 if BOTH long AND short occur
    """
    closes = df['close'].to_numpy()
    n = len(closes)

    labels_long = np.zeros(n, dtype=np.int32)
    labels_short = np.zeros(n, dtype=np.int32)
    labels_movement = np.zeros(n, dtype=np.int32)
    labels_both = np.zeros(n, dtype=np.int32)

    for i in range(n - lookahead_hours):
        current_price = closes[i]
        future_prices = closes[i+1:i+1+lookahead_hours]

        if len(future_prices) == 0:
            continue

        # Max future return (for LONG)
        max_return = (np.max(future_prices) - current_price) / current_price

        # Min future return (for SHORT)
        min_return = (np.min(future_prices) - current_price) / current_price

        # Max absolute return (for MOVEMENT)
        max_abs_return = max(abs(max_return), abs(min_return))

        # Create labels
        labels_long[i] = 1 if max_return >= threshold else 0
        labels_short[i] = 1 if min_return <= -threshold else 0
        labels_movement[i] = 1 if max_abs_return >= threshold else 0
        labels_both[i] = 1 if (labels_long[i] == 1 and labels_short[i] == 1) else 0

    df = df.with_columns([
        pl.Series('label_long', labels_long),
        pl.Series('label_short', labels_short),
        pl.Series('label_movement', labels_movement),
        pl.Series('label_both', labels_both)
    ])

    return df


def prep(data: pl.DataFrame, round_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Prepare data with features and 4 sets of labels"""

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

    # Find price lines (same features as tradeline_long_binary)
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

    # Create 4 sets of labels
    df = create_quad_labels(df, threshold_pct, lookahead_hours)

    df_clean = df.drop_nulls()

    if len(df_clean) == 0:
        raise ValueError("No data left after cleaning")

    # Identify feature columns
    exclude_cols = []
    for category in EXCLUDE_CATEGORIES.values():
        exclude_cols.extend(category)

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

    # Prepare separate datasets for each task
    data_dict = {
        '_feature_names': numeric_features,
        '_threshold_pct': threshold_pct,
        '_all_datetimes': all_datetimes
    }

    # For each task (LONG, SHORT, MOVEMENT, BOTH), create train/val/test splits
    for task in ['long', 'short', 'movement', 'both']:
        cols = ['datetime'] + numeric_features + [f'label_{task}']
        split_data = split_sequential(df_clean, ratios=split_ratios)
        task_dict = split_data_to_prep_output(split_data, cols, all_datetimes)

        # Convert to numpy
        task_dict['x_train'] = task_dict['x_train'].to_numpy()
        task_dict['y_train'] = task_dict['y_train'].to_numpy()
        task_dict['x_val'] = task_dict['x_val'].to_numpy()
        task_dict['y_val'] = task_dict['y_val'].to_numpy()
        task_dict['x_test'] = task_dict['x_test'].to_numpy()
        task_dict['y_test'] = task_dict['y_test'].to_numpy()

        data_dict[f'task_{task}'] = task_dict

    # Use first task's alignment for UEL compatibility
    data_dict['_alignment'] = data_dict['task_long']['_alignment']

    # Scale features (fit on first task, apply to all)
    scaler = StandardScaler()
    scaler.fit(data_dict['task_long']['x_train'])

    for task in ['long', 'short', 'movement', 'both']:
        td = data_dict[f'task_{task}']
        td['x_train'] = scaler.transform(td['x_train'])
        td['x_val'] = scaler.transform(td['x_val'])
        td['x_test'] = scaler.transform(td['x_test'])

    data_dict['_scaler'] = scaler

    # Add top-level keys for UEL compatibility (use LONG task)
    long_task = data_dict['task_long']
    data_dict['x_train'] = long_task['x_train']
    data_dict['y_train'] = long_task['y_train']
    data_dict['x_val'] = long_task['x_val']
    data_dict['y_val'] = long_task['y_val']
    data_dict['x_test'] = long_task['x_test']
    data_dict['y_test'] = long_task['y_test']

    # Create LightGBM datasets for all tasks
    for task in ['long', 'short', 'movement', 'both']:
        td = data_dict[f'task_{task}']
        sample_weights = apply_class_weights(td['y_train'])
        td['dtrain'] = lgb.Dataset(td['x_train'], label=td['y_train'], weight=sample_weights)
        td['dval'] = lgb.Dataset(td['x_val'], label=td['y_val'], reference=td['dtrain'])

    return data_dict


def model(data: Dict[str, Any], round_params: Dict[str, Any]) -> Dict[str, Any]:
    """Train 4 models and compute conditional probabilities"""

    use_calibration = round_params.get('use_calibration', CONFIG['use_calibration'])
    n_estimators = round_params.get('n_estimators', 300)

    lgb_params = {
        'objective': CONFIG['objective'],
        'metric': CONFIG['metric'],
        'boosting_type': CONFIG['boosting_type'],
        'num_leaves': round_params.get('num_leaves', 31),
        'max_depth': round_params.get('max_depth', -1),
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

    print('\nTraining 4 directional models...')

    # Train models for each task
    for task in ['long', 'short', 'movement', 'both']:
        print(f'  Training {task.upper()} model...')

        td = data[f'task_{task}']

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
            test_probabilities[task] = calibrated_proba
            calibrators[task] = iso_reg
        else:
            test_probabilities[task] = y_pred_proba

        models[task] = lgb_model

    print('All 4 models trained successfully')

    # Compute conditional probabilities
    print('\nComputing conditional probabilities...')

    p_long = test_probabilities['long']
    p_short = test_probabilities['short']
    p_movement = test_probabilities['movement']
    p_both = test_probabilities['both']

    # Avoid division by zero
    p_movement_safe = np.maximum(p_movement, 1e-6)

    # P(LONG | movement) = P(LONG) / P(MOVEMENT)
    p_long_given_movement = p_long / p_movement_safe

    # P(SHORT | movement) = P(SHORT) / P(MOVEMENT)
    p_short_given_movement = p_short / p_movement_safe

    # P(BOTH | movement) = P(BOTH) / P(MOVEMENT)
    p_both_given_movement = p_both / p_movement_safe

    # Clip to [0, 1]
    p_long_given_movement = np.clip(p_long_given_movement, 0, 1)
    p_short_given_movement = np.clip(p_short_given_movement, 0, 1)
    p_both_given_movement = np.clip(p_both_given_movement, 0, 1)

    # Compute SAFER directional probabilities
    # P(LONG and not SHORT | movement) = P(LONG | movement) - P(BOTH | movement)
    p_safe_long_given_movement = p_long_given_movement - p_both_given_movement

    # P(SHORT and not LONG | movement) = P(SHORT | movement) - P(BOTH | movement)
    p_safe_short_given_movement = p_short_given_movement - p_both_given_movement

    # Clip to [0, 1]
    p_safe_long_given_movement = np.clip(p_safe_long_given_movement, 0, 1)
    p_safe_short_given_movement = np.clip(p_safe_short_given_movement, 0, 1)

    # Get thresholds from round_params
    conditional_threshold = round_params.get('conditional_threshold', CONFIG['conditional_threshold'])
    movement_threshold = round_params.get('movement_threshold', CONFIG['movement_threshold'])
    use_safer = round_params.get('use_safer', CONFIG['use_safer'])

    # Choose which probability to use for predictions
    if use_safer:
        signal_prob = p_safe_long_given_movement
        signal_name = 'P(LONG only | movement)'
    else:
        signal_prob = p_long_given_movement
        signal_name = 'P(LONG | movement)'

    # Generate binary predictions based on conditional probability and movement filter
    y_pred = ((signal_prob > conditional_threshold) & (p_movement > movement_threshold)).astype(int)

    # Compute metrics for LONG model (for UEL compatibility)
    long_task = data['task_long']
    metrics = binary_metrics(data, y_pred, signal_prob)

    # Print signal configuration
    print(f'\nSignal Configuration:')
    print(f'  Using: {signal_name}')
    print(f'  Conditional threshold: {conditional_threshold}')
    print(f'  Movement threshold: {movement_threshold}')
    print(f'  Signals generated: {np.sum(y_pred)} ({np.sum(y_pred)/len(y_pred)*100:.1f}%)')

    # Package results
    round_results = metrics
    round_results['models'] = [models['long'], models['short'], models['movement'], models['both']]  # All 4 models
    round_results['_preds'] = y_pred  # Required by UEL
    round_results['extras'] = {
        'threshold_pct': data['_threshold_pct'],
        'models': models,
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
        'calibration_used': use_calibration,
        'calibrators': calibrators if use_calibration else None,
        'class_distributions': {
            'long': dict(zip(*np.unique(long_task['y_test'], return_counts=True))),
            'short': dict(zip(*np.unique(data['task_short']['y_test'], return_counts=True))),
            'movement': dict(zip(*np.unique(data['task_movement']['y_test'], return_counts=True))),
            'both': dict(zip(*np.unique(data['task_both']['y_test'], return_counts=True)))
        }
    }
    round_results['_scaler'] = data.get('_scaler')

    # Print summary statistics
    print(f'\nConditional Probability Summary:')
    print(f'  Mean P(LONG | movement):         {np.mean(p_long_given_movement):.3f}')
    print(f'  Mean P(SHORT | movement):        {np.mean(p_short_given_movement):.3f}')
    print(f'  Mean P(BOTH | movement):         {np.mean(p_both_given_movement):.3f}')
    print(f'  Mean P(MOVEMENT):                {np.mean(p_movement):.3f}')
    print(f'\nSafer Directional Probabilities:')
    print(f'  Mean P(LONG only | movement):    {np.mean(p_safe_long_given_movement):.3f}')
    print(f'  Mean P(SHORT only | movement):   {np.mean(p_safe_short_given_movement):.3f}')

    return round_results

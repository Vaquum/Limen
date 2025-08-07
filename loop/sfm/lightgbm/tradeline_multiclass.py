#!/usr/bin/env python3
"""
Tradeline Multiclass - UEL Single File Model format
Implements line-based multiclass trading predictions with parameterized quantile thresholds
"""

import numpy as np
import polars as pl
import lightgbm as lgb
import logging
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

# Import Loop utilities
from loop.utils.splits import split_sequential, split_data_to_prep_output
from loop.metrics.multiclass_metrics import multiclass_metrics

# Import feature engineering functions
from loop.sfm.lightgbm.utils.tradeline_multiclass import (
    find_price_lines,
    filter_lines_by_quantile,
    compute_line_features,
    compute_temporal_features,
    compute_price_features,
    create_multiclass_labels,
    create_lgb_wrapper,
    apply_class_weights
)

# Split ratios for train/val/test
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


def params():
    """
    Parameter space for Tradeline Multiclass model.
    
    Returns parameter ranges for:
    - Line computation parameters
    - LightGBM hyperparameters
    - Trading thresholds
    """
    p = {
        # Line computation parameters
        'quantile_threshold': [0.60, 0.70, 0.75, 0.80, 0.85],
        'min_height_pct': [0.003],
        'max_duration_hours': [48],
        
        # LightGBM parameters
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
        
        # Trading parameters
        'lookahead_hours': [48],
        
        # Threshold percentiles (independent from quantile_threshold)
        'long_threshold_percentile': [65, 75, 85],
        'short_threshold_percentile': [65, 75, 85],
        
        # Calibration parameters
        'use_calibration': [True],  # Enabled by default
        'calibration_method': ['isotonic'],
        'calibration_cv': [3]
    }
    return p


def prep(data, round_params=None):
    """
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
    """
    # Ensure we're working with Polars
    if not isinstance(data, pl.DataFrame):
        raise ValueError("Data must be a Polars DataFrame")
    
    df = data.clone()
    
    # Ensure proper column names and types
    if 'Date' in df.columns:
        df = df.rename({'Date': 'datetime'})
    
    if 'time' in df.columns and 'datetime' not in df.columns:
        df = df.rename({'time': 'datetime'})
    
    # Ensure datetime is proper type
    if df.schema['datetime'] != pl.Datetime:
        # Try different parsing approaches
        try:
            df = df.with_columns(pl.col('datetime').str.to_datetime())
        except:
            # If it's already a timestamp, convert it
            df = df.with_columns(pl.from_epoch(pl.col('datetime'), time_unit='s').alias('datetime'))
    
    # Extract parameters
    if round_params is None:
        round_params = {}
    
    quantile_threshold = round_params.get('quantile_threshold', 0.75)
    min_height_pct = round_params.get('min_height_pct', 0.003)
    max_duration_hours = round_params.get('max_duration_hours', 48)
    lookahead_hours = round_params.get('lookahead_hours', 48)
    long_threshold_percentile = round_params.get('long_threshold_percentile', 75)
    short_threshold_percentile = round_params.get('short_threshold_percentile', 75)
    
    logging.debug(f"Computing lines with quantile threshold: {quantile_threshold}")
    
    # Step 1: Compute all price lines
    long_lines, short_lines = find_price_lines(df, max_duration_hours, min_height_pct)
    logging.debug(f"Found {len(long_lines)} long lines and {len(short_lines)} short lines")
    
    # Step 2: Filter lines by quantile
    long_lines_filtered = filter_lines_by_quantile(long_lines, quantile_threshold)
    short_lines_filtered = filter_lines_by_quantile(short_lines, quantile_threshold)
    logging.debug(f"Filtered to {len(long_lines_filtered)} long and {len(short_lines_filtered)} short lines (Q{int(quantile_threshold*100)})")
    
    # Step 3: Calibrate profit thresholds using specified percentiles
    # Calculate thresholds from ALL lines using the specified percentiles
    if long_lines:
        long_heights = [abs(line['height_pct']) for line in long_lines]
        long_threshold = np.percentile(long_heights, long_threshold_percentile)
        logging.debug(f"Long threshold: {long_threshold:.3%} (P{int(long_threshold_percentile)} of all long lines)")
    else:
        long_threshold = 0.034  # Default if no lines
        
    if short_lines:
        short_heights = [abs(line['height_pct']) for line in short_lines]
        short_threshold = np.percentile(short_heights, short_threshold_percentile)
        logging.debug(f"Short threshold: {short_threshold:.3%} (P{int(short_threshold_percentile)} of all short lines)")
    else:
        short_threshold = 0.034  # Default if no lines
    
    logging.debug(f"Calibrated thresholds - Long: {long_threshold:.3%} (P{int(long_threshold_percentile)}), Short: {short_threshold:.3%} (P{int(short_threshold_percentile)})")
    
    # Step 3: Compute features
    # Temporal features
    logging.debug("Computing temporal features...")
    df = compute_temporal_features(df)
    
    # Price features
    logging.debug("Computing price features...")
    df = compute_price_features(df)
    
    # Line-based features
    logging.debug(f"Computing line-based features from {len(long_lines_filtered)} long and {len(short_lines_filtered)} short lines...")
    df = compute_line_features(df, long_lines_filtered, short_lines_filtered)
    
    # Step 4: Create labels with calibrated thresholds
    logging.debug(f"Creating multiclass labels with lookahead={lookahead_hours}h...")
    df = create_multiclass_labels(df, long_threshold, short_threshold, lookahead_hours)
    
    # Clean data (remove NaN rows from feature computation)
    logging.debug(f"Cleaning data - original size: {len(df)} rows")
    df_clean = df.drop_nulls()
    logging.debug(f"After cleaning: {len(df_clean)} rows")
    
    if len(df_clean) == 0:
        raise ValueError("No data left after cleaning")
    
    # Log label distribution
    label_counts = df_clean.group_by('label').count().sort('label')
    logging.debug("Label distribution:")
    for row in label_counts.iter_rows():
        label, count = row
        logging.debug(f"  Class {label}: {count} ({count/len(df_clean)*100:.1f}%)")
    
    # Select features (exclude non-feature columns)
    exclude_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'label']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    # Ensure all features are numeric
    numeric_features = []
    for col in feature_cols:
        if df_clean.schema[col] in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]:
            numeric_features.append(col)
    
    logging.debug(f"Using {len(numeric_features)} features")
    
    # Create column list with label as LAST column (required by split_data_to_prep_output)
    cols = numeric_features + ['label']
    
    # Step 5: Split data sequentially
    logging.debug(f"Splitting data with ratios {TRAIN_SPLIT}:{VAL_SPLIT}:{TEST_SPLIT}")
    split_data = split_sequential(df_clean, ratios=(TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT))
    
    # Use split_data_to_prep_output to format properly
    data_dict = split_data_to_prep_output(split_data, cols)
    
    # Convert to numpy arrays for LightGBM
    data_dict['x_train'] = data_dict['x_train'].to_numpy()
    data_dict['y_train'] = data_dict['y_train'].to_numpy()
    data_dict['x_val'] = data_dict['x_val'].to_numpy()
    data_dict['y_val'] = data_dict['y_val'].to_numpy()
    data_dict['x_test'] = data_dict['x_test'].to_numpy()
    data_dict['y_test'] = data_dict['y_test'].to_numpy()
    
    # Add metadata for model function
    data_dict['_feature_names'] = numeric_features
    data_dict['_quantile_threshold'] = quantile_threshold
    data_dict['_long_threshold'] = long_threshold
    data_dict['_short_threshold'] = short_threshold
    data_dict['_long_threshold_percentile'] = long_threshold_percentile
    data_dict['_short_threshold_percentile'] = short_threshold_percentile
    
    # Store filtered lines for analysis
    data_dict['_long_lines_filtered'] = long_lines_filtered
    data_dict['_short_lines_filtered'] = short_lines_filtered
    
    # Create and store scalers (data is already numpy arrays)
    from sklearn.preprocessing import StandardScaler
    logging.debug("Scaling features...")
    scaler = StandardScaler()
    data_dict['x_train'] = scaler.fit_transform(data_dict['x_train'])
    data_dict['x_val'] = scaler.transform(data_dict['x_val'])
    data_dict['x_test'] = scaler.transform(data_dict['x_test'])
    data_dict['_scaler'] = scaler
    
    logging.debug(f"Data preparation complete - Train: {len(data_dict['x_train'])}, Val: {len(data_dict['x_val'])}, Test: {len(data_dict['x_test'])}")
    
    return data_dict


def model(data, round_params):
    """
    Train Tradeline Multiclass model with LightGBM and optional calibration.
    
    Returns results compatible with UEL including:
    - Multiclass metrics (precision, recall, AUC, accuracy)
    - Model stored in 'models' key
    - Additional metrics in 'extras' key
    """
    # Extract parameters
    use_calibration = round_params.get('use_calibration', True)
    calibration_method = round_params.get('calibration_method', 'isotonic')
    calibration_cv = round_params.get('calibration_cv', 3)
    n_estimators = round_params.get('n_estimators', 500)
    
    # Extract LightGBM parameters
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
    
    # Get features and labels
    X_train = data['x_train']
    y_train = data['y_train']
    X_val = data['x_val']
    y_val = data['y_val']
    X_test = data['x_test']
    y_test = data['y_test']
    
    # Apply class weights to handle imbalance
    sample_weights = apply_class_weights(y_train)
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train LightGBM model
    logging.debug("Training LightGBM model...")
    logging.debug(f"Training samples: {len(y_train)}, Validation samples: {len(y_val)}")
    logging.debug(f"Class distribution - Train: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    logging.debug(f"Class distribution - Val: {dict(zip(*np.unique(y_val, return_counts=True)))}")
    evals_result = {}
    lgb_model = lgb.train(
        params=lgb_params,
        train_set=train_data,
        num_boost_round=n_estimators,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.record_evaluation(evals_result),
            lgb.log_evaluation(period=50)  # Print progress every 50 rounds
        ]
    )
    
    # Log training results
    final_train_loss = evals_result['train']['multi_logloss'][-1]
    final_val_loss = evals_result['val']['multi_logloss'][-1]
    logging.debug(f"Final train loss: {final_train_loss:.4f}, val loss: {final_val_loss:.4f}")
    
    # Get raw predictions first
    y_pred_proba = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Apply calibration if requested
    if use_calibration:
        logging.debug(f"Applying {calibration_method} calibration...")
        
        # Get validation predictions for calibration
        val_pred_proba = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        
        # Simple isotonic calibration for each class
        from sklearn.isotonic import IsotonicRegression
        calibrators = []
        
        for i in range(3):  # 3 classes
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            iso_reg.fit(val_pred_proba[:, i], y_val == i)
            calibrators.append(iso_reg)
        
        # Calibrate test predictions
        calibrated_proba = np.zeros_like(y_pred_proba)
        for i in range(3):
            calibrated_proba[:, i] = calibrators[i].transform(y_pred_proba[:, i])
        
        # Normalize to ensure probabilities sum to 1
        calibrated_proba = calibrated_proba / calibrated_proba.sum(axis=1, keepdims=True)
        
        y_proba = calibrated_proba
        y_pred = np.argmax(calibrated_proba, axis=1)
        
        final_model = lgb_model  # Store the base model
    else:
        # Already computed above
        y_proba = y_pred_proba
        final_model = lgb_model
    
    # Calculate metrics using Loop's multiclass_metrics
    metrics = multiclass_metrics(data, y_pred, y_proba)
    
    # Add validation loss to metrics
    metrics['val_loss'] = float(final_val_loss)
    
    # Calculate per-class metrics for extras
    from sklearn.metrics import classification_report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Feature importance (only for raw LightGBM)
    feature_importance = None
    if not use_calibration and hasattr(lgb_model, 'feature_importance'):
        feature_names = data.get('_feature_names', [])
        importance = lgb_model.feature_importance(importance_type='gain')
        if len(feature_names) == len(importance):
            feature_importance = dict(zip(feature_names, importance))
    
    # Prepare extras
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
    
    # Prepare results in UEL format
    round_results = metrics  # Contains precision, recall, auc, accuracy
    round_results['models'] = [final_model]  # Must be a list for UEL
    round_results['extras'] = extras
    round_results['_scaler'] = data.get('_scaler')  # Preserve scaler
    round_results['_preds'] = y_pred  # Store predictions
    
    return round_results
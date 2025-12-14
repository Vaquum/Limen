#!/usr/bin/env python3
'''
Long-Only Regime Binary Classifier Using TabPFN
Converted from Manifest pattern to prep/model pattern
Uses same features as ridge_classifier but with TabPFN model
'''
import numpy as np
import polars as pl
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from sklearn.preprocessing import StandardScaler

from loop.utils.splits import split_sequential, split_data_to_prep_output
from loop.metrics.binary_metrics import binary_metrics

# Feature computation imports
from loop.indicators import roc as compute_roc
from loop.indicators import ppo as compute_ppo
from loop.indicators import rolling_volatility
from loop.indicators import wilder_rsi


CONFIG = {
    'train_split': 0.6,
    'val_split': 0.2,
    'test_split': 0.2,
    'random_state': 42,
}

EXCLUDE_COLS = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'label']


def params():
    return {
        # Feature params
        'roc_period': [4, 8],
        'ppo_fast': [12],
        'ppo_slow': [26],
        'ppo_signal': [9],
        'rsi_period': [14],
        'volatility_window': [24],
        'q': [0.32, 0.37],

        # TabPFN params
        'n_estimators': [4, 8],
        'softmax_temperature': [0.9],
        'pred_threshold': [0.50, 0.55],
    }


def prep(data: pl.DataFrame, round_params=None):
    '''Prepare data with technical features for regime classification'''
    all_datetimes = data['datetime'].to_list()

    if round_params is None:
        round_params = {}

    df = data.clone()

    # Get params with defaults
    roc_period = round_params.get('roc_period', 4)
    ppo_fast = round_params.get('ppo_fast', 12)
    ppo_slow = round_params.get('ppo_slow', 26)
    ppo_signal = round_params.get('ppo_signal', 9)
    rsi_period = round_params.get('rsi_period', 14)
    volatility_window = round_params.get('volatility_window', 24)
    q = round_params.get('q', 0.35)

    # Compute ROC indicator
    df = compute_roc(df, period=roc_period)
    roc_col = f'roc_{roc_period}'

    # Compute PPO indicator
    df = compute_ppo(df, fast_period=ppo_fast, slow_period=ppo_slow, signal_period=ppo_signal)

    # Compute RSI
    df = wilder_rsi(df, period=rsi_period)

    # Compute rolling volatility
    df = rolling_volatility(df, column='close', window=volatility_window)

    # Add basic price features
    df = df.with_columns([
        # Returns
        (pl.col('close').pct_change()).alias('returns'),
        # Range
        ((pl.col('high') - pl.col('low')) / pl.col('close')).alias('range_pct'),
        # Close position in range
        ((pl.col('close') - pl.col('low')) / (pl.col('high') - pl.col('low') + 1e-10)).alias('close_position'),
        # Volume change
        (pl.col('volume').pct_change()).alias('volume_change'),
    ])

    # Create binary label based on ROC quantile
    roc_values = df[roc_col].drop_nulls().to_numpy()
    cutoff = np.quantile(roc_values, q)

    # Shift label by -1 (predict next bar's regime)
    df = df.with_columns([
        pl.when(pl.col(roc_col) > cutoff).then(1).otherwise(0).shift(-1).alias('label')
    ])

    # Drop nulls
    df_clean = df.drop_nulls()

    if len(df_clean) == 0:
        raise ValueError("No data left after cleaning")

    # Get numeric feature columns
    feature_cols = []
    for col in df_clean.columns:
        if col not in EXCLUDE_COLS:
            dtype = df_clean.schema[col]
            if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]:
                feature_cols.append(col)

    # Split data
    train_split = CONFIG['train_split']
    val_split = CONFIG['val_split']
    test_split = CONFIG['test_split']
    split_ratios = [int(train_split * 10), int(val_split * 10), int(test_split * 10)]

    cols = ['datetime'] + feature_cols + ['label']
    split_data = split_sequential(df_clean, ratios=split_ratios)
    data_dict = split_data_to_prep_output(split_data, cols, all_datetimes)

    # Convert to numpy
    data_dict['x_train'] = data_dict['x_train'].to_numpy()
    data_dict['y_train'] = data_dict['y_train'].to_numpy()
    data_dict['x_val'] = data_dict['x_val'].to_numpy()
    data_dict['y_val'] = data_dict['y_val'].to_numpy()
    data_dict['x_test'] = data_dict['x_test'].to_numpy()
    data_dict['y_test'] = data_dict['y_test'].to_numpy()

    # Scale features
    scaler = StandardScaler()
    data_dict['x_train'] = scaler.fit_transform(data_dict['x_train'])
    data_dict['x_val'] = scaler.transform(data_dict['x_val'])
    data_dict['x_test'] = scaler.transform(data_dict['x_test'])
    data_dict['_scaler'] = scaler
    data_dict['_feature_names'] = feature_cols
    data_dict['_cutoff'] = cutoff

    return data_dict


def model(data: dict, round_params: dict) -> dict:
    '''TabPFN binary classification for regime prediction'''
    n_estimators = round_params.get('n_estimators', 8)
    softmax_temperature = round_params.get('softmax_temperature', 0.9)
    pred_threshold = round_params.get('pred_threshold', 0.5)

    X_train = data['x_train']
    X_val = data['x_val']
    X_test = data['x_test']
    y_train = data['y_train']
    y_val = data['y_val']

    # Combine train and val for TabPFN
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    # Check for single-class issue
    unique_classes = np.unique(y_train_full)
    if len(unique_classes) < 2:
        y_proba = np.full(len(X_test), 0.5)
        y_pred = np.zeros(len(X_test), dtype=np.int8)
        round_results = binary_metrics(data, y_pred, y_proba)
        round_results['_preds'] = y_pred
        round_results['extras'] = {'single_class': True}
        return round_results

    # Initialize TabPFN
    clf = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        n_estimators=n_estimators,
        softmax_temperature=softmax_temperature,
        random_state=CONFIG['random_state'],
        device='cpu',
        ignore_pretraining_limits=True,
    )

    # Fit
    clf.fit(X_train_full, y_train_full)

    # Predict
    y_proba_raw = clf.predict_proba(X_test)

    # Handle single-class output
    if y_proba_raw.shape[1] == 1:
        y_proba = np.full(len(X_test), 0.5)
    else:
        y_proba = y_proba_raw[:, 1]

    y_pred = (y_proba >= pred_threshold).astype(np.int8)

    round_results = binary_metrics(data, y_pred, y_proba)
    round_results['_preds'] = y_pred
    round_results['models'] = [clf]
    round_results['_scaler'] = data.get('_scaler')
    round_results['extras'] = {
        'model_type': 'TabPFN',
        'n_estimators': n_estimators,
        'softmax_temperature': softmax_temperature,
        'cutoff': data.get('_cutoff'),
        'class_distribution': dict(zip(*np.unique(y_train_full, return_counts=True))),
    }

    return round_results

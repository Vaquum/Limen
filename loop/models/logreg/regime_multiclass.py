'''
SFM Label Model for Breakout Regime Classification using Logistic Regression
'''
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from datetime import timedelta

from loop.utils.splits import split_sequential
from loop.utils.safe_ovr_auc import safe_ovr_auc
from loop.models.lightgbm.utils.regime_multiclass import (
    build_sample_dataset_for_regime_multiclass,
    add_features_to_regime_multiclass_dataset
)
from loop.transforms.logreg_transform import LogRegTransform

# Configuration constants (same as LightGBM version)
PERCENTAGE = 5
LONG_COL = f'long_0_0{PERCENTAGE}'
SHORT_COL = f'short_0_0{PERCENTAGE}'
NUM_ROWS = 10000
TARGET_COLUMN = 'average_price'
EMA_SPAN = 6  # 6 x (12 x 2h kline)
INTERVAL_SEC = 7200  # 2 hour intervals
LOOKAHEAD_HOURS = 24  # 24 hour lookahead
LOOKBACK_BARS = 12  # look-back bars (12×2h = 1 day)
LEAKAGE_SHIFT = 12  # shift to prevent leakage (12×2h = 1 day)
TRAIN_SPLIT = 5
VAL_SPLIT = 3
TEST_SPLIT = 2
CONFIDENCE_THRESHOLD = 0.40  # Minimum confidence to make a prediction (otherwise classify as flat)

# All breakout % thresholds we track
DELTAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

def params():
    '''Return hyperparameter search space for Logistic Regression.'''
    p = {
        # Logistic Regression specific parameters
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
        'max_iter': [100, 200, 500, 1000, 2000],
        'tol': [0.0001, 0.001, 0.01, 0.1],
        'class_weight': [None, 'balanced'],
        'l1_ratio': [0.15, 0.5, 0.85],  # Only used when penalty='elasticnet'
        'multi_class': ['ovr', 'multinomial'],  # ovr = one-vs-rest, multinomial = softmax
        'fit_intercept': [True],
        'random_state': [42],
    }
    return p


def prep(data):
    '''Prepare data for training - follows template signature.'''
    # Random sequential sample (same as LightGBM version)
    df = build_sample_dataset_for_regime_multiclass(
        data,
        datetime_col='datetime',
        target_col=TARGET_COLUMN,
        interval_sec=INTERVAL_SEC,
        lookahead=timedelta(hours=LOOKAHEAD_HOURS),
        ema_span=EMA_SPAN * LOOKBACK_BARS,
        deltas=DELTAS,
        long_col=LONG_COL,
        short_col=SHORT_COL,
        leakage_shift_bars=LEAKAGE_SHIFT,
        random_slice_size=NUM_ROWS,
    )
    df = add_features_to_regime_multiclass_dataset(
        df,
        lookback_bars=LOOKBACK_BARS,
        long_col=LONG_COL,
        short_col=SHORT_COL,
    )

    # Feature selection - exclude leak columns and target
    LEAK_PREFIXES = ('long_0_', 'short_0_')
    X_cols = [
        c for c in df.columns
        if not c.startswith(LEAK_PREFIXES)
           and c not in ('datetime', 'regime')
    ]

    train, val, test = split_sequential(data=df, ratios=(TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT))

    # Convert to numpy arrays for sklearn
    x_train = train.select(X_cols).to_numpy()
    y_train = train.select('regime').to_numpy().ravel()
    x_val = val.select(X_cols).to_numpy()
    y_val = val.select('regime').to_numpy().ravel()
    x_test = test.select(X_cols).to_numpy()
    y_test = test.select('regime').to_numpy().ravel()

    # Create Polars DataFrames for scaling
    x_train_df = pl.DataFrame(x_train, schema=X_cols)
    x_val_df = pl.DataFrame(x_val, schema=X_cols)
    x_test_df = pl.DataFrame(x_test, schema=X_cols)

    # Scale features using LogRegTransform
    scaler = LogRegTransform(x_train_df)
    x_train_scaled = scaler.transform(x_train_df).to_numpy()
    x_val_scaled = scaler.transform(x_val_df).to_numpy()
    x_test_scaled = scaler.transform(x_test_df).to_numpy()

    return {
        'x_train': x_train_scaled,
        'x_val': x_val_scaled,
        'x_test': x_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        '_scaler': scaler,
        '_feature_names': X_cols,
    }


def model(data, round_params):
    '''Train Logistic Regression multiclass model and evaluate.'''
    # Validate all classes present in splits
    for split_name, y_data in [('train', data['y_train']),
                               ('val', data['y_val']),
                               ('test', data['y_test'])]:
        if len(np.unique(y_data)) < 3:
            raise ValueError(f'{split_name} split missing one of the classes 0/1/2')

    # Handle solver compatibility
    params = round_params.copy()
    
    # Solver compatibility checks
    if params['solver'] == 'liblinear':
        params['multi_class'] = 'ovr'  # liblinear only supports ovr
        if params['penalty'] == 'elasticnet':
            params['penalty'] = 'l2'  # liblinear doesn't support elasticnet
    
    if params['penalty'] == 'elasticnet' and params['solver'] not in ['saga']:
        params['solver'] = 'saga'  # Only saga supports elasticnet
    
    if params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']:
        params['solver'] = 'saga'  # Use saga for l1
    
    if params['multi_class'] == 'multinomial' and params['solver'] == 'liblinear':
        params['solver'] = 'lbfgs'  # liblinear doesn't support multinomial
    
    # Remove l1_ratio if not using elasticnet
    if params['penalty'] != 'elasticnet':
        params.pop('l1_ratio', None)
    
    # Create and train the model
    clf = LogisticRegression(
        penalty=params['penalty'],
        C=params['C'],
        solver=params['solver'],
        max_iter=params['max_iter'],
        tol=params['tol'],
        class_weight=params['class_weight'],
        multi_class=params['multi_class'],
        fit_intercept=params['fit_intercept'],
        random_state=params['random_state'],
        l1_ratio=params.get('l1_ratio', None) if params['penalty'] == 'elasticnet' else None,
        verbose=0,
        n_jobs=-1  # Use all cores
    )
    
    # Train the model
    clf.fit(data['x_train'], data['y_train'])
    
    # Get prediction probabilities
    proba = clf.predict_proba(data['x_test'])
    
    # Get predicted classes and confidence
    conf = proba.max(axis=1)
    regime = proba.argmax(axis=1)
    
    # Confidence gate: If confidence is below threshold, classify as flat (regime 0)
    # This helps reduce false positives and focuses on high-confidence predictions
    regime[conf < CONFIDENCE_THRESHOLD] = 0
    
    # Calculate metrics
    round_results = {
        'precision': round(precision_score(data['y_test'], regime, average='macro', zero_division=0), 2),
        'recall': round(recall_score(data['y_test'], regime, average='macro', zero_division=0), 2),
        'f1score': round(f1_score(data['y_test'], regime, average='macro', zero_division=0), 2),
        'auc': round(safe_ovr_auc(data['y_test'], proba), 2),
        'accuracy': round(accuracy_score(data['y_test'], regime), 2),
    }
    
    return round_results


'''
--- Example usage ---

import loop
from loop.models.logreg import regime_multiclass

context_params = {
    'kline_size': [7200],
    'start_date_limit': ['2019-01-01 00:00:00'],
    'breakout_percentage': [5],
    'n_permutations': [48],
    'random_sample_size': [10000]
}

context_params = loop.utils.ParamSpace(context_params)
p = context_params.generate()

historical = loop.HistoricalData()
historical.get_historical_klines(
    kline_size=p['kline_size'],
    start_date_limit=p['start_date_limit']
)

uel = loop.UniversalExperimentLoop(historical.data, regime_multiclass)
uel.run(
    experiment_name=f"test_label_model_regime_logreg_random_{p['random_sample_size']}_2H_breakout{p['breakout_percentage']}%",
    n_permutations=p['n_permutations'],
    prep_each_round=False,
    random_search=True,
)

print(uel.log_df.head())
'''
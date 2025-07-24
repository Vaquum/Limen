# General Guidelines
# - Keep comments to minimum in SFMs
# - No docstrings for SFMs
# - Constants all capital
# - All other variables lowercase

# Start the SFM with a title wrapped in single quotes
'SFM Label Model for Breakout Regime Classification using Logistic Regression'

# Import whole 3rd-party libraries
import numpy as np
import polars as pl

# Import parts of 3rd-party libraries (leave empty line above)
from sklearn.linear_model import LogisticRegression
from datetime import timedelta

# Import parts of our own libraries (leave empty line above)
from loop.utils.splits import split_sequential, split_data_to_prep_output
from loop.sfm.lightgbm.utils.regime_multiclass import build_sample_dataset_for_regime_multiclass
from loop.sfm.lightgbm.utils.regime_multiclass import add_features_to_regime_multiclass_dataset
from loop.transforms.logreg_transform import LogRegTransform
from loop.metrics.multiclass_metrics import multiclass_metrics

# Add configuration constants (leave empty line above)
BREAKOUT_PERCENTAGE = 5
LONG_COL = f'long_0_0{BREAKOUT_PERCENTAGE}'
SHORT_COL = f'short_0_0{BREAKOUT_PERCENTAGE}'
NUM_ROWS = 10000
TARGET_COLUMN = 'average_price'
EMA_SPAN = 6
INTERVAL_SEC = 7200
LOOKAHEAD_HOURS = 24
LOOKBACK_BARS = 12
LEAKAGE_SHIFT = 12 
TRAIN_SPLIT = 5
VAL_SPLIT = 3
TEST_SPLIT = 2
CONFIDENCE_THRESHOLD = 0.40
DELTAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

# Add sfm.params function (leave two empty lines above)
def params():

    # Leave one empty line above
    p = {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
        'max_iter': [100, 200, 500, 1000, 2000],
        'tol': [0.0001, 0.001, 0.01, 0.1],
        'class_weight': [None, 'balanced'],
        'l1_ratio': [0.15, 0.5, 0.85],
        'multi_class': ['ovr', 'multinomial'],
        'fit_intercept': [True],
        'random_state': [42],
    }
    return p

# Add sfm.prep function (leave two empty lines above)
def prep(data):

    # Leave one empty line above
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

    # This is an example for SFM where column names can change for permutations 
    LEAK_PREFIXES = ('long_0_', 'short_0_')
    cols = [
        c for c in df.columns
        if not c.startswith(LEAK_PREFIXES)
           and c not in ('datetime', 'regime')
    ]

    df = df.select(cols)

    # Always use split_sequential for data splitting
    split_data = split_sequential(data=df, ratios=(TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT))
    
    # Always use split_data_to_prep_output for getting the standard data_dict
    data_dict = split_data_to_prep_output(split_data, cols)

    '''
    x_train = train.select(cols).to_numpy()
    y_train = train.select.to_numpy().ravel()
    x_val = val.select(cols).to_numpy()
    y_val = val.select('regime').to_numpy().ravel()
    x_test = test.select(cols).to_numpy()
    y_test = test.to_numpy().ravel()

    x_train_df = pl.DataFrame(x_train, schema=cols)
    x_val_df = pl.DataFrame(x_val, schema=cols)
    x_test_df = pl.DataFrame(x_test, schema=cols)

    
    x_train_scaled = scaler.transform(x_train_df).to_numpy()
    x_val_scaled = scaler.transform(x_val_df).to_numpy()
    x_test_scaled = scaler.transform(x_test_df).to_numpy()

    '''
    # 
    scaler = LogRegTransform(data_dict['x_train'])
    for col in data_dict.keys():
        if col.startswith('x_'):
            data_dict[col] = scaler.transform(data_dict[col])

    data_dict['_scaler'] = scaler
    data_dict['_feature_names'] = cols

    return data_dict

# Add sfm.model function (leave two empty lines above)
def model(data, round_params):
    
    params = round_params.copy()

    if params['solver'] == 'liblinear':
        params['multi_class'] = 'ovr'
        if params['penalty'] == 'elasticnet':
            params['penalty'] = 'l2'
    
    if params['penalty'] == 'elasticnet' and params['solver'] not in ['saga']:
        params['solver'] = 'saga'
    
    if params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']:
        params['solver'] = 'saga'
    
    if params['multi_class'] == 'multinomial' and params['solver'] == 'liblinear':
        params['solver'] = 'lbfgs'
    
    if params['penalty'] != 'elasticnet':
        params.pop('l1_ratio', None)
    
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
        l1_ratio=params.get('l1_ratio'),
        verbose=0,
        n_jobs=-1
    )
    
    clf.fit(data['x_train'], data['y_train'])
    
    prediction_probs = clf.predict_proba(data['x_test'])
    
    preds = prediction_probs.argmax(axis=1)
    probs = prediction_probs.max(axis=1)
    
    preds[probs < CONFIDENCE_THRESHOLD] = 0
    
    round_results = multiclass_metrics(data, preds, probs)
    
    return round_results

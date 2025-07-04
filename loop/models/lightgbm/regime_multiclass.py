'''
SFM Label Model for Breakout Regime Classification
'''
import numpy as np
import polars as pl
import lightgbm as lgb
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

# Configuration constants
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
    '''Return hyperparameter search space.'''
    p = {
        'objective': ['multiclass'],
        'metric': ['multi_logloss'],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'num_boost_round': [1500],
        'num_leaves': [15, 31, 63, 127, 255],
        'max_depth': [3, 5, 7, 9, -1],
        'min_data_in_leaf': [20, 50, 100, 200, 500],
        'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
        'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
        'bagging_freq': [0, 1, 5, 10, 20],
        'lambda_l1': [0.0, 0.1, 1.0, 10.0, 100.0],
        'lambda_l2': [0.0, 0.1, 1.0, 10.0, 100.0],
        'feature_pre_filter': ['false'],
        'stopping_round': [100],
        'logging_step':[100],
        'predict_probability_cutoff': [0.5],
    }
    return p


def prep(data):
    '''Prepare data for training - follows template signature.'''
    # Random sequential sample
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

    train_pd = train.to_pandas()
    val_pd = val.to_pandas()
    test_pd = test.to_pandas()

    # Cast features to float32
    for _df in (train_pd, val_pd, test_pd):
        _df[X_cols] = _df[X_cols].astype('float32')

    x_train, y_train = train_pd[X_cols], train_pd['regime']
    x_val, y_val = val_pd[X_cols], val_pd['regime']
    x_test, y_test = test_pd[X_cols], test_pd['regime']

    dtrain = lgb.Dataset(x_train, label=y_train)
    dval = lgb.Dataset(x_val, label=y_val, reference=dtrain)

    return {
        'dtrain': dtrain,
        'dval': dval,
        'train_X': x_train,
        'val_X': x_val,
        'test_X': x_test,
        'train_y': y_train,
        'val_y': y_val,
        'test_y': y_test,
    }


def model(data, round_params):
    '''Train LightGBM multiclass model and evaluate.'''
    # Make LightGBM multiclass
    round_params = round_params.copy()
    round_params.update({
        'num_class': 3,  # 0 = flat, 1 = bullish, 2 = bearish
        'verbose': -1,
    })

    # Validate all classes present in splits
    for split_name, y_data in [('train', data['train_y']),
                               ('val', data['val_y']),
                               ('test', data['test_y'])]:
        if len(np.unique(y_data)) < 3:
            raise ValueError(f'{split_name} split missing one of the classes 0/1/2')

    model = lgb.train(
        params=round_params,
        train_set=data['dtrain'],
        num_boost_round=round_params['num_boost_round'],
        valid_sets=[data['dtrain'], data['dval']],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(round_params['stopping_round'], verbose=False),
            lgb.log_evaluation(round_params['logging_step'])
        ]
    )

    proba = model.predict(data['test_X'], num_iteration=model.best_iteration)

    if proba.ndim == 1:  # no axis-1
        conf = proba  # already 1-D
        regime = (proba >= round_params['predict_probability_cutoff']).astype(int)
    else:  # proper 3-class probs
        conf = proba.max(axis=1)
        regime = proba.argmax(axis=1)

    # Confidence gate (works for both shapes)
    # If confidence is below threshold, classify as flat (regime 0)
    # This helps reduce false positives and focuses on high-confidence predictions
    regime[conf < CONFIDENCE_THRESHOLD] = 0

    round_results = {
        'precision': round(precision_score(data['test_y'], regime, average='macro'), 2),
        'recall': round(recall_score(data['test_y'], regime, average='macro'), 2),
        'f1score': round(f1_score(data['test_y'], regime, average='macro'), 2),
        'auc': round(safe_ovr_auc(data['test_y'], proba), 2),
        'accuracy': round(accuracy_score(data['test_y'], regime), 2),
    }

    return round_results


'''
--- Example usage ---

import loop
from loop.models.lightgbm import regime_multiclass

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
    experiment_name=f"test_label_model_regime_random_{p['random_sample_size']}_2H_breakout{p['breakout_percentage']}%",
    n_permutations=p['n_permutations'],
    prep_each_round=False,
    random_search=True,
)

print(uel.log_df.head())
'''
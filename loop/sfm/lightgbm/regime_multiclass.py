'SFM Label Model for Breakout Regime Classification'

import numpy as np
import polars as pl
import lightgbm as lgb

from datetime import timedelta

from loop.utils.splits import split_sequential, split_data_to_prep_output
from loop.sfm.lightgbm.utils.regime_multiclass import build_sample_dataset_for_regime_multiclass
from loop.sfm.lightgbm.utils.regime_multiclass import add_features_to_regime_multiclass_dataset
from loop.metrics.multiclass_metrics import multiclass_metrics
from loop.transforms.logreg_transform import LogRegTransform

PERCENTAGE = 5
LONG_COL = f'long_0_0{PERCENTAGE}'
SHORT_COL = f'short_0_0{PERCENTAGE}'
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

def params():

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
        'logging_step':[0],
        'predict_probability_cutoff': [0.5],
    }
    return p


def prep(data):

    all_datetimes = data['datetime'].to_list()

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

    LEAK_PREFIXES = ('long_0_', 'short_0_')
    cols = [
        c for c in df.columns
        if not c.startswith(LEAK_PREFIXES)
           and c not in ('datetime', 'regime')
    ]

    df = df.with_columns([
        pl.col(cols).cast(pl.Float32)
    ])

    cols += ['datetime'] + ['regime']

    split_data = split_sequential(data=df, ratios=(TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT))
    
    data_dict = split_data_to_prep_output(split_data, cols, all_datetimes)

    scaler = LogRegTransform(data_dict['x_train'])
    for col in data_dict.keys():
        if col.startswith('x_'):
            data_dict[col] = scaler.transform(data_dict[col])

    data_dict['_scaler'] = scaler
    data_dict['_feature_names'] = cols

    data_dict['dtrain'] = lgb.Dataset(data_dict['x_train'], label=data_dict['y_train'].to_numpy())
    data_dict['dval'] = lgb.Dataset(data_dict['x_val'], label=data_dict['y_val'].to_numpy(), reference=data_dict['dtrain'])

    return data_dict


def model(data, round_params):

    round_params = round_params.copy()
    round_params.update({
        'num_class': 3,
        'verbose': -1,
    })

    for split_name, y_data in [('train', data['y_train']),
                               ('val', data['y_val']),
                               ('test', data['y_test'])]:
        if len(np.unique(y_data)) < 3:
            raise ValueError(f'{split_name} split missing one of the classes 0/1/2')

    round_params = round_params.copy()
    round_params.update({
        'verbose': -1,
    })

    model = lgb.train(
        params=round_params,
        train_set=data['dtrain'],
        num_boost_round=round_params['num_boost_round'],
        valid_sets=[data['dtrain'], data['dval']],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(round_params['stopping_round'], verbose=False),
            lgb.log_evaluation(round_params['logging_step'])
        ])

    prediction_probs = model.predict(data['x_test'], num_iteration=model.best_iteration)

    if prediction_probs.ndim == 1:
        probs = prediction_probs
        preds = (prediction_probs >= round_params['predict_probability_cutoff']).astype(int)
    else:
        probs = prediction_probs.max(axis=1)
        preds = prediction_probs.argmax(axis=1)

    preds[probs < CONFIDENCE_THRESHOLD] = 0

    round_results = multiclass_metrics(data, preds, prediction_probs)

    return round_results

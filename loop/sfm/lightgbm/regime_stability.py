'SFM Regime Stability Model - Enhanced regime classification with stability filtering'

import numpy as np
import polars as pl
import lightgbm as lgb

from sklearn.metrics import accuracy_score
from datetime import timedelta

from loop.utils.splits import split_sequential, split_data_to_prep_output
from loop.sfm.lightgbm.utils.regime_multiclass import build_sample_dataset_for_regime_multiclass
from loop.sfm.lightgbm.utils.regime_multiclass import add_features_to_regime_multiclass_dataset
from loop.sfm.lightgbm.utils.regime_stability import add_stability_features
from loop.metrics.multiclass_metrics import multiclass_metrics

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
PERSISTENCE_THRESHOLD = 0.60
DELTAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
STABILITY_NUM_BOOST_ROUND = 500
STABILITY_STOPPING_ROUND = 50
STABILITY_LEARNING_RATE = 0.03
NUM_CLASSES = 3
TRAIN_VERBOSE = -1
MIN_TRAIN_SIZE_FOR_VALIDATION = 100
SYNTHETIC_BASE_PRICE = 50000
FLOAT_PRECISION = 2
MIN_CLASSES_REQUIRED = 3


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
        'logging_step': [100],
        'predict_probability_cutoff': [0.5],
        'stability_learning_rate': [0.01, 0.03, 0.05],
        'stability_num_boost_round': [500],
        'stability_stopping_round': [50],
    }
    return p


def prep(data):
    
    price_cols = ['open', 'high', 'low', 'close', 'average_price']
    available_price_cols = [col for col in price_cols if col in data.columns]
    
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
        random_slice_min_pct=0.05,
        random_slice_max_pct=0.95
    )
    
    datetime_values = df.select('datetime')
    
    if available_price_cols:
        price_data = data.select(['datetime'] + available_price_cols)
        df = df.join(price_data, on='datetime', how='left')
    
    df = add_features_to_regime_multiclass_dataset(
        df,
        lookback_bars=LOOKBACK_BARS,
        long_col=LONG_COL,
        short_col=SHORT_COL,
    )
    
    if 'long_close' not in df.columns:
        if 'close' in df.columns:
            df = df.with_columns([
                pl.col('close').alias('long_close'),
                pl.col('close').alias('short_close'),
            ])
        elif 'average_price' in df.columns:
            df = df.with_columns([
                pl.col('average_price').alias('long_close'),
                pl.col('average_price').alias('short_close'),
            ])
        else:
            df = df.with_columns([
                pl.lit(SYNTHETIC_BASE_PRICE).alias('long_close'),
                pl.lit(SYNTHETIC_BASE_PRICE).alias('short_close'),
            ])
    
    df = add_stability_features(df, leakage_shift=LEAKAGE_SHIFT)
    
    df = df.drop_nulls()
    
    LEAK_PREFIXES = ('long_0_', 'short_0_')
    EXCLUDE_COLS = ('datetime', 'regime', 'regime_persists', 'current_regime_sign', 'future_regime_sign')
    
    all_features = [
        c for c in df.columns
        if not c.startswith(LEAK_PREFIXES)
        and c not in EXCLUDE_COLS
    ]
    
    stability_feature_patterns = ['vol_', 'range_', 'activity_', 'bias_vol_', 'slope_', 
                                 'consistency_', 'alignment', 'regime_age', 'min_regime_age',
                                 'price_above_ema', 'regime_age_ratio']
        
    stability_features = [c for c in df.columns if any(pattern in c for pattern in stability_feature_patterns)]
    
    primary_model_cols = all_features + ['regime']
    df_primary_model = df.select(primary_model_cols)
    primary_model_split_data = split_sequential(
        data=df_primary_model,
        ratios=(TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)
    )
    primary_model_data_dict = split_data_to_prep_output(primary_model_split_data, primary_model_cols)

    primary_model_data_dict['dtrain'] = lgb.Dataset(
        primary_model_data_dict['x_train'],
        label=primary_model_data_dict['y_train'].to_numpy()
    )
    primary_model_data_dict['dval'] = lgb.Dataset(
        primary_model_data_dict['x_val'],
        label=primary_model_data_dict['y_val'].to_numpy(),
        reference=primary_model_data_dict['dtrain']
    )

    stability_model_cols = stability_features + ['regime_persists']
    df_stability_model = df.select(stability_model_cols)
    stability_model_split_data = split_sequential(
        data=df_stability_model,
        ratios=(TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)
    )
    stability_model_data_dict = split_data_to_prep_output(stability_model_split_data, stability_model_cols)

    stability_model_data_dict['dtrain'] = lgb.Dataset(
        stability_model_data_dict['x_train'],
        label=stability_model_data_dict['y_train'].to_numpy()
    )
    stability_model_data_dict['dval'] = lgb.Dataset(
        stability_model_data_dict['x_val'],
        label=stability_model_data_dict['y_val'].to_numpy(),
        reference=stability_model_data_dict['dtrain']
    )

    return {
        'primary_model_data_dict': primary_model_data_dict,
        'stability_model_data_dict': stability_model_data_dict,
        'num_main_features': len(all_features),
        'num_stability_features': len(stability_features),
    }


def model(data, round_params):

    primary_data, stability_data = (data['primary_model_data_dict'], data['stability_model_data_dict'])

    if 'x_train' not in  stability_data or data['num_stability_features'] == 0:
        raise ValueError("No stability features found in data - check prep() function")
    
    primary_params = round_params.copy()
    primary_params.update({
        'num_class': NUM_CLASSES,
        'verbose': TRAIN_VERBOSE,
    })
    
    for split_name, y_data in [('train', primary_data['y_train']),
                               ('val', primary_data['y_val']),
                               ('test', primary_data['y_test'])]:
        unique_classes = np.unique(y_data)
        if len(unique_classes) < MIN_CLASSES_REQUIRED:
            print(f"Warning: {split_name} split has only classes {unique_classes}")
            raise ValueError(f'{split_name} split missing one of the classes 0/1/2. Try increasing data size.')
    
    primary_model = lgb.train(
        params=primary_params,
        train_set=primary_data['dtrain'],
        num_boost_round=round_params['num_boost_round'],
        valid_sets=[primary_data['dtrain'], primary_data['dval']],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(round_params['stopping_round'], verbose=False),
            lgb.log_evaluation(round_params['logging_step'])
        ]
    )
    
    stability_params = round_params.copy()
    stability_params.update({
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': round_params.get('stability_learning_rate', STABILITY_LEARNING_RATE),
        'verbose': TRAIN_VERBOSE,
    })
    
    stability_model = lgb.train(
        params=stability_params,
        train_set=stability_data['dtrain'],
        num_boost_round=round_params.get('stability_num_boost_round', STABILITY_NUM_BOOST_ROUND),
        valid_sets=[stability_data['dtrain'], stability_data['dval']],
        valid_names=['train_stability', 'valid_stability'],
        callbacks=[
            lgb.early_stopping(round_params.get('stability_stopping_round', STABILITY_STOPPING_ROUND), verbose=False)
        ]
    )
    
    regime_proba = primary_model.predict(primary_data['x_test'], num_iteration=primary_model.best_iteration)
    persistence_proba = stability_model.predict(stability_data['x_test'], num_iteration=stability_model.best_iteration)
    
    preds = regime_proba.argmax(axis=1)
    probs = regime_proba.max(axis=1)
    
    filtered_pred = preds.copy()
    confidence_threshold = round_params.get('predict_probability_cutoff', CONFIDENCE_THRESHOLD)
    low_confidence = (probs < confidence_threshold) | (persistence_proba < PERSISTENCE_THRESHOLD)
    filtered_pred[low_confidence] = 0
    
    filter_rate = low_confidence.mean()
    unfiltered_acc = accuracy_score(primary_data['y_test'], preds)
    
    round_results = multiclass_metrics(primary_data, preds, regime_proba)

    round_results['extras'] = {
            'filter_rate': round(filter_rate, FLOAT_PRECISION),
            'unfiltered_accuracy': round(unfiltered_acc, FLOAT_PRECISION),
            'num_main_features': data['num_main_features'],
            'num_stability_features': data['num_stability_features']
            }
    
    return round_results

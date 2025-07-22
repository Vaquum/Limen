'''
SFM Regime Stability Model - Enhanced regime classification with stability filtering
Predicts regime (flat/bullish/bearish) with additional stability analysis to filter uncertain predictions
'''

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from datetime import timedelta

from loop.utils.splits import split_sequential
from loop.utils.safe_ovr_auc import safe_ovr_auc
from loop.models.lightgbm.utils.regime_multiclass import (
    build_sample_dataset_for_regime_multiclass, 
    add_features_to_regime_multiclass_dataset
)
from loop.models.lightgbm.utils.regime_stability import add_stability_features

from loop.utils.metrics import multiclass_metrics

# Configuration constants
PERCENTAGE = 5
LONG_COL = f'long_0_0{PERCENTAGE}'
SHORT_COL = f'short_0_0{PERCENTAGE}'
NUM_ROWS = 10000
TARGET_COLUMN = 'average_price'
EMA_SPAN = 6  # 6 x (12 x 2h kline)
INTERVAL_SEC = 7200  # 2 hour intervals
LOOKAHEAD_HOURS = 24  # 24 hour lookahead for stability prediction
LOOKBACK_BARS = 12  # look-back bars (12×2h = 1 day)
LEAKAGE_SHIFT = 12  # shift to prevent leakage (12×2h = 1 day)
TRAIN_SPLIT = 5
VAL_SPLIT = 3
TEST_SPLIT = 2
CONFIDENCE_THRESHOLD = 0.40  # Main model confidence threshold
PERSISTENCE_THRESHOLD = 0.60  # Stability model threshold for regime persistence

# All breakout % thresholds we track
DELTAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

# Default stability model parameters
STABILITY_NUM_BOOST_ROUND = 500
STABILITY_STOPPING_ROUND = 50
STABILITY_LEARNING_RATE = 0.03

# Class constants
NUM_CLASSES = 3  # 0 = flat, 1 = bullish, 2 = bearish
TRAIN_VERBOSE = -1  # LightGBM verbosity
MIN_TRAIN_SIZE_FOR_VALIDATION = 100  # Minimum training size to enforce all classes
SYNTHETIC_BASE_PRICE = 50000  # Fallback price when no price data available
FLOAT_PRECISION = 2  # Decimal places for rounding metrics
MIN_CLASSES_REQUIRED = 3  # Minimum number of classes required


def params():
    '''
    Return hyperparameter search space for the regime stability model.
    
    Returns:
        dict: Dictionary of hyperparameter lists for random search
    '''
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
        
        # Stability filter params
        'stability_learning_rate': [0.01, 0.03, 0.05],
        'stability_num_boost_round': [500],
        'stability_stopping_round': [50],
    }
    return p


def prep(data):
    '''
    Prepare data for training the regime stability model.
    
    This function:
    1. Builds the base regime dataset with breakout labels
    2. Adds lag features and regime counts
    3. Adds stability-specific features (volatility, activity, momentum)
    4. Creates stability labels for filtering
    5. Splits data and prepares LightGBM datasets
    
    Args:
        data (pl.DataFrame): Raw historical price data
        
    Returns:
        dict: Dictionary containing train/val/test datasets and features for
              both regime prediction and stability filtering
    '''
    
    # Store original price data before transformation
    price_cols = ['open', 'high', 'low', 'close', 'average_price']
    available_price_cols = [col for col in price_cols if col in data.columns]
    
    # Build base dataset with regime labels
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
    
    # Get the datetime values to rejoin price data
    datetime_values = df.select('datetime')
    
    # Rejoin the original price data
    if available_price_cols:
        price_data = data.select(['datetime'] + available_price_cols)
        df = df.join(price_data, on='datetime', how='left')
    
    # Add base features
    df = add_features_to_regime_multiclass_dataset(
        df,
        lookback_bars=LOOKBACK_BARS,
        long_col=LONG_COL,
        short_col=SHORT_COL,
    )
    
    # Ensure we have price columns for stability features
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
            # Create synthetic price data from regime signals as last resort
            df = df.with_columns([
                pl.lit(SYNTHETIC_BASE_PRICE).alias('long_close'),
                pl.lit(SYNTHETIC_BASE_PRICE).alias('short_close'),
            ])
    
    # Add stability features
    df = add_stability_features(df, leakage_shift=LEAKAGE_SHIFT)
    
    # Drop rows with NaNs
    df = df.drop_nulls()
    
    # Define feature sets
    LEAK_PREFIXES = ('long_0_', 'short_0_')
    EXCLUDE_COLS = ('datetime', 'regime', 'regime_persists', 'current_regime_sign', 'future_regime_sign')
    
    # Get all features for main model (including stability features)
    all_features = [
        c for c in df.columns
        if not c.startswith(LEAK_PREFIXES)
        and c not in EXCLUDE_COLS
    ]
    
    # Stability features (stability predictors only)
    stability_feature_patterns = ['vol_', 'range_', 'activity_', 'bias_vol_', 'slope_', 
                                 'consistency_', 'alignment', 'regime_age', 'min_regime_age',
                                 'price_above_ema', 'regime_age_ratio']
        
    stability_features = [c for c in df.columns if any(pattern in c for pattern in stability_feature_patterns)]
    
    # Split data
    train, val, test = split_sequential(data=df, ratios=(TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT))
    
    train_pd = train.to_pandas()
    val_pd = val.to_pandas()
    test_pd = test.to_pandas()
    
    # Cast features to float32
    for _df in (train_pd, val_pd, test_pd):
        _df[all_features] = _df[all_features].astype('float32')
        _df[stability_features] = _df[stability_features].astype('float32')
    
    # Prepare main model data
    x_train = train_pd[all_features]
    y_train = train_pd['regime']
    x_val = val_pd[all_features]
    y_val = val_pd['regime']
    x_test = test_pd[all_features]
    y_test = test_pd['regime']
    
    dtrain = lgb.Dataset(x_train, label=y_train)
    dval = lgb.Dataset(x_val, label=y_val, reference=dtrain)
    
    # Prepare stability model data
    x_train_stability = train_pd[stability_features]
    y_train_stability = train_pd['regime_persists']
    x_val_stability = val_pd[stability_features]
    y_val_stability = val_pd['regime_persists']
    x_test_stability = test_pd[stability_features]
    y_test_stability = test_pd['regime_persists']
    
    dtrain_stability = lgb.Dataset(x_train_stability, label=y_train_stability)
    dval_stability = lgb.Dataset(x_val_stability, label=y_val_stability, reference=dtrain_stability)
    
    return {
        # Main model data
        'dtrain': dtrain,
        'dval': dval,
        'train_X': x_train,
        'val_X': x_val,
        'test_X': x_test,
        'train_y': y_train,
        'val_y': y_val,
        'test_y': y_test,
        
        # Stability model data
        'dtrain_stability': dtrain_stability,
        'dval_stability': dval_stability,
        'train_X_stability': x_train_stability,
        'val_X_stability': x_val_stability,
        'test_X_stability': x_test_stability,
        'train_y_stability': y_train_stability,
        'val_y_stability': y_val_stability,
        'test_y_stability': y_test_stability,
        
        # Feature info
        'num_main_features': len(all_features),
        'num_stability_features': len(stability_features),
    }


def model(data, round_params):
    '''
    Train the regime stability model with dual prediction capabilities.
    
    This model:
    1. Predicts regime (flat/bullish/bearish) using enhanced features
    2. Predicts regime persistence using stability indicators
    3. Filters predictions based on confidence and persistence thresholds
    
    The key innovation is using stability features both for better regime
    prediction AND for filtering out unstable/uncertain predictions.
    
    Args:
        data (dict): Prepared datasets from prep() function
        round_params (dict): Hyperparameters for this training round
        
    Returns:
        dict: Performance metrics including precision, recall, f1score, auc,
              accuracy, plus additional metrics like filter_rate
    '''
    
    # Check if we have stability features
    if 'train_X_stability' not in data or data['num_stability_features'] == 0:
        raise ValueError("No stability features found in data - check prep() function")
    
    # 1. Train main regime model (with enhanced features)
    main_params = round_params.copy()
    main_params.update({
        'num_class': NUM_CLASSES,
        'verbose': TRAIN_VERBOSE,
    })
    
    # Validate all classes present
    for split_name, y_data in [('train', data['train_y']),
                               ('val', data['val_y']),
                               ('test', data['test_y'])]:
        unique_classes = np.unique(y_data)
        if len(unique_classes) < MIN_CLASSES_REQUIRED:
            print(f"Warning: {split_name} split has only classes {unique_classes}")
            raise ValueError(f'{split_name} split missing one of the classes 0/1/2. Try increasing data size.')
    
    main_model = lgb.train(
        params=main_params,
        train_set=data['dtrain'],
        num_boost_round=round_params['num_boost_round'],
        valid_sets=[data['dtrain'], data['dval']],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(round_params['stopping_round'], verbose=False),
            lgb.log_evaluation(round_params['logging_step'])
        ]
    )
    
    # 2. Train stability filter model
    stability_params = round_params.copy()
    stability_params.update({
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': round_params.get('stability_learning_rate', STABILITY_LEARNING_RATE),
        'verbose': TRAIN_VERBOSE,
    })
    
    stability_model = lgb.train(
        params=stability_params,
        train_set=data['dtrain_stability'],
        num_boost_round=round_params.get('stability_num_boost_round', STABILITY_NUM_BOOST_ROUND),
        valid_sets=[data['dtrain_stability'], data['dval_stability']],
        valid_names=['train_stability', 'valid_stability'],
        callbacks=[
            lgb.early_stopping(round_params.get('stability_stopping_round', STABILITY_STOPPING_ROUND), verbose=False)
        ]
    )
    
    # 3. Make filtered predictions
    regime_proba = main_model.predict(data['test_X'], num_iteration=main_model.best_iteration)
    persistence_proba = stability_model.predict(data['test_X_stability'], num_iteration=stability_model.best_iteration)
    
    # Get regime predictions and confidence
    preds = regime_proba.argmax(axis=1)
    probs = regime_proba.max(axis=1)
    
    # Apply dual filtering: confidence + stability
    filtered_pred = preds.copy()
    confidence_threshold = round_params.get('predict_probability_cutoff', CONFIDENCE_THRESHOLD)
    low_confidence = (probs < confidence_threshold) | (persistence_proba < PERSISTENCE_THRESHOLD)
    filtered_pred[low_confidence] = 0  # Set to flat when uncertain
    
    # Calculate metrics
    filter_rate = low_confidence.mean()
    unfiltered_acc = accuracy_score(data['test_y'], preds)
    
    round_results = multiclass_metrics(data, preds, preds)

    # Calculate final metrics
    round_results['extras'] = {
            'filter_rate': round(filter_rate, FLOAT_PRECISION),
            'unfiltered_accuracy': round(unfiltered_acc, FLOAT_PRECISION),
            'num_main_features': data['num_main_features'],
            'num_stability_features': data['num_stability_features']
            }
    
    return round_results


'''
--- Example usage ---

import loop
from loop.models.lightgbm import regime_stability

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

uel = loop.UniversalExperimentLoop(historical.data, regime_stability)
uel.run(
    experiment_name=f"regime_stability_random_{p['random_sample_size']}_2H_breakout{p['breakout_percentage']}%",
    n_permutations=p['n_permutations'],
    prep_each_round=False,
    random_search=True,
)

# Analysis of results
print("Results Summary:")
print(uel.log_df.head())
print(f"\nAverage filter rate: {uel.log_df['filter_rate'].mean():.2%}")
print(f"Accuracy improvement: {(uel.log_df['accuracy'] - uel.log_df['unfiltered_accuracy']).mean():.2%}")
'''
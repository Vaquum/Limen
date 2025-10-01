import numpy as np
import polars as pl
import lightgbm as lgb

from lightgbm import early_stopping, log_evaluation

from loop.features.time_features import time_features
from loop.features.lagged_features import lag_range_cols
from loop.indicators.sma import sma
from loop.utils.add_breakout_ema import add_breakout_ema
from loop.utils.random_slice import random_slice
from loop.metrics.binary_metrics import binary_metrics
from loop.manifest import Manifest
from loop.data import compute_data_bars


TARGET_COLUMN = 'liquidity_sum'
LOOKBACK_WINDOW_SIZE = 100
PREDICTION_HORIZON = 3
NUM_SLICES = 10
TARGET_COLUMN_CLASS = 'breakout_ema'


def manifest():
    return (Manifest()
        .set_pre_split_data_selector(
            random_slice,
            rows='random_slice_size',
            safe_range_low='random_slice_min_pct',
            safe_range_high='random_slice_max_pct',
            seed='random_seed'
        )
        .set_split_config(6, 2, 2)
        .set_bar_formation(compute_data_bars,
            bar_type='bar_type',
            trade_threshold='trade_threshold',
            volume_threshold='volume_threshold',
            liquidity_threshold='liquidity_threshold')
        .set_required_bar_columns([
            'datetime', 'high', 'low', 'open', 'close', 'mean',
            'volume', 'maker_ratio', 'no_of_trades', 'maker_volume', 'maker_liquidity'
        ])
        .add_feature(time_features)
        .add_indicator(sma, column='liquidity_sum', period=3)
        .add_feature(lambda data: data.with_columns([
            pl.col('liquidity_sum').ewm_mean(span=5).alias('liq_ewm_5'),
            pl.col('liquidity_sum').ewm_mean(span=10).alias('liq_ewm_10')
        ]))
        .add_feature(lag_range_cols,
            cols=['liquidity_sum', 'hour', 'minute', 'weekday',
                  'liquidity_sum_sma_3', 'liq_ewm_5', 'liq_ewm_10'],
            start=1,
            end=LOOKBACK_WINDOW_SIZE)
        .add_feature(lambda data: data.with_columns([
            (pl.col('liquidity_sum') - pl.col('liquidity_sum_lag_1')).alias('delta_liq')
        ]))
        .with_target(TARGET_COLUMN_CLASS)
            .add_transform(add_breakout_ema, target_col=TARGET_COLUMN)
            .done()
    )


def params():

    p = {
        'random_slice_size': [5000],
        'random_slice_min_pct': [0.25],
        'random_slice_max_pct': [0.75],
        'random_seed': [42],
        'bar_type': ['base', 'trade', 'volume', 'liquidity'],
        'trade_threshold': [5000, 10000, 30000, 100000, 500000],
        'volume_threshold': [100, 250, 500, 750, 1000, 5000],
        'liquidity_threshold': [50000, 1000000, 5000000, 50000000, 100000000],
        'objective': ['binary'],
        'metric': ['auc'],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'num_leaves': [15, 31, 63, 127, 255],
        'max_depth': [3, 5, 7, 9, -1],
        'min_data_in_leaf': [20, 50, 100, 200, 500],
        'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
        'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
        'bagging_freq': [0, 1, 5, 10, 20],
        'lambda_l1': [0.0, 0.1, 1.0, 10.0, 100.0],
        'lambda_l2': [0.0, 0.1, 1.0, 10.0, 100.0],
        'feature_pre_filter': ['false'],
    }

    return p


def prep(data, round_params, manifest):

    data_dict = manifest.prepare_data(data, round_params)

    # LightGBM-specific dataset preparation
    data_dict['dtrain'] = lgb.Dataset(data_dict['x_train'], label=data_dict['y_train'].to_numpy())
    data_dict['dval'] = lgb.Dataset(data_dict['x_val'], label=data_dict['y_val'].to_numpy(), reference=data_dict['dtrain'])

    return data_dict


def model(data: dict, round_params):

    round_params = round_params.copy()
    round_params.update({
      'verbose': -1,
    })

    pos_cnt = data['y_train'].sum()
    neg_cnt = len(data['y_train']) - pos_cnt
    round_params["scale_pos_weight"] = round((neg_cnt / pos_cnt) if pos_cnt > 0 else 1.0, 4)

    model = lgb.train(
        params=round_params,
        train_set=data['dtrain'],
        num_boost_round=4000,
        valid_sets=[data['dtrain'], data['dval']],
        valid_names=["train", "valid"],
        callbacks=[early_stopping(stopping_rounds=200, verbose=False),
                   log_evaluation(period=0)])
    
    pred_prob = model.predict(data['x_test'], num_iteration=model.best_iteration)
    pred_bin = (pred_prob >= 0.5).astype(int)

    round_results = binary_metrics(data, pred_bin, pred_prob)
    round_results['_preds'] = pred_prob

    return round_results 
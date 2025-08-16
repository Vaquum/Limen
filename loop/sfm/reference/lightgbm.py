import numpy as np
import polars as pl
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from loop.features.lag_range import lag_range
from loop.utils.slice_time_series import slice_time_series
from loop.utils.splits import split_sequential, split_data_to_prep_output
from loop.utils.add_breakout_ema import add_breakout_ema
from loop.metrics.binary_metrics import binary_metrics

TARGET_COLUMN = 'liquidity_sum'
LOOKBACK_WINDOW_SIZE = 100
PREDICTION_HORIZON = 3
NUM_SLICES = 10
TARGET_COLUMN_CLASS = 'breakout_ema'


def params():

    p = {
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


def prep(data):
    
    all_datetimes = data['datetime'].to_list()

    slices = slice_time_series(data, TARGET_COLUMN, NUM_SLICES)
    df_slice = slices[2]

    df_slice = df_slice.with_columns([
        pl.col('datetime').dt.hour().alias('hour'),
        pl.col('datetime').dt.minute().alias('minute'),
        pl.col('datetime').dt.weekday().alias('weekday'),
        (pl.col('datetime').diff().dt.total_seconds()).fill_null(0).alias("delta_t_s")
    ])
    
    df_slice = df_slice.with_columns([
        pl.col('liquidity_sum').cast(pl.Float32),
        pl.col('hour').cast(pl.UInt8),
        pl.col('minute').cast(pl.UInt8),
        pl.col('weekday').cast(pl.UInt8),
        pl.col('delta_t_s').cast(pl.UInt32)
    ])
    
    df_slice = df_slice.with_columns([
        pl.col('liquidity_sum').shift(1).alias('liq_prev_1'),
        pl.col('liquidity_sum').shift(2).alias('liq_prev_2'),
        pl.col('liquidity_sum').rolling_mean(window_size=3, min_samples=1).alias('liq_rolling_3'),
        pl.col('liquidity_sum').ewm_mean(span=5).alias('liq_ewm_5'),
        pl.col('liquidity_sum').ewm_mean(span=10).alias('liq_ewm_10')
    ])
    
    df_slice = df_slice.with_columns([
      (pl.col('liquidity_sum') - pl.col('liq_prev_1')).alias('delta_liq')
    ])

    feature_cols = ['datetime',
            'liquidity_sum',
            'hour',
            'minute',
            'weekday',
            'delta_t_s',
            'liq_prev_1',
            'liq_prev_2',
            'liq_rolling_3', 
            'liq_ewm_5',
            'liq_ewm_10',
            'delta_liq']
    
    for col in feature_cols:
        df_slice = lag_range(df_slice, col, 1, LOOKBACK_WINDOW_SIZE)

    df_slice = add_breakout_ema(df_slice, TARGET_COLUMN)

    split_data = split_sequential(data=df_slice, ratios=(6, 2, 2))
    
    cols = feature_cols + [TARGET_COLUMN_CLASS]


    data_dict = split_data_to_prep_output(split_data, cols, all_datetimes)

    data_dict['dtrain'] = lgb.Dataset(data_dict['x_train'], label=data_dict['y_train'].to_numpy())
    data_dict['dval'] = lgb.Dataset(data_dict['x_val'], label=data_dict['y_val'].to_numpy(), reference=data_dict['dtrain'])

    return data_dict

def model(data, round_params):

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
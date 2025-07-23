import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from vaquum_tools.slice_time_series import slice_time_series
from vaquum_tools.utils.split_sequential import split_sequential
from vaquum_tools.utils.add_breakout_ema import add_breakout_ema
from vaquum_tools.utils.create_vectorized_sliding_window import create_vectorized_sliding_window

TARGET_COLUMN = 'quote_quantity'
LOOKBACK_WINDOW_SIZE = 100
PREDICTION_HORIZON = 3
NUM_SLICES = 10
TARGET_COLUMN_CLASS = "breakout_ema"


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
        'feature_pre_filter': ['false']
    }

    return p


def prep(data):

    feature_cols = ["quote_quantity",
                    "hour",
                    "minute",
                    "weekday",
                    "delta_t_s",
                    "liq_prev_1",
                    "liq_prev_2",
                    "liq_rolling_3", 
                    "liq_ewm_5",
                    "liq_ewm_10",
                    "delta_liq"]

    data = data.with_columns([
        (pl.col('price') * pl.col('quantity')).alias('quote_quantity')
    ])

    data = (data.group_by("timestamp").agg([
        pl.col("quote_quantity").sum().alias("quote_quantity"),
        pl.col("datetime").first().alias("datetime")
    ]).sort("timestamp"))

    slices = slice_time_series(data, TARGET_COLUMN, NUM_SLICES)

    df_95_perc = slices[2]

    df_95_perc = (df_95_perc.with_columns([
        pl.col("datetime").dt.hour().alias("hour"),
        pl.col("datetime").dt.minute().alias("minute"),
        pl.col("datetime").dt.weekday().alias("weekday"),
       ((pl.col("timestamp").cast(pl.Int64) - pl.col("timestamp").shift(1).cast(pl.Int64)) ).fill_null(pl.duration(seconds = 0)).alias("delta_t_s")
    ]).drop('datetime'))
    
    df_95_perc = df_95_perc.with_columns([
        pl.col("quote_quantity").cast(pl.Float32),
        pl.col("hour").cast(pl.UInt8),
        pl.col("minute").cast(pl.UInt8),
        pl.col("weekday").cast(pl.UInt8),
        pl.col("delta_t_s").cast(pl.UInt32)
    ])
    
    df_95_perc = df_95_perc.with_columns([
        pl.col("quote_quantity").shift(1).alias("liq_prev_1"),
        pl.col("quote_quantity").shift(2).alias("liq_prev_2"),
        pl.col("quote_quantity").rolling_mean(window_size=3, min_samples=1).alias("liq_rolling_3"),
        pl.col("quote_quantity").ewm_mean(span=5).alias("liq_ewm_5"),
        pl.col("quote_quantity").ewm_mean(span=10).alias("liq_ewm_10")
    ])
    
    df_95_perc = df_95_perc.with_columns([
      (pl.col("quote_quantity") - pl.col("liq_prev_1")).alias("delta_liq")
    ])

    df_train, df_val, df_test = split_sequential(data=df_95_perc, ratios=(6, 2, 2))
    
    df_train = add_breakout_ema(df_train, TARGET_COLUMN)
    df_val = add_breakout_ema(df_val, TARGET_COLUMN)
    df_test = add_breakout_ema(df_test, TARGET_COLUMN)

    train_X, train_y = create_vectorized_sliding_window(df_train,
                                 feature_cols,
                                 TARGET_COLUMN_CLASS,
                                 LOOKBACK_WINDOW_SIZE,
                                 PREDICTION_HORIZON)

    val_X, val_y = create_vectorized_sliding_window(df_val,
                             feature_cols,
                             TARGET_COLUMN_CLASS,
                             LOOKBACK_WINDOW_SIZE,
                             PREDICTION_HORIZON)

    test_X, test_y = create_vectorized_sliding_window(df_test,
                               feature_cols,
                               TARGET_COLUMN_CLASS,
                               LOOKBACK_WINDOW_SIZE,
                               PREDICTION_HORIZON)

    train_X = np.nan_to_num(train_X, nan=0.0, copy=False).astype(np.float32, copy=False)
    val_X = np.nan_to_num(val_X, nan=0.0, copy=False).astype(np.float32, copy=False)
    test_X = np.nan_to_num(test_X, nan=0.0, copy=False).astype(np.float32, copy=False)
    
    good_idx = np.where(pd.DataFrame(train_X).nunique().values > 1)[0]
    train_X = train_X[:, good_idx]
    val_X = val_X[:, good_idx]
    test_X = test_X[:, good_idx]
    
    flattened_names_full = [f"{col}_lag{lag}" for lag in range(LOOKBACK_WINDOW_SIZE) for col in feature_cols]
    feature_names = [flattened_names_full[i] for i in good_idx]

    dtrain = lgb.Dataset(train_X, label=train_y, feature_name=feature_names)
    dval = lgb.Dataset(val_X, label=val_y, reference=dtrain, feature_name=feature_names)

    return {'dtrain': dtrain,
            'dval': dval,
            'train_X': train_X,
            'val_X': val_X,
            'test_X': test_X,
            'train_y': train_y,
            'val_y': val_y,
            'test_y': test_y}


def model(data, round_params):

    model = None

    pos_cnt = data['train_y'].sum()
    neg_cnt = len(data['train_y']) - pos_cnt
    round_params["scale_pos_weight"] = round((neg_cnt / pos_cnt) if pos_cnt > 0 else 1.0, 4)

    model = lgb.train(
        params=round_params,
        train_set=data['dtrain'],
        num_boost_round=4000,
        valid_sets=[data['dtrain'], data['dval']],
        valid_names=["train", "valid"],
        callbacks=[early_stopping(stopping_rounds=200, verbose=False),
                   log_evaluation(period=0)])
    
    pred_prob = model.predict(data['test_X'], num_iteration=model.best_iteration)
    pred_bin = (pred_prob >= 0.5).astype(int)

    round_results = {'recall': round(recall_score(data['test_y'], pred_bin), 2),
                     'precision': round(precision_score(data['test_y'], pred_bin), 2),
                     'f1score': round(f1_score(data['test_y'], pred_bin), 2),
                     'auc': round(roc_auc_score(data['test_y'], pred_bin), 2),
                     'accuracy': round(accuracy_score(data['test_y'], pred_bin), 2)}

    return round_results 
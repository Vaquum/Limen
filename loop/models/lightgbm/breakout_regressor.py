'''
SFM Label Model for Breakout regime regressor
'''

import loop
import polars as pl
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import timedelta

from loop.utils.splits import split_sequential
from loop.models.lightgbm.utils  import build_sample_dataset_for_breakout_regressor, extract_xy
from loop.indicators.breakout_features import breakout_features


# Configuration constants
INTERVAL_SEC = 7200  # 2 hour intervals
DATETIME_COL = 'datetime'
NUM_PERMUTATIONS = 48
NUM_ROWS = 15000
PREDICTION_HORIZON = 12  # number of 2h bars to look back → e.g. 36 ≙ 3 days
PRICE_COLUMN='average_price'
EMA_SPAN = 6  # 6 x (12 x 2h kline)
LOOKAHEAD_HOURS = 24  # 24 hour lookahead
LOOKBACK_BARS = 12  # look-back bars (12×2h = 1 day)
TRAIN_SPLIT = 5
VAL_SPLIT = 3
TEST_SPLIT = 2
CONFIDENCE_THRESHOLD = 0.40  # Minimum confidence to make a prediction (otherwise classify as flat)
LONG_COL_PREFIX = 'long_0_'
SHORT_COL_PREFIX = 'short_0_'
BREAKOUT_COL_PREFIX = 'breakout'
BREAKOUT_LONG_COL= f"{BREAKOUT_COL_PREFIX}_long"
BREAKOUT_SHORT_COL=f"{BREAKOUT_COL_PREFIX}_short"
TARGET = BREAKOUT_LONG_COL # or BREAKOUT_SHORT_COL

# All breakout % thresholds we track
DELTAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

def params():
    '''Return hyperparameter search space.'''
    p = {
        'objective': ['regression'],
        'metric': ['mae'],
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
    }
    return p

def prep(data):
    df = build_sample_dataset_for_breakout_regressor(
        data,
        datetime_col=DATETIME_COL,
        target_col=PRICE_COLUMN,
        interval_sec=INTERVAL_SEC,
        lookahead=timedelta(hours=LOOKAHEAD_HOURS),
        ema_span=EMA_SPAN * LOOKBACK_BARS,
        deltas=DELTAS,
        long_col_prefix=LONG_COL_PREFIX,
        short_col_prefix=SHORT_COL_PREFIX,
        shift_bars=-PREDICTION_HORIZON,
        random_slice_size=NUM_ROWS,
        long_target_col=BREAKOUT_LONG_COL,
        short_target_col=BREAKOUT_SHORT_COL,
    )

    # Generate full lagged feature DataFrame
    df_feat = breakout_features(
        df,
        long_col=BREAKOUT_LONG_COL,
        short_col=BREAKOUT_SHORT_COL,
        lookback=LOOKBACK_BARS,
        horizon=PREDICTION_HORIZON,
        target=TARGET
    )

    # Split into train, validate, test dataset
    train, val, test = split_sequential(df_feat, ratios=(TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT))
    dt_test = test[DATETIME_COL].to_list()

    #3 x/y setup
    x_train, y_train = extract_xy(train, TARGET, PREDICTION_HORIZON, LOOKBACK_BARS)
    x_val, y_val = extract_xy(val, TARGET, PREDICTION_HORIZON, LOOKBACK_BARS)
    x_test, y_test = extract_xy(test, TARGET, PREDICTION_HORIZON, LOOKBACK_BARS)

    # debug debug debug
    # build the exact same feat_cols as extract_xy (INCLUDING extra features)
    lag_indices = range(PREDICTION_HORIZON, PREDICTION_HORIZON + LOOKBACK_BARS)

    # lagged flag features
    lag_cols = [f"long_t-{i}" for i in lag_indices] + \
               [f"short_t-{i}" for i in lag_indices]

    # additional features from build_lagged_flags
    # extra_cols = ['long_roll_mean','long_roll_std', 'short_roll_mean', 'short_roll_std', 'roc_long_1', 'roc_short_1']
    extra_cols = df_feat.columns
    feat_cols = list(set(lag_cols + extra_cols))

    assert x_train.shape[1] == len(feat_cols), \
        f"x_train has {x_train.shape[1]} cols but feat_cols is {len(feat_cols)}"

    # Extract datetime for training rows (after dropping nulls)
    dt_train = train[DATETIME_COL].to_list()

    # 4. LightGBM
    dtrain = lgb.Dataset(x_train, label=y_train)
    dval = lgb.Dataset(x_val, label=y_val, reference=dtrain)

    return {
        'dtrain':  dtrain,
        'dval':    dval,
        'x_train': x_train,
        'x_val':   x_val,
        'x_test':  x_test,
        'y_train': y_train,
        'y_val':   y_val,
        'y_test':  y_test,
        'dt_test': dt_test
    }

def model(data, round_params):

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
        callbacks=[lgb.early_stopping(round_params['stopping_round'], verbose=False),
                   lgb.log_evaluation(round_params['logging_step'])])

    # Predict on test set
    y_pred = model.predict(data['x_test'])

    # Metrics
    mae = mean_absolute_error(data['y_test'], y_pred)
    rmse = mean_squared_error(data['y_test'], y_pred, squared=False)
    r2 = r2_score(data['y_test'], y_pred)

    round_results = {
        'models': [model],
        'extras': {'rmse': rmse, 'mae': mae, 'r2': r2}
    }

    return round_results

'''
--- Example usage ---

import loop
from loop.models.lightgbm import breakout_regressor

context_params = {
    'kline_size': [7200],
    'start_date_limit': ['2019-01-01 00:00:00'],
    'n_permutations': [48],
    'random_sample_size': [15000],
    'target': ['breakout_long']
}

context_params_loop = loop.utils.ParamSpace(context_params)
p = context_params_loop.generate()

historical = loop.HistoricalData()
historical.get_historical_klines(
    kline_size=p['kline_size'],
    start_date_limit=p['start_date_limit']
)

uel = loop.UniversalExperimentLoop(historical.data, breakout_regressor)
uel.run(
    experiment_name=f"test1_label_model_regressor_random_{p['random_sample_size']}_2H_PREDICT_{p['target']}",
    n_permutations=p['n_permutations'],
    prep_each_round=False,
    random_search=True,
)

print(uel.log_df.head())
'''

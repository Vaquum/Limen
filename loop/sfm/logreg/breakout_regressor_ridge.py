'SFM Label Model for Breakout regressor using Ridge Regression'

from sklearn.linear_model import Ridge
from datetime import timedelta

from loop.utils.splits import split_sequential, split_data_to_prep_output
from loop.sfm.logreg.utils import build_sample_dataset_for_breakout_regressor
from loop.features.breakout_features import breakout_features
from loop.transforms.logreg_transform import LogRegTransform
from loop.metrics.continuous_metrics import continuous_metrics

INTERVAL_SEC = 7200
DATETIME_COL = 'datetime'
NUM_PERMUTATIONS = 48
NUM_ROWS = 15000
PREDICTION_HORIZON = 12
PRICE_COLUMN = 'average_price'
EMA_SPAN = 6
LOOKAHEAD_HOURS = 24
LOOKBACK_BARS = 12
TRAIN_SPLIT = 5
VAL_SPLIT = 3
TEST_SPLIT = 2
LONG_COL_PREFIX = 'long_0_'
SHORT_COL_PREFIX = 'short_0_'
BREAKOUT_COL_PREFIX = 'breakout'
BREAKOUT_LONG_COL = f"{BREAKOUT_COL_PREFIX}_long"
BREAKOUT_SHORT_COL = f"{BREAKOUT_COL_PREFIX}_short"
TARGET = BREAKOUT_LONG_COL
DELTAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]


def params():

    p = {
        'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
        'max_iter': [100, 500, 1000, 2000, 5000],
        'tol': [0.0001, 0.001, 0.01, 0.1],
        'fit_intercept': [True, False],
        'random_state': [42],
    }
    return p


def prep(data):

    all_datetimes = data['datetime'].to_list()

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

    df = breakout_features(
        df,
        long_col=BREAKOUT_LONG_COL,
        short_col=BREAKOUT_SHORT_COL,
        lookback=LOOKBACK_BARS,
        horizon=PREDICTION_HORIZON,
        target=TARGET
    )

    feature_cols = [col for col in df.columns if col != TARGET and not col.startswith('breakout_')]
    df = df.select(feature_cols + [TARGET])

    split_data = split_sequential(data=df, ratios=(TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT))

    cols = df.columns

    data_dict = split_data_to_prep_output(split_data, cols, all_datetimes)

    scaler = LogRegTransform(data_dict['x_train'])
    for col in data_dict.keys():
        if col.startswith('x_'):
            data_dict[col] = scaler.transform(data_dict[col])

    data_dict['_scaler'] = scaler
    data_dict['_feature_names'] = cols

    return data_dict


def model(data, round_params):

    params = round_params.copy()

    ridge_params = {
        'alpha': params['alpha'],
        'solver': params['solver'],
        'max_iter': params['max_iter'],
        'tol': params['tol'],
        'fit_intercept': params['fit_intercept'],
        'random_state': params['random_state'],
    }

    if params['solver'] == 'lbfgs':
        ridge_params['positive'] = True

    ridge = Ridge(**ridge_params)

    ridge.fit(data['x_train'], data['y_train'])

    preds = ridge.predict(data['x_test'])
    
    round_results = continuous_metrics(data, preds)
        
    return round_results

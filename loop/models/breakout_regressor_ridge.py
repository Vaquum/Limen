'''
SFM Label Model for Breakout regressor using Ridge Regression
'''

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import timedelta

from loop.utils.splits import split_sequential
from loop.models.lightgbm.utils import build_sample_dataset_for_breakout_regressor, extract_xy_polars
from loop.indicators.breakout_features import breakout_features
from loop.transforms.logreg_transform import LogRegTransform


# Configuration constants (same as LightGBM version)
INTERVAL_SEC = 7200  # 2 hour intervals
DATETIME_COL = 'datetime'
NUM_PERMUTATIONS = 48
NUM_ROWS = 15000
PREDICTION_HORIZON = 12  # number of 2h bars to look back → e.g. 36 ≙ 3 days
PRICE_COLUMN = 'average_price'
EMA_SPAN = 6  # 6 x (12 x 2h kline)
LOOKAHEAD_HOURS = 24  # 24 hour lookahead
LOOKBACK_BARS = 12  # look-back bars (12×2h = 1 day)
TRAIN_SPLIT = 5
VAL_SPLIT = 3
TEST_SPLIT = 2
LONG_COL_PREFIX = 'long_0_'
SHORT_COL_PREFIX = 'short_0_'
BREAKOUT_COL_PREFIX = 'breakout'
BREAKOUT_LONG_COL = f"{BREAKOUT_COL_PREFIX}_long"
BREAKOUT_SHORT_COL = f"{BREAKOUT_COL_PREFIX}_short"
TARGET = BREAKOUT_LONG_COL  # or BREAKOUT_SHORT_COL

# All breakout % thresholds we track
DELTAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]


def params():
    '''Return hyperparameter search space for Ridge Regression.'''
    p = {
        # Ridge specific parameters
        'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        'max_iter': [100, 500, 1000, 2000, 5000],
        'tol': [0.0001, 0.001, 0.01, 0.1],
        'fit_intercept': [True, False],
        'random_state': [42],
        # For sag/saga solvers
        'positive': [True, False],  # Forces coefficients to be positive
    }
    return p


def prep(data):
    '''Prepare data for training - follows same structure as LightGBM version.'''
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

    # Extract x/y as Polars DataFrames
    x_train_df, y_train = extract_xy_polars(train, TARGET, PREDICTION_HORIZON, LOOKBACK_BARS)
    x_val_df, y_val = extract_xy_polars(val, TARGET, PREDICTION_HORIZON, LOOKBACK_BARS)
    x_test_df, y_test = extract_xy_polars(test, TARGET, PREDICTION_HORIZON, LOOKBACK_BARS)

    # Get feature columns for tracking (same logic as in extract_xy)
    lag_indices = range(PREDICTION_HORIZON, PREDICTION_HORIZON + LOOKBACK_BARS)
    lag_cols = [f"long_t-{i}" for i in lag_indices] + \
               [f"short_t-{i}" for i in lag_indices]
    extra_cols = df_feat.columns
    feat_cols = list(set(lag_cols + extra_cols))

    # Scale features using LogRegTransform
    scaler = LogRegTransform(x_train_df)
    x_train_scaled = scaler.transform(x_train_df).to_numpy()
    x_val_scaled = scaler.transform(x_val_df).to_numpy()
    x_test_scaled = scaler.transform(x_test_df).to_numpy()

    # Extract datetime for training rows
    dt_train = train[DATETIME_COL].to_list()

    return {
        'x_train': x_train_scaled,
        'x_val': x_val_scaled,
        'x_test': x_test_scaled,
        'y_train': y_train.to_numpy(),
        'y_val': y_val.to_numpy(),
        'y_test': y_test.to_numpy(),
        'dt_test': dt_test,
        'dt_train': dt_train,
        '_scaler': scaler,
        '_feature_names': feat_cols[:x_train_df.shape[1]],
    }


def model(data, round_params):
    '''Train Ridge Regression model and evaluate.'''
    
    # Handle solver compatibility
    params = round_params.copy()
    
    # Handle positive parameter compatibility
    # Actually, most Ridge solvers don't support positive - only 'lbfgs' does
    if params['solver'] == 'lbfgs':
        # lbfgs supports positive, keep the parameter
        pass
    else:
        # All other solvers don't support positive
        params['positive'] = False
    
    # Create and train the model
    ridge = Ridge(
        alpha=params['alpha'],
        solver=params['solver'],
        max_iter=params['max_iter'],
        tol=params['tol'],
        fit_intercept=params['fit_intercept'],
        random_state=params['random_state'],
        positive=params.get('positive', False) if 'positive' in params else False,
    )
    
    # Train the model
    ridge.fit(data['x_train'], data['y_train'])
    
    # Predict on test set
    y_pred = ridge.predict(data['x_test'])
    
    # Calculate metrics
    mae = mean_absolute_error(data['y_test'], y_pred)
    rmse = np.sqrt(mean_squared_error(data['y_test'], y_pred))
    r2 = r2_score(data['y_test'], y_pred)
    
    # Calculate validation metrics for model selection
    y_val_pred = ridge.predict(data['x_val'])
    val_mae = mean_absolute_error(data['y_val'], y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(data['y_val'], y_val_pred))
    val_r2 = r2_score(data['y_val'], y_val_pred)
    
    round_results = {
        'models': [ridge],
        'extras': {
            'rmse': round(rmse, 4),
            'mae': round(mae, 4),
            'r2': round(r2, 4),
            'val_rmse': round(val_rmse, 4),
            'val_mae': round(val_mae, 4),
            'val_r2': round(val_r2, 4),
        }
    }
    
    return round_results


'''
--- Example usage ---

import loop
from loop.models import breakout_regressor_ridge

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

uel = loop.UniversalExperimentLoop(historical.data, breakout_regressor_ridge)
uel.run(
    experiment_name=f"test1_label_model_ridge_regressor_random_{p['random_sample_size']}_2H_PREDICT_{p['target']}",
    n_permutations=p['n_permutations'],
    prep_each_round=False,
    random_search=True,
)

print(uel.log_df.head())
'''
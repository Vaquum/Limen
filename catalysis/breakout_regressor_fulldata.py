"""
Modified Breakout Regressor for Full Dataset Testing

This is a fork of loop.sfm.logreg.breakout_regressor_ridge
modified to use the full dataset instead of random_slice for proper testing.
"""

from datetime import timedelta
from loop.sfm.logreg.utils.breakout_regressor import build_breakout_regressor_base_features
from loop.features.breakout_features import breakout_features
from loop.transforms.logreg_transform import LogRegTransform
from loop.manifest import Manifest
from loop.sfm.model import ridge_regression

INTERVAL_SEC = 7200  # 2 hours - matches original
DATETIME_COL = 'datetime'
PREDICTION_HORIZON = 12
PRICE_COLUMN = 'average_price'  # Computed from aggregation
EMA_SPAN = 6
LOOKAHEAD_HOURS = 24  # 24 hours lookahead - matches original
LOOKBACK_BARS = 12
TRAIN_SPLIT = 7  # 70% train
VAL_SPLIT = 1     # 10% val
TEST_SPLIT = 2    # 20% test
LONG_COL_PREFIX = 'long_0_'
SHORT_COL_PREFIX = 'short_0_'
BREAKOUT_COL_PREFIX = 'breakout'
BREAKOUT_LONG_COL = f"{BREAKOUT_COL_PREFIX}_long"
BREAKOUT_SHORT_COL = f"{BREAKOUT_COL_PREFIX}_short"
TARGET = BREAKOUT_LONG_COL
DELTAS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]


def manifest():

    return (Manifest()
        # No random_slice - use full dataset
        .set_split_config(TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)
        .add_feature(build_breakout_regressor_base_features,
            datetime_col=DATETIME_COL,
            target_col=PRICE_COLUMN,
            interval_sec=INTERVAL_SEC,
            lookahead=timedelta(hours=LOOKAHEAD_HOURS),
            ema_span=EMA_SPAN * LOOKBACK_BARS,
            deltas=DELTAS,
            long_col_prefix=LONG_COL_PREFIX,
            short_col_prefix=SHORT_COL_PREFIX,
            shift_bars=-PREDICTION_HORIZON,
            long_target_col=BREAKOUT_LONG_COL,
            short_target_col=BREAKOUT_SHORT_COL)
        .add_feature(breakout_features,
            long_col=BREAKOUT_LONG_COL,
            short_col=BREAKOUT_SHORT_COL,
            lookback=LOOKBACK_BARS,
            horizon=PREDICTION_HORIZON,
            target=TARGET)
        .with_target(TARGET)
            .add_transform(lambda data: data.select([
                c for c in data.columns
                if c == TARGET or (not c.startswith('breakout_'))
            ]))
            .done()
        .set_scaler(LogRegTransform)
        .with_model(ridge_regression)
    )


def params():

    return {
        'alpha': [1.0],  # Fixed alpha for testing
        'solver': ['auto'],
        'max_iter': [1000],
        'tol': [0.0001],
        'fit_intercept': [True],
        'random_state': [42],
    }

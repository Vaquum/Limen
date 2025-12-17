import polars as pl

from datetime import timedelta

from loop.sfm.logreg.utils.regime_multiclass import build_regime_base_features
from loop.sfm.logreg.utils.regime_multiclass import add_regime_lag_features
from loop.utils.random_slice import random_slice
from loop.transforms.logreg_transform import LogRegTransform
from loop.manifest import Manifest
from loop.sfm.model import logreg_multiclass

BREAKOUT_PERCENTAGE = 5
LONG_COL = f'long_0_0{BREAKOUT_PERCENTAGE}'
SHORT_COL = f'short_0_0{BREAKOUT_PERCENTAGE}'
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

    return {
        'random_slice_size': [NUM_ROWS],
        'random_slice_min_pct': [0.25],
        'random_slice_max_pct': [0.75],
        'random_seed': [42],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
        'max_iter': [100, 200, 500, 1000, 2000],
        'tol': [0.0001, 0.001, 0.01, 0.1],
        'class_weight': ['None', 'balanced'],
        'l1_ratio': [0.15, 0.5, 0.85],
        'fit_intercept': [True],
        'random_state': [42],
    }


def manifest():

    return (Manifest()
        .set_data_source(
            method='get_spot_klines',
            params={'kline_size': 3600, 'start_date_limit': '2023-01-01'}
        )
        .set_test_data_source(method='get_klines_data_large')
        .set_pre_split_data_selector(
            random_slice,
            rows='random_slice_size',
            safe_range_low='random_slice_min_pct',
            safe_range_high='random_slice_max_pct',
            seed='random_seed'
        )
        .set_split_config(TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)
        .add_feature(build_regime_base_features,
            datetime_col='datetime',
            target_col=TARGET_COLUMN,
            interval_sec=INTERVAL_SEC,
            lookahead=timedelta(hours=LOOKAHEAD_HOURS),
            ema_span=EMA_SPAN * LOOKBACK_BARS,
            deltas=DELTAS,
            long_col=LONG_COL,
            short_col=SHORT_COL,
            leakage_shift_bars=LEAKAGE_SHIFT)
        .add_feature(add_regime_lag_features,
            lookback_bars=LOOKBACK_BARS)
        .with_target('regime')
            .add_transform(lambda data: data.with_columns(
                pl.when((pl.col(LONG_COL) == 1) & (pl.col(SHORT_COL) == 0)).then(1)
                  .when((pl.col(SHORT_COL) == 1) & (pl.col(LONG_COL) == 0)).then(2)
                  .otherwise(0)
                  .alias('regime')
            ))
            .add_transform(lambda data: data.select([
                c for c in data.columns
                if not c.startswith(('long_0_', 'short_0_'))
            ]))
            .done()
        .set_scaler(LogRegTransform)
        .with_model(logreg_multiclass)
    )

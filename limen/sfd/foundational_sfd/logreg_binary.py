from limen.calibration import grid_threshold_optimizer
from limen.calibration import sklearn_probability_calibrator
from limen.data import HistoricalData
from limen.experiment import MLManifest
from limen.experiment import Manifest
from limen.features import fractional_diff
from limen.features import kline_imbalance
from limen.features import vwap
from limen.indicators import atr
from limen.indicators import ppo
from limen.indicators import roc
from limen.indicators import wilder_rsi
from limen.metrics.balanced_metric import balanced_metric
from limen.sfd.reference_architecture.logreg_binary import logreg_binary
from limen.targets import QuantileBinaryTarget


def params() -> dict:

    return {
        # data prep parameters
        'frac_diff_d': [0.0],
        'shift': [-1, -2, -3, -4, -5],
        'q': [0.35, 0.38, 0.41, 0.44, 0.47, 0.50, 0.53],
        'roc_period': [1, 4, 12, 24, 144],
        # view perturbation
        'scaler_type': ['logreg', 'robust', 'rank_gauss'],
        # feature perturbation
        'feature_groups': ['all', 'momentum', 'momentum|volatility'],
        # calibration parameters
        'use_calibration': [True, False],
        'use_threshold': [True, False],
        'cal_method': ['isotonic', 'sigmoid'],
        'threshold_min': [0.20, 0.25, 0.30],
        'threshold_max': [0.60, 0.65, 0.70],
        'threshold_step': [0.05, 0.10],
        # classifier parameters
        'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'newton-cholesky'],
        'penalty': ['l2'],
        'dual': [False],
        'tol': [0.001, 0.01, 0.03, 0.1, 0.3],
        'C': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'fit_intercept': [True, False],
        'intercept_scaling': [1.0],
        'class_weight': [0.45, 0.55, 0.65, 0.75, 0.85],
        'random_state': [42],
        'max_iter': [30, 60, 90, 120, 180, 240],
        'multi_class': ['deprecated'],
        'verbose': [0],
        'warm_start': [False],
        'n_jobs': [None],
        'l1_ratio': [None],
    }


def manifest() -> Manifest:

    return (MLManifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'start_date_limit': '2025-01-01'}
        )
        .set_test_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 7200, 'row_count_limit': 5000}
        )
        .set_split_config(8, 1, 2)

        .add_feature(fractional_diff, d='frac_diff_d', cols=['close'])

        .add_indicator(roc, period='roc_period', group='momentum')
        .add_indicator(atr, period=14, group='volatility')
        .add_indicator(ppo, group='momentum')
        .add_indicator(wilder_rsi, group='momentum')

        .add_feature(vwap, group='volume')
        .add_feature(kline_imbalance, group='volume')

        .with_target_label(
            'quantile_flag',
            QuantileBinaryTarget,
            fit_params={'source_column': 'roc_{roc_period}', 'quantile': 'q'},
            transform_params={'shift': 'shift'},
        )

        .set_scaler_from_params('scaler_type')
        .set_feature_ablation()
        .set_strict_mode(True)

        .with_reference_architecture(logreg_binary)

        .with_calibration()
        .probability_calibration(func=sklearn_probability_calibrator, method='cal_method')
        .threshold_function(func=grid_threshold_optimizer, metric=balanced_metric,
                            threshold_min='threshold_min',
                            threshold_max='threshold_max',
                            threshold_step='threshold_step')
        .done()
    )

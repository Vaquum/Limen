from limen.calibration import grid_threshold_optimizer
from limen.calibration import sklearn_probability_calibrator
from limen.data import HistoricalData
from limen.experiment import Manifest
from limen.experiment import MLManifest
from limen.features import cyclical_time_features
from limen.features import distance_from_high
from limen.features import distance_from_low
from limen.features import parkinson_volatility
from limen.features import price_lines
from limen.features import price_range_position
from limen.features import quantile_price_lines
from limen.features import volatility_ratio
from limen.indicators import roc
from limen.metrics.balanced_metric import balanced_metric
from limen.sfd.reference_architecture.lightgbm_binary import lightgbm_binary
from limen.targets import TradelineLongBinaryTarget


def params() -> dict:

    return {
        # line geometry
        'max_duration_hours': [24, 48, 72, 96],
        'min_height_pct': [0.001, 0.002, 0.003, 0.004, 0.005],
        'quantile_threshold': [0.60, 0.70, 0.75, 0.80, 0.85],
        # label
        'long_threshold_percentile': [65, 75, 85],
        'lookahead_hours': [24, 48, 72],
        'confirmation_hours': [24],
        # context features
        'ret_period': [1, 6, 12, 24, 48],
        # view perturbation
        'scaler_type': ['robust', 'rank_gauss', 'logreg'],
        # feature perturbation
        'feature_groups': ['all', 'lines', 'lines|momentum', 'lines|volatility', 'momentum|volatility|position'],
        'feature_drop_count': [0],
        'feature_drop_seed': [42],
        # calibration
        'use_calibration': [True, False],
        'use_threshold': [True, False],
        'cal_method': ['isotonic', 'sigmoid'],
        'threshold_min': [0.30, 0.45],
        'threshold_max': [0.65, 0.80],
        'threshold_step': [0.05],
        # backtest economics
        'fee_bps': [1.0, 5.0, 10.0],
        'slip_bps': [2.0, 5.0],
        # classifier
        'objective': [None],
        'boosting_type': ['gbdt'],
        'num_leaves': [31, 63, 127],
        'max_depth': [-1],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 300, 500],
        'subsample_for_bin': [200000],
        'min_split_gain': [0.0],
        'min_child_weight': [0.001],
        'min_child_samples': [10, 20],
        'subsample': [0.8, 0.9, 1.0],
        'subsample_freq': [0, 5],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0.0, 0.1],
        'reg_lambda': [0.0, 0.1],
        'class_weight': ['balanced', None],
        'random_state': [42],
        'n_jobs': [None],
        'importance_type': ['split'],
        'early_stopping_rounds': [50],
        'deterministic': [True],
        'force_row_wise': [True],
        'verbosity': [-1],
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

        .add_feature(price_lines,
                     max_duration_hours='max_duration_hours',
                     min_height_pct='min_height_pct',
                     include_research_only=False,
                     group='lines')
        .add_feature(quantile_price_lines,
                     max_duration_hours='max_duration_hours',
                     min_height_pct='min_height_pct',
                     quantile_threshold='quantile_threshold',
                     include_research_only=False,
                     group='lines')

        .add_indicator(roc, period='ret_period', group='momentum')

        .add_feature(distance_from_high, period=24, group='position')
        .add_feature(distance_from_low, period=24, group='position')
        .add_feature(price_range_position, period=24, group='position')

        .add_feature(parkinson_volatility, window=24, group='volatility')
        .add_feature(volatility_ratio, short_window=6, long_window=24, group='volatility')

        .add_feature(cyclical_time_features)

        .with_target_label(
            'tradeline_long',
            TradelineLongBinaryTarget,
            fit_params={
                'max_duration_hours': 'max_duration_hours',
                'min_height_pct': 'min_height_pct',
                'long_threshold_percentile': 'long_threshold_percentile',
            },
            transform_params={
                'lookahead_hours': 'lookahead_hours',
                'confirmation_hours': 'confirmation_hours',
            },
        )

        .set_scaler_from_params('scaler_type')
        .set_feature_ablation()
        .set_strict_mode(True)

        .with_reference_architecture(lightgbm_binary)

        .with_calibration()
        .probability_calibration(func=sklearn_probability_calibrator, method='cal_method')
        .threshold_function(func=grid_threshold_optimizer, metric=balanced_metric,
                            threshold_min='threshold_min',
                            threshold_max='threshold_max',
                            threshold_step='threshold_step')
        .done()

        .set_backtest_config(fee_bps='fee_bps', slip_bps='slip_bps')
    )

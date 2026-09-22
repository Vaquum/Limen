from typing import cast

from limen.data import HistoricalData
from limen.experiment import MLManifest
from limen.features.lagged_features import lag_range
from limen.indicators.window_return import window_return
from limen.scalers import RobustScaler
from limen.sfd.reference_architecture.ridge_regressor import ridge_regressor
from limen.targets import NextReturnTarget

__all__ = ['manifest', 'params']


def params() -> dict[str, list[object]]:
    """Return a small search space over native sklearn Ridge parameters."""
    return {
        'alpha': [0.1, 1.0, 10.0],
        'fit_intercept': [True, False],
        'copy_X': [True],
        'max_iter': [None],
        'tol': [1e-4],
        'solver': ['auto'],
        'positive': [False],
        'random_state': [42],
    }


def manifest() -> MLManifest:
    """Build a next-return SFD with lagged returns and train-fitted scaling."""
    base = (
        MLManifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'start_date_limit': '2025-01-01'},
        )
        .set_test_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 7200, 'row_count_limit': 5000},
        )
        .set_split_config(8, 1, 2)
        .add_indicator(window_return, period=1)
        .add_feature(lag_range, col='ret_1', start=1, end=5)
        .with_target_label(
            'next_return', NextReturnTarget,
            transform_params={'periods': 1, 'scale': 100.0},
        )
    )
    return (
        cast(MLManifest, base)
        .set_scaler(RobustScaler)
        .set_strict_mode(True)
        .with_reference_architecture(ridge_regressor)
    )

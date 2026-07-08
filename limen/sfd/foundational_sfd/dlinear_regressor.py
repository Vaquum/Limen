from typing import cast

from limen.data import HistoricalData
from limen.experiment import Manifest
from limen.experiment import MLManifest
from limen.features.lagged_features import lag_range
from limen.indicators.window_return import window_return
from limen.sfd.reference_architecture.dlinear_regressor import dlinear_regressor
from limen.targets import NextReturnTarget


def params():
    return {
        'lookback_end': [23, 47, 95, 167],
        'kernel_size': [13, 25],
        'alpha': [1.0, 10.0, 100.0],
        'horizon': [1, 4, 24],
    }


def manifest() -> Manifest:

    base = (
        MLManifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'start_date_limit': '2025-01-01'}
        )
        .set_test_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 7200, 'row_count_limit': 5000}
        )
        .set_split_config(8, 1, 2)
        .add_indicator(window_return, period=1)
        .add_feature(lag_range, col='ret_1', start=0, end='lookback_end')
        .with_target_label('next_return', NextReturnTarget, transform_params={'periods': 'horizon'})
    )
    return (
        cast(MLManifest, base)
        .set_strict_mode(True)
        .with_reference_architecture(dlinear_regressor)
    )

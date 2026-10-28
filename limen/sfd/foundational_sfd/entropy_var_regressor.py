from typing import cast

from limen.data import HistoricalData
from limen.experiment import MLManifest
from limen.features.lagged_features import lag_range
from limen.indicators.window_return import window_return
from limen.sfd.reference_architecture.entropy_var_regressor import entropy_var_regressor
from limen.targets import NextReturnTarget

__all__ = ['manifest', 'params']


def params() -> dict[str, list[object]]:
    '''Search entropy lookback, return horizon and native quantile controls.'''
    return {'lookback_end': [23, 47], 'horizon': [24], 'quantile': [0.01, 0.05],
            'alpha': [0.0], 'fit_intercept': [True], 'solver': ['highs'], 'solver_options': [None]}


def manifest() -> MLManifest:
    '''Fit completed hourly return-sign entropy to the following daily return quantile.'''
    base = (MLManifest()
        .set_data_source(method=HistoricalData.get_spot_klines,
                         params={'kline_size': 3600, 'start_date_limit': '2024-01-01', 'row_count_limit': 2048})
        .set_split_config(8, 1, 2)
        .add_indicator(window_return, period=1)
        .add_feature(lag_range, col='ret_1', start=0, end='lookback_end')
        .with_target_label('next_return', NextReturnTarget,
                           transform_params={'periods': 'horizon', 'scale': 100.0}))
    return cast(MLManifest, base).set_strict_mode(True).with_reference_architecture(entropy_var_regressor)

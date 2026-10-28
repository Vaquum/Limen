from typing import cast

from limen.data import HistoricalData
from limen.experiment import MLManifest
from limen.features.lagged_features import lag_range
from limen.indicators.window_return import window_return
from limen.scalers import RobustScaler
from limen.sfd.reference_architecture.residual_regressor import residual_regressor
from limen.targets import NextReturnTarget

__all__ = ['manifest', 'params']


def params() -> dict[str, list[object]]:
    return {'ridge_params': [{'alpha': .1, 'solver': 'svd'}, {'alpha': 1., 'solver': 'svd'}],
            'xgboost_params': [{'n_estimators': 32}, {'n_estimators': 64}],
            'correction_scale': [.5, 1.]}


def manifest() -> MLManifest:
    base = (MLManifest().set_data_source(HistoricalData.get_spot_klines,
                {'kline_size': 3600, 'start_date_limit': '2025-01-01', 'row_count_limit': 1024})
            .set_split_config(8, 1, 2).add_indicator(window_return, period=1)
            .add_feature(lag_range, col='ret_1', start=1, end=5)
            .with_target_label('next_return', NextReturnTarget,
                               transform_params={'periods': 1, 'scale': 100.}))
    return (cast(MLManifest, base).set_scaler(RobustScaler).set_strict_mode(True)
            .with_reference_architecture(residual_regressor))

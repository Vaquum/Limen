import hashlib
import json
from typing import cast

import numpy as np
import numpy.typing as npt
import polars as pl
from typing_extensions import override

from limen.metrics.continuous_metrics import continuous_metrics
from limen.sfd.reference_architecture.base import ReferenceModel
from limen.sfd.reference_architecture.ridge_regressor import RidgeRegressor
from limen.sfd.reference_architecture.xgboost_regressor import XGBoostRegressor

__all__ = ['ResidualRegressor', 'residual_regressor']
_FloatArray = npt.NDArray[np.float64]


def _fingerprint(value: object) -> str:
    array = np.asarray(value, dtype='<f8')
    columns = value.columns if isinstance(value, pl.DataFrame) else None
    header = json.dumps([array.shape, columns], separators=(',', ':')).encode()
    return hashlib.sha256(header + array.tobytes(order='C')).hexdigest()


def _residual(primary: RidgeRegressor, inputs: object, observed: object) -> _FloatArray:
    truth = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(primary.predict({'x_test': inputs})['_preds'], dtype=np.float64)
    if truth.ndim != 1 or truth.shape != predicted.shape or not len(truth) or not np.isfinite(truth).all():
        raise ValueError('ResidualRegressor requires aligned finite observed returns')
    result = truth - predicted
    if not np.isfinite(result).all():
        raise ValueError('ResidualRegressor produced nonfinite residuals')
    return result


class ResidualRegressor(ReferenceModel):
    '''Ridge return prediction plus an XGBoost correction fitted to observed errors.'''
    deterministic, inference_output = False, ('continuous', False)

    def __init__(self) -> None:
        super().__init__()
        self.primary: RidgeRegressor | None = None
        self.correction: XGBoostRegressor | None = None
        self._scale = 1.0
        self._provenance: dict[str, object] = {}

    @override
    def train(self, data: dict[str, object], *, ridge_params: dict[str, object] | None = None,
              xgboost_params: dict[str, object] | None = None,
              correction_scale: float = 1.0, **params: object) -> 'ResidualRegressor':
        if params:
            raise TypeError('ResidualRegressor accepts only ridge_params, xgboost_params and correction_scale')
        if not np.isfinite(correction_scale):
            raise ValueError('ResidualRegressor correction_scale must be finite')
        settings = {'n_estimators': 32, 'max_depth': 2, 'learning_rate': .1,
                    'n_jobs': 1, 'tree_method': 'hist', 'device': 'cpu',
                    'random_state': 42, 'early_stopping_rounds': None} | (xgboost_params or {})
        if settings.get('objective', 'reg:squarederror') != 'reg:squarederror':
            raise ValueError('ResidualRegressor requires the reg:squarederror correction objective')
        self.primary, self.correction = None, None
        primary = RidgeRegressor().train({'x_train': data['x_train'], 'y_train': data['y_train']},
                                        **({'alpha': 1.0, 'solver': 'svd'} | (ridge_params or {})))
        residual = _residual(primary, data['x_train'], data['y_train'])
        linked = {'x_train': data['x_train'], 'y_train': residual}
        if settings['early_stopping_rounds'] is not None:
            if 'x_val' not in data or 'y_val' not in data:
                raise ValueError('ResidualRegressor early stopping requires observed validation data')
            linked.update(x_val=data['x_val'], y_val=_residual(primary, data['x_val'], data['y_val']))
        correction = XGBoostRegressor().train(linked, **settings)
        self.primary, self.correction, self._scale = primary, correction, correction_scale
        self._provenance = {'linkage_kind': 'residual',
            'linkage_primary': 'limen.sfd.reference_architecture.RidgeRegressor',
            'linkage_correction': 'limen.sfd.reference_architecture.XGBoostRegressor',
            'linkage_training_rows': len(residual), 'linkage_features_sha256': _fingerprint(data['x_train']),
            'linkage_observed_sha256': _fingerprint(data['y_train']), 'linkage_residual_sha256': _fingerprint(residual)}
        return self

    @override
    def predict(self, data: dict[str, object]) -> dict[str, _FloatArray]:
        if self.primary is None or self.correction is None:
            raise ValueError('ResidualRegressor is not fitted')
        primary = np.asarray(self.primary.predict(data)['_preds'], dtype=np.float64)
        correction = np.asarray(self.correction.predict(data)['_preds'], dtype=np.float64)
        if primary.ndim != 1 or primary.shape != correction.shape:
            raise ValueError('ResidualRegressor requires aligned component predictions')
        prediction = primary + self._scale * correction
        if not np.isfinite(prediction).all():
            raise ValueError('ResidualRegressor produced nonfinite predictions')
        return {'_preds': prediction}

    @override
    def evaluate(self, data: dict[str, object], inline_metrics: bool = True) -> dict[str, object]:
        del inline_metrics
        prediction = self.predict(data)['_preds']
        return cast(dict[str, object], continuous_metrics(data, prediction)) | self._provenance | {'_preds': prediction}


def residual_regressor(data: dict[str, object], ridge_params: dict[str, object] | None = None,
                       xgboost_params: dict[str, object] | None = None,
                       correction_scale: float = 1.0) -> dict[str, object]:
    model = ResidualRegressor().train(data, ridge_params=ridge_params,
                                    xgboost_params=xgboost_params, correction_scale=correction_scale)
    return model.evaluate(data) | {'_model': model}

from collections.abc import Callable
import re
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.linear_model import QuantileRegressor
from typing_extensions import override

from limen.metrics.continuous_metrics import continuous_metrics
from limen.sfd.reference_architecture.base import ReferenceModel

__all__ = ['EntropyVaRRegressor', 'entropy_var_regressor']
_FloatArray = npt.NDArray[np.float64]
_DIMENSIONS = 2
_MEDIAN = 0.5


class _QuantileModel(Protocol):
    quantile: float
    def fit(self, X: _FloatArray, y: _FloatArray) -> object: ...
    def predict(self, X: _FloatArray) -> _FloatArray: ...


class EntropyVaRRegressor(ReferenceModel):
    '''Predict signed lower return quantiles from completed raw return-sign entropy.'''
    deterministic, inference_output = False, ('continuous', False)

    def __init__(self) -> None:
        super().__init__()
        self.model: _QuantileModel | None = None
        self._columns: list[str] = []
        self._indices: list[int] = []
        self._width = 0

    def _entropy(self, inputs: object) -> _FloatArray:
        if isinstance(inputs, pl.DataFrame):
            values = inputs.select(self._columns).to_numpy()
        else:
            matrix = np.asarray(inputs, dtype=np.float64)
            if matrix.ndim != _DIMENSIONS or matrix.shape[1] != self._width:
                raise ValueError('EntropyVaRRegressor inference requires the fitted feature order and width')
            values = matrix[:, self._indices]
        windows = np.asarray(values, dtype=np.float64)
        if not np.isfinite(windows).all():
            raise ValueError('EntropyVaRRegressor requires finite raw return windows')
        p = np.mean(windows < 0, axis=1)
        entropy = np.zeros(len(p), dtype=np.float64)
        interior = (p > 0) & (p < 1)
        fraction = p[interior]
        entropy[interior] = -fraction * np.log2(fraction) - (1 - fraction) * np.log2(1 - fraction)
        return entropy[:, None]

    @override
    def train(self, data: dict[str, object], quantile: float = 0.05, **params: object) -> 'EntropyVaRRegressor':
        if not 0 < quantile < _MEDIAN:
            raise ValueError('EntropyVaRRegressor quantile must be strictly between 0 and 0.5')
        if any(data.get(key) is not None for key in ('_scaler', '_pca', '_trade_policy')):
            raise ValueError('EntropyVaRRegressor requires raw returns without scaler, PCA or execution policy')
        inputs = data['x_train']
        if not isinstance(inputs, pl.DataFrame):
            raise TypeError('EntropyVaRRegressor training requires named raw ret_1_lag_N columns')
        matches = sorted((int(match[1]), column) for column in inputs.columns
                         if (match := re.fullmatch(r'ret_1_lag_(\d+)', column)))
        if not matches or [lag for lag, _ in matches] != list(range(len(matches))):
            raise ValueError('EntropyVaRRegressor requires contiguous ret_1_lag_0..N columns')
        self._columns = [column for _, column in matches]
        self._indices = [inputs.columns.index(column) for column in self._columns]
        self._width = inputs.width
        features = self._entropy(inputs)
        target = np.asarray(data['y_train'], dtype=np.float64)
        if target.ndim != 1 or len(target) != len(features) or not len(target) or not np.isfinite(target).all():
            raise ValueError('EntropyVaRRegressor requires aligned finite observed training returns')
        self.model = cast(Callable[..., _QuantileModel], QuantileRegressor)(
            **({'alpha': 0.0, 'solver': 'highs'} | params | {'quantile': quantile}))
        _ = self.model.fit(features, target)
        return self

    @override
    def predict(self, data: dict[str, object]) -> dict[str, _FloatArray]:
        if self.model is None:
            raise ValueError('EntropyVaRRegressor is not fitted')
        return {'_preds': self.model.predict(self._entropy(data['x_test']))}

    @override
    def evaluate(self, data: dict[str, object], inline_metrics: bool = True) -> dict[str, object]:
        del inline_metrics
        if self.model is None:
            raise ValueError('EntropyVaRRegressor is not fitted')
        preds = self.predict(data)['_preds']
        truth = np.asarray(data['y_test'], dtype=np.float64)
        residual = truth - preds
        quantile = self.model.quantile
        return continuous_metrics(data, preds) | {'_preds': preds,
            'pinball_loss': float(np.mean(np.maximum(quantile * residual, (quantile - 1) * residual))),
            'violation_rate': float(np.mean(truth < preds))}


def entropy_var_regressor(data: dict[str, object], quantile: float = 0.05, alpha: float = 0.0,
                          fit_intercept: bool = True, solver: str = 'highs',
                          solver_options: dict[str, object] | None = None) -> dict[str, object]:
    '''Fit a lower return quantile; report risk loss without interpreting it as a position.'''
    model = EntropyVaRRegressor().train(data, quantile=quantile, alpha=alpha,
        fit_intercept=fit_intercept, solver=solver, solver_options=solver_options)
    return model.evaluate(data) | {'_model': model}

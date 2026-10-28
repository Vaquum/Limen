from collections.abc import Callable
from math import erf, sqrt
import re
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import polars as pl
from typing_extensions import override

from limen._optional import require_optional
from limen.metrics.binary_metrics import binary_metrics
from limen.sfd.reference_architecture.base import ReferenceModel

__all__ = ['GaussianHMMBinary', 'gaussian_hmm_binary']
_FloatArray = npt.NDArray[np.float64]
_PERCENT = 100.0
_THRESHOLD = 0.5
_FEATURE_DIMENSIONS = 2


class _GaussianModel(Protocol):
    means_: _FloatArray
    covars_: _FloatArray
    transmat_: _FloatArray

    def fit(self, X: _FloatArray) -> object: ...
    def score_samples(self, X: _FloatArray) -> tuple[float, _FloatArray]: ...


def _hmm(params: dict[str, object]) -> _GaussianModel:
    backend = require_optional('hmmlearn.hmm', 'hmmlearn', 'hmm')
    return cast(Callable[..., _GaussianModel], backend.GaussianHMM)(**params)


class GaussianHMMBinary(ReferenceModel):
    '''Fit homogeneous Gaussian return regimes; predict the next positive return.'''
    deterministic, inference_output = False, ('binary', True)

    def __init__(self) -> None:
        super().__init__()
        self.model = _hmm({})
        self._columns: list[str] = []
        self._indices: list[int] = []
        self._width = 0

    def _windows(self, inputs: object) -> _FloatArray:
        if isinstance(inputs, pl.DataFrame):
            values = inputs.select(self._columns).to_numpy()
        else:
            matrix = np.asarray(inputs, dtype=np.float64)
            if matrix.ndim != _FEATURE_DIMENSIONS or matrix.shape[1] != self._width:
                raise ValueError('GaussianHMMBinary inference requires the fitted feature order and width')
            values = matrix[:, self._indices]
        windows = np.asarray(values, dtype=np.float64) * _PERCENT
        if not np.isfinite(windows).all():
            raise ValueError('GaussianHMMBinary requires finite return windows')
        return windows

    @override
    def train(self, data: dict[str, object], **params: object) -> 'GaussianHMMBinary':
        inputs = data['x_train']
        if not isinstance(inputs, pl.DataFrame):
            raise TypeError('GaussianHMMBinary training requires named ret_1_lag_N columns')
        matches = sorted((int(match[1]), column) for column in inputs.columns
                         if (match := re.fullmatch(r'ret_1_lag_(\d+)', column)))
        if not matches or [lag for lag, _ in matches] != list(range(len(matches))):
            raise ValueError('GaussianHMMBinary requires contiguous ret_1_lag_0..N columns')
        self._columns = [column for _, column in reversed(matches)]
        self._indices = [inputs.columns.index(column) for column in self._columns]
        self._width = inputs.width
        self.model = _hmm(params)
        _ = self.model.fit(self._windows(inputs)[:, -1:])
        return self

    @override
    def predict(self, data: dict[str, object]) -> dict[str, object]:
        if not self._columns:
            raise ValueError('GaussianHMMBinary is not fitted')
        windows = self._windows(data['x_test'])
        means, variances = self.model.means_[:, 0], self.model.covars_[:, 0, 0]
        positive = np.asarray([_THRESHOLD * (1 + erf(float(mean) / sqrt(2 * float(variance))))
                               for mean, variance in zip(means, variances, strict=True)])
        probability = np.asarray([self.model.score_samples(row[:, None])[1][-1] @ self.model.transmat_ @ positive
                                  for row in windows], dtype=np.float64)
        return {'_preds': (probability > _THRESHOLD).astype(np.int8), '_probs': probability}

    @override
    def evaluate(self, data: dict[str, object], inline_metrics: bool = True) -> dict[str, object]:
        prediction = self.predict(data)
        preds = cast(npt.NDArray[np.int8], prediction['_preds'])
        probs = cast(_FloatArray, prediction['_probs'])
        self._record_probabilities(data, prediction)
        result = binary_metrics(data, preds, probs)
        result['_preds'] = preds
        if inline_metrics:
            result.update(self._compute_confusion(preds, np.asarray(data['y_test']), data.get('price_data_for_backtest')))
            result.update(self._compute_backtest(preds, data))
        return result


def gaussian_hmm_binary(data: dict[str, object], n_components: int = 2, covariance_type: str = 'diag',
                        min_covar: float = 1e-3, startprob_prior: float = 1.0, transmat_prior: float = 1.0,
                        means_prior: float = 0.0, means_weight: float = 0.0, covars_prior: float = 0.01,
                        covars_weight: float = 1.0, random_state: int | None = 42, n_iter: int = 30,
                        tol: float = 0.01, verbose: bool = False, implementation: str = 'scaling') -> dict[str, object]:
    '''Fit observed training returns; score each completed window independently.'''
    model = GaussianHMMBinary().train(data, n_components=n_components, covariance_type=covariance_type,
        min_covar=min_covar, startprob_prior=startprob_prior, transmat_prior=transmat_prior,
        means_prior=means_prior, means_weight=means_weight, covars_prior=covars_prior,
        covars_weight=covars_weight, random_state=random_state, n_iter=n_iter, tol=tol,
        verbose=verbose, implementation=implementation)
    result = model.evaluate(data)
    result['_model'] = model
    return result

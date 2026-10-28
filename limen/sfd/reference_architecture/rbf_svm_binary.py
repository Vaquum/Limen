from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.svm import SVC
from typing_extensions import override

from limen.calibration import fit_calibrator
from limen.metrics.binary_metrics import binary_metrics
from limen.sfd.reference_architecture.base import ReferenceModel

if TYPE_CHECKING:
    from limen.experiment.manifest_core import CalibrationConfig

__all__ = ['RBFSVMBinary', 'rbf_svm_binary']
_FloatArray = npt.NDArray[np.float64]
_BinaryArray = npt.NDArray[np.int8]
_DEFAULT_THRESHOLD = 0.5


class _ProbabilityModel(Protocol):
    def predict_proba(self, X: object) -> _FloatArray: ...


class _SVCModel(_ProbabilityModel, Protocol):
    def fit(self, X: object, y: object) -> object: ...


def _svc(params: dict[str, object]) -> _SVCModel:
    return cast(Callable[..., _SVCModel], SVC)(kernel='rbf', probability=True, **params)


class RBFSVMBinary(ReferenceModel):
    '''RBF SVC with class-1 probabilities and optional validation calibration.'''
    deterministic, inference_output = False, ('binary', True)

    def __init__(self, prediction_calibration_config: 'CalibrationConfig | None' = None) -> None:
        super().__init__()
        self.model = _svc({})
        self.prediction_calibration_config = prediction_calibration_config
        self._fitted_calibrator: _ProbabilityModel | None = None
        self._calibration_threshold = _DEFAULT_THRESHOLD
        self._val_score: float | None = None

    @override
    def train(self, data: dict[str, object], **params: object) -> 'RBFSVMBinary':
        config = params.pop('prediction_calibration_config', self.prediction_calibration_config)
        self.prediction_calibration_config = cast('CalibrationConfig | None', config)
        y = np.asarray(data['y_train'])
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError('RBFSVMBinary requires both numeric binary training classes 0 and 1')
        calibrated = self.prediction_calibration_config is not None and self.prediction_calibration_config.calibration_func is not None
        self.model = _svc(params)
        _ = self.model.fit(data['x_train'] if calibrated else np.asarray(data['x_train']), y)
        self._fitted_calibrator, self._calibration_threshold, self._val_score = None, _DEFAULT_THRESHOLD, None
        return self

    @override
    def predict(self, data: dict[str, object]) -> dict[str, object]:
        config = self.prediction_calibration_config
        if config is not None:
            if self._fitted_calibrator is None:
                fitted, self._calibration_threshold, self._val_score = fit_calibrator(
                    self.model, config, cast(_FloatArray | pl.DataFrame, data['x_val']),
                    cast(_BinaryArray | pl.Series, data['y_val']))
                self._fitted_calibrator = cast(_ProbabilityModel, fitted)
            frame = np.asarray(data['x_test']) if config.calibration_func is None else data['x_test']
            probability = self._fitted_calibrator.predict_proba(frame)[:, 1]
            return {'_preds': (probability >= self._calibration_threshold).astype(np.int8), '_probs': probability,
                    'optimal_threshold': self._calibration_threshold, 'val_score': self._val_score}
        frame = np.asarray(data['x_test'])
        probability = self.model.predict_proba(frame)[:, 1]
        return {'_preds': (probability > _DEFAULT_THRESHOLD).astype(np.int8), '_probs': probability}

    @override
    def evaluate(self, data: dict[str, object], inline_metrics: bool = True) -> dict[str, object]:
        prediction = self.predict(data)
        preds, probs = cast(_BinaryArray, prediction['_preds']), cast(_FloatArray, prediction['_probs'])
        self._record_probabilities(data, prediction)
        results = binary_metrics(data, preds, probs)
        results.update(_preds=preds, optimal_threshold=prediction.get('optimal_threshold'),
                       val_score=prediction.get('val_score'))
        if inline_metrics:
            results.update(self._compute_confusion(preds, np.asarray(data['y_test']), data.get('price_data_for_backtest')))
            results.update(self._compute_backtest(preds, data))
        return results


def rbf_svm_binary(data: dict[str, object], C: float = 1.0, gamma: str | float = 'scale',
                   shrinking: bool = True, tol: float = 1e-3, cache_size: float = 200,
                   class_weight: str | dict[int, float] | None = None, verbose: bool = False,
                   max_iter: int = -1, random_state: int | None = 42,
                   prediction_calibration_config: 'CalibrationConfig | None' = None) -> dict[str, object]:
    '''Fit binary RBF SVC with native controls; probability training is always enabled.'''
    model = RBFSVMBinary(prediction_calibration_config).train(
        data, C=C, gamma=gamma, shrinking=shrinking, tol=tol, cache_size=cache_size,
        class_weight=class_weight, verbose=verbose, max_iter=max_iter, random_state=random_state)
    result = model.evaluate(data)
    result['_model'] = model
    return result

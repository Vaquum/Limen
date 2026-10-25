from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from importlib import import_module
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import polars as pl

from limen.calibration._objective_threshold import validate_probabilities
from limen.experiment._objective_optimizer import ObjectiveOptimizer
from limen.calibration.pipeline import CalibrationConfigProtocol
from limen.experiment._validation_decisions import Scorer

__all__: list[str] = []
_DEFAULT_THRESHOLD = 0.5


class _Predictor(Protocol):
    def predict_proba(self, values: object) -> object: ...


class _Resolvable(Protocol):
    def resolve(self, params: dict[str, object]) -> CalibrationConfigProtocol: ...


def _closed_scorer(_labels: object, _predictions: object) -> float:
    raise ValueError('Objective validation receipt is already finished')


def resolve_config(value: object, params: dict[str, object]) -> CalibrationConfigProtocol | None:
    use_calibration, use_threshold = params.get('use_calibration', True), params.get('use_threshold', True)
    if not use_calibration and not use_threshold:
        return None
    kind = vars(import_module('limen.experiment.manifest_core'))['CalibrationConfig']
    if not isinstance(value, cast(type[object], kind)):
        raise ValueError('Objective requires a manifest calibration configuration')
    resolve = cast(_Resolvable, value).resolve
    config = resolve(params)
    factory = cast(Callable[..., CalibrationConfigProtocol], kind)
    return factory(calibration_func=config.calibration_func if use_calibration else None, calibration_params=config.calibration_params,
                   threshold_func=config.threshold_func if use_threshold else None, threshold_params=config.threshold_params)


class _Calibrated:
    def __init__(self, predictor: object, receipt: ValidationReceipt) -> None:
        super().__init__()
        self.predictor, self.receipt = cast(_Predictor, predictor), receipt

    def predict_proba(self, values: object) -> object:
        output = self.predictor.predict_proba(values)
        if self.receipt.inputs is not None and np.array_equal(np.asarray(values), self.receipt.inputs, equal_nan=True):
            self.receipt.probabilities = validate_probabilities(np.asarray(output)[:, 1]).copy()
        return output


class ValidationReceipt:
    def __init__(self, config: CalibrationConfigProtocol, scorer: Scorer, maximize: bool,
                 *, validation_inputs: object, validation_labels: object) -> None:
        super().__init__()
        self.config = config
        self.scorer: Callable[[object, object], float] = scorer
        self.maximize = maximize
        self.inputs: npt.NDArray[np.float64] | None = np.asarray(validation_inputs, dtype=np.float64).copy()
        self.labels: npt.NDArray[np.float64] | None = np.asarray(validation_labels, dtype=np.float64).copy()
        self.probabilities: npt.NDArray[np.float64] | None = None
        self.predictor: _Calibrated | None = None
        self.optimizer = None if config.threshold_func is None else ObjectiveOptimizer(config.threshold_func, scorer, maximize, scorer.inputs.signals.height, labels=self.labels)

    def calibrate(self, clf: object, x_val: object, y_val: object, **params: object) -> object:
        if self.inputs is None or self.labels is None or not np.array_equal(np.asarray(x_val), self.inputs, equal_nan=True) or not np.array_equal(np.asarray(y_val), self.labels, equal_nan=True):
            raise ValueError('Objective calibration fit must use the original ordered validation features and labels')
        function = self.config.calibration_func
        assert function is not None
        self.predictor = _Calibrated(function(clf, cast(npt.NDArray[np.generic] | pl.DataFrame, x_val), cast(npt.NDArray[np.generic] | pl.Series, y_val), **params), self)
        return self.predictor

    def resolved(self) -> CalibrationConfigProtocol:
        factory = cast(Callable[..., CalibrationConfigProtocol], vars(import_module('limen.experiment.manifest_core'))['CalibrationConfig'])
        return factory(calibration_func=self.calibrate if self.config.calibration_func is not None else None, calibration_params=self.config.calibration_params,
                       threshold_func=self.optimizer, threshold_params={**self.config.threshold_params, 'metric': self.scorer, '_objective_maximize': self.maximize})

    def verify(self, predictions: object, probabilities: object, payload: Mapping[str, object]) -> None:
        if probabilities is None:
            raise ValueError('Active objective calibration/threshold selection requires validation probabilities')
        values = validate_probabilities(probabilities)
        if self.config.calibration_func is not None and (self.probabilities is None or not np.array_equal(values, self.probabilities)):
            raise ValueError('Objective validation must use probabilities from its actual fitted calibrator')
        if self.optimizer is not None:
            if self.optimizer.selection is None or not np.array_equal(values, self.optimizer.selection[0]) or not np.array_equal(np.asarray(predictions), values >= self.optimizer.selection[1]):
                raise ValueError('Objective decisions must use its selected threshold on the original validation probabilities')
        else:
            mapping = payload.get('probability_to_decision')
            if mapping is None:
                expected = values >= _DEFAULT_THRESHOLD
            elif callable(mapping):
                expected = np.asarray(mapping(values.copy()))
                if not np.array_equal(expected, np.asarray(mapping(values.copy()))):
                    raise ValueError('Objective calibrated decision mapping must be stable')
            else:
                raise ValueError('Objective calibrated decision mapping must be callable')
            if not np.array_equal(np.asarray(predictions), expected):
                raise ValueError('Objective decisions must use its calibrated validation probabilities')

    def finish(self, model: object) -> None:
        restored: dict[str, object] = {}
        if self.predictor is not None and getattr(model, '_fitted_calibrator', None) is self.predictor:
            restored['_fitted_calibrator'] = self.predictor.predictor
        if getattr(model, 'prediction_calibration_config', None) is not None:
            restored['prediction_calibration_config'] = deepcopy(self.config)
        for name, value in restored.items():
            setattr(model, name, value)
        self.inputs = self.labels = self.probabilities = None
        self.scorer = _closed_scorer
        if self.optimizer is not None:
            self.optimizer.selection = None
            self.optimizer.labels = None
            self.optimizer.evaluated.clear()
            self.optimizer.scorer = _closed_scorer

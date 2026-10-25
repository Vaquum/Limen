from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from limen.calibration._objective_threshold import validate_probabilities
from limen.experiment._objective_optimizer import ObjectiveOptimizer

if TYPE_CHECKING:
    from limen.experiment._objective import ObjectiveConfig, ValidationScorer
    from limen.experiment.manifest_core import CalibrationConfig

__all__: list[str] = []


class _Calibrated:
    def __init__(self, predictor: object, receipt: ValidationReceipt) -> None:
        self.predictor, self.receipt = predictor, receipt

    def predict_proba(self, values: object) -> object:
        output = getattr(self.predictor, 'predict_proba')(values)
        if self.receipt.inputs is not None and np.array_equal(np.asarray(values), self.receipt.inputs, equal_nan=True):
            self.receipt.probabilities = validate_probabilities(np.asarray(output)[:, 1]).copy()
        return output


class ValidationReceipt:
    def __init__(self, config: CalibrationConfig, scorer: ValidationScorer, objective: ObjectiveConfig,
                 *, validation_inputs: object, validation_labels: object) -> None:
        self.config = config
        self.scorer, self.objective = scorer, objective
        self.inputs: npt.NDArray[np.float64] | None = np.asarray(validation_inputs, dtype=np.float64).copy()
        self.labels: npt.NDArray[np.float64] | None = np.asarray(validation_labels, dtype=np.float64).copy()
        self.probabilities: npt.NDArray[np.float64] | None = None
        self.predictor: _Calibrated | None = None
        self.optimizer = None if config.threshold_func is None else ObjectiveOptimizer(config.threshold_func, scorer, objective.maximize, scorer.inputs.signals.height)

    def calibrate(self, clf: object, x_val: object, y_val: object, **params: object) -> object:
        if not np.array_equal(np.asarray(x_val), self.inputs, equal_nan=True) or not np.array_equal(np.asarray(y_val), self.labels, equal_nan=True):
            raise ValueError('Objective calibration fit must use the original ordered validation features and labels')
        function = self.config.calibration_func
        assert function is not None
        self.predictor = _Calibrated(function(clf, x_val, y_val, **params), self)
        return self.predictor

    def resolved(self) -> CalibrationConfig:
        from limen.experiment._objective import threshold_params
        return replace(self.config, calibration_func=self.calibrate if self.config.calibration_func is not None else None,
                       threshold_func=self.optimizer, threshold_params=threshold_params(self.config, self.scorer, self.objective))

    def verify(self, predictions: object, probabilities: object, payload: Mapping[str, object]) -> None:
        if probabilities is None:
            raise ValueError('Active objective calibration/threshold selection requires validation probabilities')
        values = validate_probabilities(probabilities)
        if self.config.calibration_func is not None:
            if self.probabilities is None or not np.array_equal(values, self.probabilities):
                raise ValueError('Objective validation must use probabilities from its actual fitted calibrator')
        if self.optimizer is not None:
            if self.optimizer.selection is None or not np.array_equal(values, self.optimizer.selection[0]) or not np.array_equal(predictions, values >= self.optimizer.selection[1]):
                raise ValueError('Objective decisions must use its selected threshold on the original validation probabilities')
        else:
            mapping = payload.get('probability_to_decision')
            if mapping is None:
                expected = values >= 0.5
            elif callable(mapping):
                expected = mapping(values.copy())
                if not np.array_equal(expected, mapping(values.copy())):
                    raise ValueError('Objective calibrated decision mapping must be stable')
            else:
                raise ValueError('Objective calibrated decision mapping must be callable')
            if not np.array_equal(predictions, expected):
                raise ValueError('Objective decisions must use its calibrated validation probabilities')

    def finish(self, model: object) -> None:
        if self.predictor is not None and getattr(model, '_fitted_calibrator', None) is self.predictor:
            setattr(model, '_fitted_calibrator', self.predictor.predictor)
        if getattr(model, 'prediction_calibration_config', None) is not None:
            setattr(model, 'prediction_calibration_config', deepcopy(self.config))
        self.inputs = self.labels = self.probabilities = None
        if self.optimizer is not None:
            self.optimizer.selection = None

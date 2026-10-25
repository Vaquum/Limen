from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
import inspect
import math
from numbers import Real
from typing import cast

import numpy as np
import numpy.typing as npt
import polars as pl

from limen.calibration._objective_threshold import validate_probabilities
from limen.calibration.pipeline import CalibrationConfigProtocol, ThresholdOptimizerProtocol

__all__: list[str] = []
_PROTECTED = {'metric', '_objective_maximize', 'y_val', 'val_proba'}
_RESULT_FIELDS = 2


def check_optimizer(config: CalibrationConfigProtocol) -> None:
    function = config.threshold_func
    if function is None:
        return
    if not callable(function) or _PROTECTED.intersection(config.threshold_params):
        raise ValueError('Objective conflicts with explicit threshold metric/direction or validation operands')
    while isinstance(function, partial):
        bound = inspect.signature(function.func).bind_partial(*function.args, **function.keywords)
        if _PROTECTED.intersection(bound.arguments) or _PROTECTED.intersection(function.keywords):
            raise ValueError('Objective conflicts with bound threshold metric/direction or validation operands')
        function = function.func


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f'Objective optimizer {name} must be a finite real number')
    return float(value)


@dataclass
class ObjectiveOptimizer:
    optimizer: ThresholdOptimizerProtocol
    scorer: Callable[[object, object], float]
    maximize: bool
    rows: int
    evaluated: dict[bytes, float] = field(default_factory=dict[bytes, float])
    selection: tuple[npt.NDArray[np.float64], float] | None = None

    def score(self, labels: object, predictions: object) -> float:
        result = self.scorer(labels, predictions)
        self.evaluated[np.asarray(predictions, dtype=np.int8).tobytes()] = result
        return result

    def __call__(self, y_val: object, val_proba: object, **params: object) -> tuple[float, float]:
        probabilities = validate_probabilities(val_proba).copy()
        if probabilities.size != self.rows:
            raise ValueError('Objective optimizer probabilities do not match validation rows')
        if params.get('metric') is not self.scorer or params.get('_objective_maximize') is not self.maximize:
            raise ValueError('Objective optimizer metric/direction changed')
        self.evaluated.clear()
        self.selection = None
        labels = cast(npt.NDArray[np.generic] | pl.Series, y_val)
        result = cast(object, self.optimizer(labels, probabilities.copy(), **{**params, 'metric': self.score}))
        pair = cast(tuple[object, ...], result)
        if not isinstance(result, tuple) or len(pair) != _RESULT_FIELDS:
            raise ValueError('Objective optimizer must return a threshold/score pair')
        threshold, score = _finite(pair[0], 'threshold'), _finite(pair[1], 'score')
        decisions = (probabilities >= threshold).astype(np.int8)
        expected = self.scorer(y_val, decisions)
        if decisions.tobytes() not in self.evaluated or not np.isclose(score, expected, rtol=1e-9, atol=1e-12):
            raise ValueError('Objective optimizer must score its selected validation decisions with the declared objective')
        self.evaluated.clear()
        self.selection = (probabilities, threshold)
        return threshold, score

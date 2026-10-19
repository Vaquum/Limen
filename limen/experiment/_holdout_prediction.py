from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import inspect
import random
import sys
from types import FrameType
from typing import cast

import numpy as np
import polars as pl

from limen.experiment._holdout_dependencies import children, fitted_witness, inference_functions
from limen.inference.sensor import Sensor

__all__ = ['readonly_prediction', 'validate_prediction']
_EFFECTS = {'fit', 'fit_transform', 'partial_fit', 'train', 'fit_calibrator', 'grid_threshold_optimizer'}


def _roots(sensor: Sensor) -> list[object]:
    return [vars(sensor)['_model'], vars(sensor)['_fitted_params']]


@contextmanager
def readonly_prediction(sensors: list[Sensor]) -> Iterator[None]:
    forbidden: set[object] = set()
    seen: set[int] = set()

    def collect(value: object) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        for name in _EFFECTS:
            method = getattr(value, name, None)
            if callable(method):
                method = inspect.unwrap(method)
                if inspect.ismethod(method):
                    method = method.__func__
                if inspect.isfunction(method):
                    forbidden.add(method.__code__)
        config = getattr(value, 'prediction_calibration_config', None)
        if config is not None:
            for name in ('calibration_func', 'threshold_func'):
                method = getattr(config, name, None)
                if inspect.isfunction(method):
                    forbidden.add(method.__code__)
        for part in children(value):
            collect(part)

    for sensor in sensors:
        for root in _roots(sensor):
            collect(root)
    numpy_state, python_state = fitted_witness(np.random.get_state()), random.getstate()
    previous = sys.getprofile()
    violation = False

    def profile(frame: FrameType, event: str, arg: object) -> None:
        nonlocal violation
        if event == 'call' and (frame.f_code in forbidden or frame.f_code.co_name in _EFFECTS):
            violation = True
            raise ValueError('Holdout inference attempted fitting/calibration/threshold search')
        if previous is not None:
            previous(frame, event, arg)

    sys.setprofile(profile)
    try:
        yield
        if violation:
            raise ValueError('Holdout inference attempted fitting/calibration/threshold search')
        if fitted_witness(np.random.get_state()) != numpy_state or random.getstate() != python_state:
            raise ValueError('Holdout inference consumed randomness')
    finally:
        sys.setprofile(previous)


def validate_prediction(sensor: Sensor) -> None:
    roots = _roots(sensor)
    functions, _ = inference_functions(roots)
    for function in functions:
        if getattr(function, '__module__', type(function).__module__).startswith(('numpy.random', 'random')):
            raise ValueError('Holdout inference has an unowned random dependency')
        if not inspect.isfunction(function) or function.__module__.startswith(('numpy.', 'polars.', 'sklearn.', 'lightgbm.', 'scipy.')):
            continue
        if any(name in function.__code__.co_names for name in ('random', 'default_rng', 'RandomState', 'Generator')):
            raise ValueError('Holdout inference has an unowned random dependency')
    raw = vars(sensor)['_fitted_params'].get('_holdout_probe')
    if not isinstance(raw, pl.DataFrame) or raw.is_empty():
        raise ValueError('Holdout requires recorded Trainer inference geometry')
    state = fitted_witness(roots)
    with readonly_prediction([sensor]):
        first = sensor.predict(raw)
    with readonly_prediction([sensor]):
        second = sensor.predict(raw)
    if fitted_witness(roots) != state:
        raise ValueError('Holdout inference mutated fitted state')
    if first != second or first.reason is not None or isinstance(first.prediction, bool) or first.prediction not in (0, 1):
        raise ValueError('Holdout requires fixed fitted binary inference')
    if first.probability is not None and (not np.isfinite(first.probability) or not 0 <= cast(float, first.probability) <= 1):
        raise ValueError('Holdout requires finite fitted probabilities')

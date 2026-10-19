from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import inspect
from functools import partial
import random
import sys
from types import CodeType, FrameType
from typing import cast

import numpy as np
import polars as pl

from limen.experiment._holdout_dependencies import FITTING_METHODS, children, fitted_witness, fitting_call, inference_functions
from limen.inference.sensor import Sensor
from limen.experiment.manifest_core import MLManifest

__all__ = ['readonly_prediction', 'validate_prediction']
_EFFECTS = FITTING_METHODS


def _roots(sensor: Sensor) -> list[object]:
    return [vars(sensor)['_model'], vars(sensor)['_fitted_params']]


def _call_codes(value: object) -> tuple[CodeType, ...]:
    if isinstance(value, partial):
        return _call_codes(value.func)
    if isinstance(value, type):
        return (*_call_codes(value.__init__), *_call_codes(value.__new__), *_call_codes(type(value).__call__))
    method = inspect.unwrap(value) if callable(value) else value
    if inspect.ismethod(method):
        method = method.__func__
    if inspect.isfunction(method):
        return (method.__code__,)
    call = inspect.getattr_static(cast(type[object], type(method)), '__call__', None)
    return (call.__code__,) if inspect.isfunction(call) else ()


@contextmanager
def readonly_prediction(sensors: list[Sensor]) -> Iterator[None]:
    forbidden: set[object] = set()
    seen: set[int] = set()

    def collect(value: object) -> None:
        if value is None or isinstance(value, (str, bool, int, float, bytes, np.generic, np.ndarray, pl.DataFrame, pl.Series)):
            return
        if id(value) in seen:
            return
        seen.add(id(value))
        for name in _EFFECTS:
            method = getattr(value, name, None)
            forbidden.update(_call_codes(method))
        config = getattr(value, 'prediction_calibration_config', None)
        if config is not None:
            for name in ('calibration_func', 'threshold_func'):
                method = getattr(config, name, None)
                forbidden.update(_call_codes(method))
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
        fixed_labels = frame.f_globals.get('__name__') == 'sklearn.preprocessing._label' and frame.f_code.co_name == 'fit' and np.array_equal(frame.f_locals.get('y', []), [0, 1])
        if event == 'call' and not fixed_labels and (frame.f_code in forbidden or fitting_call(str(frame.f_globals.get('__name__', '')), getattr(frame.f_code, 'co_qualname', frame.f_code.co_name))):
            violation = True
            raise ValueError('Holdout inference attempted fitting/calibration/threshold search')
        if previous is not None:
            _ = previous(frame, event, arg)

    sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(previous)
        if violation:
            raise ValueError('Holdout inference attempted fitting/calibration/threshold search')
        if fitted_witness(np.random.get_state()) != numpy_state or random.getstate() != python_state:
            raise ValueError('Holdout inference consumed randomness')


def _probe_output(sensor: Sensor) -> object:
    raw = vars(sensor)['_fitted_params'].get('_holdout_probe')
    if not isinstance(raw, pl.DataFrame) or raw.is_empty():
        raise ValueError('Holdout requires recorded Trainer inference geometry')
    manifest = cast(MLManifest, vars(Sensor)['_get_manifest'](sensor))
    guarded = manifest.test_predict_guard
    manifest.test_predict_guard = False
    try:
        result = sensor.predict(raw)
    finally:
        manifest.test_predict_guard = guarded
    if result.reason is not None or isinstance(result.prediction, bool) or result.prediction not in (0, 1):
        raise ValueError(f'Holdout requires fitted binary inference: {result.reason}')
    if result.probability is not None and (not np.isfinite(result.probability) or not 0 <= result.probability <= 1):
        raise ValueError('Holdout requires finite fitted probabilities')
    return result


def validate_prediction(sensor: Sensor) -> None:
    roots = _roots(sensor)
    config = getattr(vars(sensor)['_model'], 'prediction_calibration_config', None)
    functions, _ = inference_functions(roots, exclude=() if config is None else (config,))
    for function in functions:
        if getattr(function, '__module__', type(function).__module__).startswith(('numpy.random', 'random')):
            raise ValueError('Holdout inference has an unowned random dependency')
        if not inspect.isfunction(function) or function.__module__.startswith(('numpy.', 'polars.', 'sklearn.', 'lightgbm.', 'scipy.')):
            continue
        if any(name in function.__code__.co_names for name in ('random', 'default_rng', 'RandomState', 'Generator')):
            raise ValueError('Holdout inference has an unowned random dependency')
    state = fitted_witness(roots)
    with readonly_prediction([sensor]):
        first = fitted_witness(_probe_output(sensor))
    with readonly_prediction([sensor]):
        second = fitted_witness(_probe_output(sensor))
    if fitted_witness(roots) != state:
        raise ValueError('Holdout inference mutated fitted state')
    if first != second:
        raise ValueError('Holdout requires fixed fitted binary inference')

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import inspect
from hashlib import sha256
import marshal
import random
import sys
from types import CodeType, FrameType

import numpy as np
import polars as pl

from limen.experiment._holdout_dependencies import children, fitted_witness, inference_functions
from limen.inference.sensor import Sensor

__all__ = ['readonly_prediction', 'validate_prediction']
_EFFECTS = {'fit', 'fit_transform', 'partial_fit', 'train', 'fit_calibrator', 'grid_threshold_optimizer'}


def _roots(sensor: Sensor) -> list[object]:
    return [vars(sensor)['_model'], vars(sensor)['_fitted_params']]


@contextmanager
def readonly_prediction(sensors: list[Sensor], calls: set[CodeType] | None = None) -> Iterator[None]:
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
        if calls is not None and event == 'call':
            calls.add(frame.f_code)
        fixed_labels = frame.f_globals.get('__name__') == 'sklearn.preprocessing._label' and frame.f_code.co_name == 'fit' and np.array_equal(frame.f_locals.get('y'), [0, 1])
        if event == 'call' and not fixed_labels and (frame.f_code in forbidden or frame.f_code.co_name in _EFFECTS):
            violation = True
            raise ValueError('Holdout inference attempted fitting/calibration/threshold search')
        if previous is not None:
            previous(frame, event, arg)

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
    manifest = sensor._get_manifest()
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
    functions, _ = inference_functions(roots)
    for function in functions:
        if getattr(function, '__module__', type(function).__module__).startswith(('numpy.random', 'random')):
            raise ValueError('Holdout inference has an unowned random dependency')
        if not inspect.isfunction(function) or function.__module__.startswith(('numpy.', 'polars.', 'sklearn.', 'lightgbm.', 'scipy.')):
            continue
        if any(name in function.__code__.co_names for name in ('random', 'default_rng', 'RandomState', 'Generator')):
            raise ValueError('Holdout inference has an unowned random dependency')
    state = fitted_witness(roots)
    calls: set[CodeType] = set()
    with readonly_prediction([sensor], calls):
        first = fitted_witness(_probe_output(sensor))
    with readonly_prediction([sensor], calls):
        second = fitted_witness(_probe_output(sensor))
    if fitted_witness(roots) != state:
        raise ValueError('Holdout inference mutated fitted state')
    if first != second:
        raise ValueError('Holdout requires fixed fitted binary inference')
    vars(sensor)['_fitted_params']['_holdout_inference_calls'] = tuple(sorted(sha256(marshal.dumps(code)).hexdigest() for code in calls))

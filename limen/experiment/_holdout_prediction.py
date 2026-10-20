from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import inspect
from functools import partial
import random
import sys
from types import CodeType, FrameType, FunctionType
from typing import cast

import numpy as np
import polars as pl

from limen.experiment._holdout_dependencies import FITTING_METHODS, children, fitted_witness, fitting_call, object_array
from limen.inference.sensor import Sensor
from limen.experiment.manifest_core import MLManifest

__all__ = ['readonly_prediction', 'restore_iterations', 'validate_prediction']
_EFFECTS = FITTING_METHODS


def restore_iterations(source: object, memo: Mapping[int, object]) -> None:
    seen: set[int] = set()

    def visit(value: object) -> None:
        if value is None or isinstance(value, (str, bool, int, float, bytes, np.generic)) or id(value) in seen:
            return
        seen.add(id(value))
        if type(value).__module__ == 'lightgbm.basic' and type(value).__name__ == 'Booster':
            from lightgbm import Booster
            original = cast(Booster, value)
            prediction = inspect.getattr_static(cast(type[object], type(original)), 'predict', None)
            if 'predict' in vars(original) or not inspect.isfunction(prediction) or (prediction.__module__, prediction.__qualname__) != ('lightgbm.basic', 'Booster.predict'):
                raise ValueError('Holdout refuses replaced Booster prediction methods')
            copied = cast(Booster, memo[id(value)])
            copied.best_iteration = original.best_iteration
        for part in children(value):
            visit(part)

    visit(source)


def _roots(sensor: Sensor) -> list[object]:
    return [vars(sensor)['_model'], vars(sensor)['_fitted_params']]


def _call_functions(value: object) -> tuple[FunctionType, ...]:
    if isinstance(value, partial):
        provider = cast(partial[object], value)
        return (*_call_functions(provider.func), *_call_functions(type(provider).__call__))
    if isinstance(value, type):
        return (*_call_functions(value.__init__), *_call_functions(value.__new__), *_call_functions(type(value).__call__))
    if inspect.ismethod(value):
        value = value.__func__
    if inspect.isfunction(value):
        unwrapped = inspect.unwrap(value)
        return (value, *_call_functions(unwrapped)) if unwrapped is not value else (value,)
    call = inspect.getattr_static(cast(type[object], type(value)), '__call__', None)
    functions = _call_functions(call.fget) if isinstance(call, property) else _call_functions(call) if inspect.isfunction(call) else ()
    unwrapped = inspect.unwrap(value) if callable(value) else value
    return (*functions, *_call_functions(unwrapped)) if unwrapped is not value else functions


def _matches(function: FunctionType, frame: FrameType) -> bool:
    return frame.f_globals is function.__globals__ and all(
        frame.f_locals.get(name) is cell.cell_contents
        for name, cell in zip(function.__code__.co_freevars, function.__closure__ or (), strict=True))


@contextmanager
def readonly_prediction(sensors: list[Sensor]) -> Iterator[None]:
    forbidden: dict[CodeType, set[FunctionType]] = {}
    seen: set[int] = set()

    def collect(value: object) -> None:
        if value is None or isinstance(value, (str, bool, int, float, bytes, np.generic, pl.DataFrame, pl.Series)) or (issubclass(type(value), np.ndarray) and not object_array(value)):
            return
        if id(value) in seen:
            return
        seen.add(id(value))
        for name in _EFFECTS:
            method = getattr(value, name, None)
            for function in _call_functions(method):
                forbidden.setdefault(function.__code__, set()).add(function)
        config = getattr(value, 'prediction_calibration_config', None)
        if config is not None:
            for name in ('calibration_func', 'threshold_func'):
                method = getattr(config, name, None)
                for function in _call_functions(method):
                forbidden.setdefault(function.__code__, set()).add(function)
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
        matched = event == 'call' and frame.f_code in forbidden and any(_matches(function, frame) for function in forbidden[frame.f_code])
        if event == 'call' and not fixed_labels and (matched or fitting_call(str(frame.f_globals.get('__name__', '')), getattr(frame.f_code, 'co_qualname', frame.f_code.co_name))):
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
    state = fitted_witness(roots)
    with readonly_prediction([sensor]):
        first = fitted_witness(_probe_output(sensor))
    with readonly_prediction([sensor]):
        second = fitted_witness(_probe_output(sensor))
    if fitted_witness(roots) != state:
        raise ValueError('Holdout inference mutated fitted state')
    if first != second:
        raise ValueError('Holdout requires fixed fitted binary inference')

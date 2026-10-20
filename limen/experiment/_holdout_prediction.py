from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import inspect
from functools import partial
import random
import sys
from types import FrameType, FunctionType
from typing import cast

import numpy as np
import polars as pl

from limen.experiment._holdout_dependencies import FITTING_METHODS, call_functions, children, fitted_witness, fitting_call, entropy_call, inference_evidence, object_array, restore_iterations
from limen.experiment._holdout_calls import call_codes
from limen.inference.sensor import Sensor
from limen.experiment.manifest_core import MLManifest

__all__ = ['readonly_prediction', 'restore_iterations', 'validate_prediction']
_EFFECTS = FITTING_METHODS


def _roots(sensor: Sensor) -> list[object]:
    return [vars(sensor)['_model'], vars(sensor)['_fitted_params']]


def _matches(function: FunctionType, frame: FrameType, owner: object | None) -> bool:
    if owner is not None:
        code = frame.f_code
        if code.co_argcount:
            receiver = frame.f_locals.get(code.co_varnames[0])
        elif code.co_flags & inspect.CO_VARARGS:
            arguments = frame.f_locals.get(code.co_varnames[code.co_kwonlyargcount], ())
            receiver = cast(tuple[object, ...], arguments)[0] if isinstance(arguments, tuple) and arguments else None
        else:
            return False
        if receiver is not owner:
            return False
    return frame.f_globals is function.__globals__ and all(
        frame.f_locals.get(name) is cell.cell_contents
        for name, cell in zip(function.__code__.co_freevars, function.__closure__ or (), strict=True))


@contextmanager
def readonly_prediction(sensors: list[Sensor], *, providers: Sequence[object] | None = None) -> Iterator[None]:
    forbidden: dict[FunctionType, list[object | None]] = {}
    seen: set[int] = set()
    configs: list[object] = []

    def forbid(method: object) -> None:
        if not callable(method):
            return
        call = None if inspect.isfunction(method) or inspect.ismethod(method) else inspect.getattr_static(cast(type[object], type(method)), '__call__', None)
        calls = call_functions(call)
        functions = call_functions(method)
        if not functions:
            raise ValueError('Holdout refuses fitting callbacks without observable Python call protection')
        bound_owner = method.__self__ if inspect.ismethod(method) else None
        call_owner: object | None = type(method) if isinstance(call, classmethod) else None if isinstance(call, staticmethod) else method
        wrapped = cast(partial[object], method).func if isinstance(method, partial) else None
        while isinstance(wrapped, (partial, staticmethod)):
            if isinstance(wrapped, staticmethod):
                wrapped = cast('staticmethod[..., object]', wrapped).__func__
            else:
                provider = cast(partial[object], wrapped)
                wrapped = provider
                if inspect.getattr_static(cast(type[object], type(provider)), '__call__', None) is not partial[object].__call__:
                    break
                wrapped = provider.func
        if inspect.ismethod(wrapped):
            wrapped_owner = wrapped.__self__
        elif wrapped is None or inspect.isfunction(wrapped) or isinstance(wrapped, type):
            wrapped_owner = None
        else:
            wrapped_call = inspect.getattr_static(cast(type[object], type(wrapped)), '__call__', None)
            wrapped_owner = type(wrapped) if isinstance(wrapped_call, classmethod) else None if isinstance(wrapped_call, staticmethod) else wrapped
        for function in functions:
            owner = bound_owner if bound_owner is not None else call_owner if function in calls else wrapped_owner
            forbidden.setdefault(function, []).append(owner)

    def collect(value: object) -> None:
        if value is None or type(value) in (str, bool, int, float, bytes) or isinstance(value, (pl.DataFrame, pl.Series)) or (type(value).__module__ == 'numpy' and type(value).__name__ == 'ndarray' and not object_array(value)):
            return
        if id(value) in seen:
            return
        seen.add(id(value))
        for name in _EFFECTS:
            forbid(getattr(value, name, None))
        config = getattr(value, 'prediction_calibration_config', None)
        if config is not None:
            configs.append(config)
            for name in ('calibration_func', 'threshold_func'):
                forbid(getattr(config, name, None))
        parts = children(value)
        if inspect.ismethod(value):
            parts += (value.__func__, value.__self__)
        elif inspect.isfunction(value):
            parts += (value.__defaults__, value.__kwdefaults__, tuple(cell.cell_contents for cell in value.__closure__ or ()))
        for part in parts:
            collect(part)

    for sensor in sensors:
        for root in _roots(sensor):
            collect(root)
    if providers is None:
        reachable: list[object] = []
        roots = [root for sensor in sensors for root in _roots(sensor)]
        roots.extend(getattr(Sensor, name) for name in ('predict', '_get_manifest', '_last_bar_inside_training_window'))
        roots.extend((MLManifest.sensor_input_prep, MLManifest._prepare_column_shims, MLManifest.resolve_trade_policy))
        _ = inference_evidence(roots, exclude=configs, resolved=(vars(Sensor)['_get_manifest'],), capture=reachable)
        providers = reachable
    for value in providers:
        collect(value)
    numpy_state, python_state = fitted_witness(np.random.get_state()), random.getstate()
    previous = sys.getprofile()
    violation: str | None = None

    def profile(frame: FrameType, event: str, arg: object) -> None:
        nonlocal violation
        if (event == 'call' and (frame.f_globals.get('__name__'), frame.f_code.co_name) in (('threading', 'start'), ('concurrent.futures.thread', 'submit'))) or (event == 'c_call' and getattr(arg, '__module__', None) == '_thread' and getattr(arg, '__name__', None) == 'start_new_thread'):
            violation = 'Holdout inference attempted thread dispatch'
            raise ValueError(violation)
        if event == 'c_call' and entropy_call(arg):
            violation = 'Holdout inference consumed unowned randomness'
            raise ValueError(violation)
        fixed_labels = frame.f_globals.get('__name__') == 'sklearn.preprocessing._label' and frame.f_code.co_name == 'fit' and np.array_equal(frame.f_locals.get('y', []), [0, 1])
        matched = event == 'call' and id(frame.f_code) in observed and any(_matches(observed[id(frame.f_code)][0], frame, owner) for owner in observed[id(frame.f_code)][1])
        if event == 'call' and not fixed_labels and (matched or fitting_call(str(frame.f_globals.get('__name__', '')), getattr(frame.f_code, 'co_qualname', frame.f_code.co_name))):
            violation = 'Holdout inference attempted fitting/calibration/threshold search'
            raise ValueError(violation)
        if previous is not None:
            _ = previous(frame, event, arg)

    try:
        with call_codes(tuple(forbidden)):
            observed = {id(function.__code__): (function, owners) for function, owners in forbidden.items()}
            sys.setprofile(profile)
            try:
                yield
            finally:
                sys.setprofile(previous)
    finally:
        if violation is not None:
            raise ValueError(violation)
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


def validate_prediction(sensor: Sensor, *, providers: Sequence[object] | None = None) -> None:
    roots = _roots(sensor)
    state = fitted_witness(roots)
    with readonly_prediction([sensor], providers=providers):
        first = fitted_witness(_probe_output(sensor))
    with readonly_prediction([sensor], providers=providers):
        second = fitted_witness(_probe_output(sensor))
    if fitted_witness(roots) != state:
        raise ValueError('Holdout inference mutated fitted state')
    if first != second:
        raise ValueError('Holdout requires fixed fitted binary inference')

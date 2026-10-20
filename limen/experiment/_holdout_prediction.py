from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from cProfile import Profile
import inspect
from functools import partial
import random
import sys
from types import FrameType, FunctionType
from typing import cast

import numpy as np
import polars as pl

from limen.experiment._holdout_dependencies import FITTING_METHODS, children, fitted_witness, fitting_call, entropy_call, inference_evidence, object_payload, restore_iterations
from limen.experiment._holdout_calls import call_codes, call_functions, constructor_entry, matches as _matches, native_attribute, receiver_position, same_callback
from limen.inference.sensor import Sensor
from limen.experiment.manifest_core import MLManifest

__all__ = ['readonly_prediction', 'restore_iterations', 'validate_prediction']
_EFFECTS = FITTING_METHODS


def _roots(sensor: Sensor) -> list[object]:
    return [vars(sensor)['_model'], vars(sensor)['_fitted_params']]



@contextmanager
def readonly_prediction(sensors: list[Sensor], *, providers: Sequence[object] | None = None) -> Iterator[None]:
    forbidden: dict[FunctionType, list[tuple[object | None, bool, int]]] = {}
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
        call_owner: object | None = call.__self__ if inspect.ismethod(call) else type(method) if isinstance(call, classmethod) else None if isinstance(call, staticmethod) else method
        wrapped = native_attribute(cast(object, method), 'func') if isinstance(method, partial) else None
        while isinstance(wrapped, (partial, staticmethod)):
            if isinstance(wrapped, staticmethod):
                wrapped = native_attribute(cast(object, wrapped), '__func__')
            else:
                provider = cast(partial[object], wrapped)
                wrapped = provider
                if inspect.getattr_static(cast(type[object], type(provider)), '__call__', None) is not partial[object].__call__:
                    break
                wrapped = native_attribute(provider, 'func')
        wrapped_entry: object
        if inspect.ismethod(wrapped):
            wrapped_owner = wrapped.__self__
            wrapped_entry = wrapped
        elif wrapped is None or inspect.isfunction(wrapped):
            wrapped_owner = None
            wrapped_entry = wrapped
        else:
            wrapped_call = inspect.getattr_static(cast(type[object], type(wrapped)), '__call__', None)
            wrapped_owner = wrapped_call.__self__ if inspect.ismethod(wrapped_call) else type(wrapped) if isinstance(wrapped_call, classmethod) else None if isinstance(wrapped_call, staticmethod) else wrapped
            wrapped_entry = cast(object, wrapped_call)
        class_target: type[object] | None = wrapped if isinstance(wrapped, type) else method if isinstance(method, type) else None
        for function in functions:
            owner = bound_owner if bound_owner is not None else call_owner if function in calls else wrapped_owner
            instance, index = False, 0
            if class_target is not None and function not in calls and inspect.getattr_static(type(class_target), '__call__', None) is type.__call__:
                entry = constructor_entry(class_target)
                owner, instance, index = class_target, entry.instance, entry.position
            else:
                source = cast(object, method if bound_owner is not None else call if function in calls else wrapped_entry)
                index = receiver_position(source) if owner is not None else 0
                if index is None:
                    raise ValueError('Holdout refuses fitting callbacks without observable Python call protection')
            forbidden.setdefault(function, []).append((owner, instance, index))

    def collect(value: object) -> None:
        if value is None or type(value) in (str, bool, int, float, bytes) or isinstance(value, (pl.DataFrame, pl.Series)) or (type(value).__module__ == 'numpy' and type(value).__name__ == 'ndarray' and not object_payload(value)):
            return
        if id(value) in seen:
            return
        seen.add(id(value))
        if type(value) not in (dict, list, tuple, set, frozenset, bytearray, np.ndarray):
            for name in _EFFECTS:
                forbid(getattr(value, name, None))
            config = getattr(value, 'prediction_calibration_config', None)
            if config is not getattr(value, 'prediction_calibration_config', None) and any(callable(getattr(config, name, None)) for name in ('calibration_func', 'threshold_func')):
                raise ValueError('Holdout refuses calibration providers without stable observable binding')
            if config is not None:
                configs.append(config)
                for name in ('calibration_func', 'threshold_func'):
                    callback = getattr(config, name, None)
                    if not same_callback(callback, getattr(config, name, None)):
                        raise ValueError('Holdout refuses fitting callbacks without stable observable binding')
                    forbid(callback)
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
        if event == 'call':
            module, code = frame.f_globals.get('__name__', ''), frame.f_code
            if (module, code.co_name) in (('threading', 'start'), ('concurrent.futures.thread', 'submit')):
                violation = 'Holdout inference attempted thread dispatch'
                raise ValueError(violation)
            fixed_labels = module == 'sklearn.preprocessing._label' and code.co_name == 'fit' and np.array_equal(frame.f_locals.get('y', []), [0, 1])
            entry = observed.get(id(code))
            matched = entry is not None and any(_matches(entry[0], frame, owner, instance, index) for owner, instance, index in entry[1])
            if not fixed_labels and (matched or fitting_call(str(module), getattr(code, 'co_qualname', code.co_name))):
                violation = 'Holdout inference attempted fitting/calibration/threshold search'
                raise ValueError(violation)
        elif event == 'c_call':
            name = getattr(arg, '__name__', None)
            if name == 'start_new_thread' and getattr(arg, '__module__', None) == '_thread':
                violation = 'Holdout inference attempted thread dispatch'
                raise ValueError(violation)
            if name in ('urandom', 'getrandom') and entropy_call(arg):
                violation = 'Holdout inference consumed unowned randomness'
                raise ValueError(violation)
        if callable(previous):
            _ = previous(frame, event, arg)

    try:
        with call_codes(tuple(forbidden)):
            observed = {id(function.__code__): (function, owners) for function, owners in forbidden.items()}
            if isinstance(previous, Profile):
                previous.disable()
            sys.setprofile(profile)
            try:
                yield
            finally:
                if isinstance(previous, Profile):
                    previous.enable()
                else:
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

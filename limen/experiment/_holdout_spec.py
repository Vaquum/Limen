from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from functools import partial
from hashlib import sha256
import inspect
import json
import math
import numbers
import sys
from typing import cast

from limen.experiment._holdout_state import fitted_witness
from limen.experiment._holdout_namespaces import namespace_bindings
from limen.experiment._holdout_calls import call_functions, function_module, native_attribute

__all__ = ['digest', 'encoded', 'normalized']
_LIBRARIES = ('numpy', 'polars', 'pandas', 'sklearn', 'lightgbm', 'scipy')


def normalized(value: object) -> object:
    result: object
    if callable(value):
        callable_value = cast(object, value)
        factory = callable_value
        while isinstance(factory, partial):
            provider = cast(partial[object], factory)
            if inspect.getattr_static(type(provider), '__call__', None) is not partial[object].__call__:
                break
            factory = native_attribute(provider, 'func')
        constructors = tuple(inspect.getattr_static(owner, name, None) for owner, name in ((factory, '__new__'), (factory, '__init__'), (type(factory), '__call__'))) if isinstance(factory, type) else ()
        bindings = namespace_bindings(tuple(function for target in (callable_value, *constructors) for function in call_functions(target)), lambda function: function_module(function, _LIBRARIES).split('.')[0] in sys.stdlib_module_names or function_module(function, _LIBRARIES).split('.')[0] in _LIBRARIES)
        target = cast(object, factory)
        owner: object = target.__self__ if inspect.ismethod(target) else target
        factory_class = owner if isinstance(owner, type) else type(owner)
        classes = (cast(type[object], factory_class), cast(type[object], type(factory_class))) if not inspect.isfunction(target) else ()
        class_state = tuple((owner, {name: inspect.getattr_static(owner, name) for function, _ in bindings for name in function.__code__.co_names if any(name in vars(parent) for parent in owner.__mro__)}) for owner in classes)
        result = {'callable': f'{getattr(callable_value, "__module__", type(callable_value).__module__)}.{getattr(callable_value, "__qualname__", type(callable_value).__qualname__)}',
                  'implementation': sha256(fitted_witness((callable_value, constructors, bindings, class_state))).hexdigest()}
    elif value is None or type(value) in (str, bool, int):
        result = value
    elif isinstance(value, numbers.Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('Holdout specification must be finite')
    elif isinstance(value, (date, datetime)):
        result = value.isoformat()
    elif isinstance(value, Mapping):
        items = cast(Mapping[object, object], value)
        if not all(isinstance(key, str) for key in items):
            raise ValueError('Holdout specification requires string keys')
        result = {str(key): normalized(item) for key, item in items.items()}
    elif isinstance(value, (list, tuple)):
        result = [normalized(item) for item in cast(list[object] | tuple[object, ...], value)]
    elif is_dataclass(value) and not isinstance(value, type) and type(value).__module__.startswith('limen.'):
        result = {field.name: normalized(getattr(value, field.name)) for field in fields(value)}
    else:
        raise ValueError(f'Unsupported holdout specification value: {type(value).__name__}')
    return result


def encoded(value: object) -> bytes:
    return json.dumps(normalized(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value: object) -> str:
    return sha256(encoded(value)).hexdigest()


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
from typing import cast

from limen.experiment._holdout_state import fitted_witness
from limen.experiment._holdout_calls import call_functions, native_attribute

__all__ = ['digest', 'encoded', 'normalized']


def normalized(value: object) -> object:
    result: object
    if value is None or type(value) in (str, bool, int):
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
    elif inspect.ismethod(value) or inspect.isfunction(value) or isinstance(value, (type, partial)):
        callable_value = cast(object, value)
        factory = callable_value
        while isinstance(factory, partial):
            provider = cast(partial[object], factory)
            if inspect.getattr_static(type(provider), '__call__', None) is not partial[object].__call__:
                break
            factory = native_attribute(provider, 'func')
        constructors = tuple(inspect.getattr_static(owner, name, None) for owner, name in ((factory, '__new__'), (factory, '__init__'), (type(factory), '__call__'))) if isinstance(factory, type) else ()
        bindings = tuple((function, {name: function.__globals__[name] for name in function.__code__.co_names if name in function.__globals__}) for target in (callable_value, *constructors) for function in call_functions(target))
        result = {'callable': f'{getattr(callable_value, "__module__", type(callable_value).__module__)}.{getattr(callable_value, "__qualname__", type(callable_value).__qualname__)}',
                  'implementation': sha256(fitted_witness((callable_value, constructors, bindings))).hexdigest()}
    elif is_dataclass(value) and not isinstance(value, type) and type(value).__module__.startswith('limen.'):
        result = {field.name: normalized(getattr(value, field.name)) for field in fields(value)}
    else:
        raise ValueError(f'Unsupported holdout specification value: {type(value).__name__}')
    return result


def encoded(value: object) -> bytes:
    return json.dumps(normalized(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value: object) -> str:
    return sha256(encoded(value)).hexdigest()


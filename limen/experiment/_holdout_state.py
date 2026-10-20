from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from functools import lru_cache, partial
from hashlib import sha256
import inspect
from io import IOBase
import logging
import marshal
import pickle
from types import CodeType, GetSetDescriptorType, ModuleType
from typing import TypeGuard, cast

import numpy as np
import numpy.typing as npt
import polars as pl

__all__ = ['children', 'code_digest', 'fitted_witness', 'object_array', 'restore_iterations']


@lru_cache(maxsize=2048)
def code_digest(code: CodeType) -> str:
    return sha256(marshal.dumps(code)).hexdigest()


def _native_values(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return tuple(tuple[object, ...].__iter__(cast(tuple[object, ...], value)))
    if isinstance(value, list):
        return tuple(list[object].__iter__(cast(list[object], value)))
    if isinstance(value, deque):
        return tuple(deque[object].__iter__(cast(deque[object], value)))
    if isinstance(value, set):
        return tuple(set[object].__iter__(cast(set[object], value)))
    return tuple(frozenset[object].__iter__(cast(frozenset[object], value)))


def _booster(value: object) -> bool:
    return type(value).__module__ == 'lightgbm.basic' and type(value).__name__ == 'Booster'


def object_array(value: object) -> TypeGuard[npt.NDArray[np.generic]]:
    return isinstance(value, np.ndarray) and cast(npt.NDArray[np.generic], value).dtype.hasobject


def children(value: object) -> tuple[object, ...]:
    if isinstance(value, Mapping):
        entries = dict[object, object].items(cast(dict[object, object], value)) if isinstance(value, dict) else cast(Mapping[object, object], value).items()
        return (*tuple(part for pair in entries for part in pair), *_attributes(cast(object, value)).values(), *((value.default_factory,) if isinstance(value, defaultdict) else ()))
    if isinstance(value, (tuple, list, set, frozenset, deque)):
        return (*_native_values(cast(object, value)), *_attributes(cast(object, value)).values())
    if object_array(value):
        return (npt.NDArray[np.generic].tolist(value), *_attributes(cast(object, value)).values())
    if isinstance(value, (type, ModuleType, pl.DataFrame, pl.Series)) or (inspect.isroutine(value) and type(value).__module__ == 'builtins') or _booster(value):
        return ()
    if isinstance(value, partial):
        return (value.func, value.args, value.keywords, _attributes(cast(object, value)))
    return tuple(_attributes(value).values())


def _attributes(value: object) -> dict[str, object]:
    values: dict[str, object] = {}
    native_dict = False
    for cls in type(value).__mro__:
        descriptor = cls.__dict__.get('__dict__')
        if isinstance(descriptor, GetSetDescriptorType):
            values.update(cast(dict[str, object], descriptor.__get__(value, type(value))))
            native_dict = True
    if type(value).__dictoffset__ and not native_dict:
        raise ValueError('Holdout refuses hidden native instance dictionaries')
    for cls in type(value).__mro__:
        declaration = cls.__dict__.get('__slots__', ())
        slots = (declaration,) if isinstance(declaration, str) else declaration
        for slot in slots:
            if slot in ('__dict__', '__weakref__'):
                continue
            name = f'_{cls.__name__.lstrip("_")}{slot}' if slot.startswith('__') and not slot.endswith('__') else slot
            descriptor = cls.__dict__[name]
            if hasattr(value, name):
                values[f'{cls.__module__}.{cls.__qualname__}:{name}'] = descriptor.__get__(value, type(value))
    return values


def fitted_witness(value: object) -> bytes:
    seen: dict[int, tuple[int, object]] = {}

    def node(item: object) -> object:
        if item is None or type(item) in (str, bool, int, float, bytes):
            return item
        if isinstance(item, (ModuleType, type)):
            return ('module', item.__name__) if isinstance(item, ModuleType) else ('class', item.__module__, item.__qualname__)
        if id(item) in seen:
            return ('ref', seen[id(item)][0])
        seen[id(item)] = (len(seen), item)
        return structure(item)

    def structure(item: object) -> object:
        kind = (type(item).__module__, type(item).__qualname__)
        attributes = _attributes(item) if issubclass(type(item), (Mapping, tuple, list, set, frozenset, deque, np.ndarray)) else {}
        state = node(attributes) if attributes else None
        if _booster(item):
            from lightgbm import Booster
            booster = cast(Booster, item)
            result = (kind, booster.model_to_string(num_iteration=-1), booster.num_trees(), booster.current_iteration(), booster.best_iteration)
        elif inspect.ismethod(item):
            result = ('method', node(item.__func__), node(item.__self__))
        elif type(item).__module__ == 'builtins' and (inspect.ismethoddescriptor(item) or inspect.isdatadescriptor(item)):
            result = ('descriptor', node(getattr(item, '__objclass__', None)), getattr(item, '__name__', None))
        elif type(item) is logging.Logger:
            result = ('logger', item.name)
        elif inspect.isbuiltin(item):
            owner = getattr(item, '__self__', None)
            result = ('builtin', item.__module__, item.__qualname__, None if isinstance(owner, ModuleType) else node(owner))
        elif inspect.isfunction(item):
            closure = tuple(cell.cell_contents for cell in item.__closure__ or ())
            result = ('function', item.__module__, item.__qualname__, code_digest(item.__code__), node(item.__defaults__), node(item.__kwdefaults__), node(closure), node(vars(item)))
        elif isinstance(item, partial):
            result = (kind, 'partial', node(item.func), node(item.args), node(item.keywords), node(_attributes(cast(object, item))))
        elif isinstance(item, bytearray):
            result = (kind, bytes(memoryview(item)), node(_attributes(item)))
        elif isinstance(item, (str, int, float, bytes)):
            payload = str.__str__(item) if isinstance(item, str) else int.__int__(item) if isinstance(item, int) else float.__float__(item) if isinstance(item, float) else bytes.__repr__(item)
            result = (kind, payload, node(_attributes(item)))
        elif isinstance(item, Mapping):
            factory = node(item.default_factory) if isinstance(item, defaultdict) else None
            result = (kind, factory, tuple((node(key), node(part)) for key, part in (dict[object, object].items(cast(dict[object, object], item)) if isinstance(item, dict) else cast(Mapping[object, object], item).items())), state)
        elif isinstance(item, (tuple, list, deque)):
            result = (kind, tuple(node(part) for part in _native_values(cast(object, item))), item.maxlen if isinstance(item, deque) else None, state)
        elif isinstance(item, (set, frozenset)):
            result = (kind, tuple(sorted((node(part) for part in _native_values(cast(object, item))), key=pickle.dumps)), state)
        elif object_array(item):
            result = (kind, item.shape, item.dtype.descr, node(npt.NDArray[np.generic].tolist(item)), state)
        elif isinstance(item, np.ndarray):
            native = cast(object, npt.NDArray[np.generic].view(cast(npt.NDArray[np.generic], item), type=np.ndarray))
            result = (kind, pickle.dumps(native, protocol=5), state)
        elif isinstance(item, (pl.DataFrame, pl.Series)):
            result = (kind, pickle.dumps(item, protocol=5), state)
        elif kind[0].split('.')[0] == 'numpy' and kind[1] == '_ArrayFunctionDispatcher':
            result = (kind, node(vars(type(item))['_implementation'].__get__(item, type(item))), node(_attributes(item)))
        elif isinstance(item, IOBase):
            raise ValueError('Holdout refuses opaque external prediction dependencies')
        elif type(item).__dictoffset__ or any('__slots__' in cls.__dict__ for cls in type(item).__mro__):
            if any(base is not object and inspect.isbuiltin(base.__dict__.get('__new__')) for base in type(item).__mro__):
                raise ValueError('Holdout refuses opaque native prediction state')
            result = (kind, node(_attributes(item)))
        else:
            result = (kind, pickle.dumps(item, protocol=5))
        return result
    return pickle.dumps(node(value), protocol=5)


def restore_iterations(source: object, memo: Mapping[int, object]) -> None:
    seen: set[int] = set()

    def visit(value: object) -> None:
        if value is None or type(value) in (str, bool, int, float, bytes) or id(value) in seen:
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



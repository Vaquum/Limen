from __future__ import annotations

from builtins import globals as _native_globals
from collections import defaultdict, deque
from collections.abc import Mapping
from functools import lru_cache, partial
from hashlib import sha256
import inspect
from io import IOBase
import logging
import marshal
import pickle
import typing
from types import CellType, CodeType, GenericAlias, GetSetDescriptorType, MappingProxyType, MemberDescriptorType, MethodWrapperType, ModuleType
from typing import TypeGuard, cast

import numpy as np
import numpy.typing as npt
import polars as pl

from limen.experiment._holdout_calls import native_attribute, logger_attributes

__all__ = ['children', 'code_digest', 'fitted_witness', 'object_payload']


_PARAMETER_TYPES = tuple(getattr(typing, name) for name in ('TypeVar', 'ParamSpec', 'TypeVarTuple', 'ParamSpecArgs', 'ParamSpecKwargs') if hasattr(typing, name))
_FORM_TYPES = (type(typing.Union), type(typing.Literal))
_ATOMIC_TYPES = (str, bool, bytes)
_NO_ATTRIBUTE_TYPES = (dict, list, tuple, set, frozenset, defaultdict, deque, np.ndarray, str, int, float, bytes, bytearray)
_CONTAINER_TYPES: tuple[type[object], ...] = (Mapping, tuple, list, set, frozenset, deque, np.ndarray, np.generic)
# Verify canonical identity rather than assume the interpreter's integer cache.
_CANONICAL_INTS: MappingProxyType[object, int] = MappingProxyType({value: int(str(value)) for value in range(-5, 257) if int(str(value)) is int(str(value))})

def _typing_metadata(value: object) -> dict[str, object] | None:
    if type(value) in _FORM_TYPES:
        return {'name': str(value)}
    return {name: getattr(value, name) for name in ('__name__', '__bound__', '__constraints__', '__covariant__', '__contravariant__', '__infer_variance__', '__default__', '__origin__') if hasattr(value, name)} if type(value) in _PARAMETER_TYPES else None


@lru_cache(maxsize=2048)
def _code_hash(code: CodeType, _identity: int) -> str:
    return sha256(marshal.dumps(code, 2)).hexdigest()


def code_digest(code: CodeType) -> str:
    return _code_hash(code, id(code))

def _native_values(value: object) -> tuple[object, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(tuple[object, ...].__iter__(cast(tuple[object, ...], value))) if isinstance(value, tuple) else tuple(list[object].__iter__(cast(list[object], value)))
    if isinstance(value, deque):
        return tuple(deque[object].__iter__(cast(deque[object], value)))
    if isinstance(value, set):
        return tuple(set[object].__iter__(cast(set[object], value)))
    return tuple(frozenset[object].__iter__(cast(frozenset[object], value)))


def _booster(value: object) -> bool:
    return type(value).__module__ == 'lightgbm.basic' and type(value).__name__ == 'Booster'


def object_payload(value: object) -> TypeGuard[npt.NDArray[np.generic] | np.generic]:
    return isinstance(value, (np.ndarray, np.generic)) and cast(np.dtype[np.generic], native_attribute(cast(object, value), 'dtype', np.ndarray if isinstance(value, np.ndarray) else np.generic)).hasobject


def _cell_parts(value: CellType) -> tuple[object, ...]:
    try:
        return (True, value.cell_contents)
    except ValueError:
        return (False,)


def children(value: object, *, namespace_readers: set[str] | None = None) -> tuple[object, ...]:
    if isinstance(value, Mapping):
        entries = dict[object, object].items(cast(dict[object, object], value)) if isinstance(value, dict) else cast(Mapping[object, object], value).items()
        return (*tuple(part for pair in entries for part in pair), *_attributes(cast(object, value), namespace_readers).values(), *((native_attribute(cast(object, value), 'default_factory', defaultdict),) if isinstance(value, defaultdict) else ()))
    if isinstance(value, (tuple, list, set, frozenset, deque)):
        return (*_native_values(cast(object, value)), *_attributes(cast(object, value), namespace_readers).values())
    if object_payload(value):
        return (npt.NDArray[np.generic].tolist(value) if isinstance(value, np.ndarray) else np.generic.item(value), *_attributes(cast(object, value), namespace_readers).values())
    metadata = _typing_metadata(value)
    if isinstance(value, (partial, property, staticmethod, classmethod, GenericAlias)) or metadata is not None:
        parts: tuple[object, ...]
        if metadata is not None:
            parts = tuple(metadata.values())
        elif isinstance(value, (staticmethod, classmethod)):
            parts = (native_attribute(cast(object, value), '__func__'),)
        elif isinstance(value, GenericAlias):
            parts = tuple(vars(GenericAlias)[name].__get__(value, type(value)) for name in ('__origin__', '__args__', '__unpacked__') if name in vars(GenericAlias))
        elif isinstance(value, property):
            parts = tuple(native_attribute(value, name) for name in ('fget', 'fset', 'fdel', '__doc__'))
        else:
            parts = tuple(native_attribute(cast(object, value), name) for name in ('func', 'args', 'keywords'))
        return (*parts, _attributes(cast(object, value), namespace_readers))
    if isinstance(value, (type, ModuleType, pl.DataFrame, pl.Series)) or (inspect.isroutine(value) and type(value).__module__ == 'builtins') or _booster(value) or type(value) is np.ufunc:
        return ()
    return _cell_parts(value) if type(value) is CellType else tuple(_attributes(value, namespace_readers).values())

def _attributes(value: object, namespace_readers: set[str] | None = None) -> dict[str, object]:
    if type(value) in _FORM_TYPES or type(value) in _NO_ATTRIBUTE_TYPES:
        return {}
    values: dict[str, object] = {}
    for cls in type(value).__mro__:
        descriptor = cls.__dict__.get('__dict__')
        if isinstance(descriptor, GetSetDescriptorType):
            values[f'{cls.__module__}.{cls.__qualname__}:__dict__'] = descriptor.__get__(value, type(value))
    if type(value).__dictoffset__ and not values and _typing_metadata(value) is None:
        raise ValueError('Holdout refuses hidden native instance dictionaries')
    for cls in type(value).__mro__:
        if '__slots__' not in cls.__dict__:
            continue
        for name, descriptor in cls.__dict__.items():
            if not isinstance(descriptor, MemberDescriptorType) or descriptor.__objclass__ is not cls:
                continue
            key = f'{cls.__module__}.{cls.__qualname__}:{name}'
            try:
                values[key] = (True, content := descriptor.__get__(value, type(value)))
                if namespace_readers is not None and content is _native_globals:
                    namespace_readers.add(name)
            except AttributeError:
                values[key] = (False,)
    return values


def fitted_witness(value: object, *, class_aliases: bool = False) -> bytes:
    seen: dict[int, int] = {}
    retained: list[object] = []
    classes: dict[int, type[object]] = {}

    def node(item: object) -> object:
        item_type = type(item)
        if item is None or item_type in _ATOMIC_TYPES or (item_type is int and item is _CANONICAL_INTS.get(item)):
            return item
        if (reference := seen.get(id(item))) is not None:
            return ('ref', reference)
        if class_aliases and item_type.__module__ != 'builtins':
            classes[id(item_type)] = item_type
        seen[id(item)] = len(retained)
        retained.append(item)
        if (module := isinstance(item, ModuleType)) or isinstance(item, type):
            return ('module', item.__name__) if module else ('class', item.__module__, item.__qualname__)
        if item_type is int or item_type is float:
            return ('scalar', item_type.__name__, item)
        kind = (item_type.__module__, item_type.__qualname__, ('ref', reference) if (reference := seen.get(id(item_type))) is not None else node(item_type))
        result: object
        if item_type is dict:
            result = (kind, None, tuple(((key if key is None or type(key) in _ATOMIC_TYPES else ('ref', key_ref) if (key_ref := seen.get(id(key))) is not None else node(key)), (part if part is None or type(part) in _ATOMIC_TYPES else ('ref', part_ref) if (part_ref := seen.get(id(part))) is not None else node(part))) for key, part in cast(dict[object, object], item).items()), None)
        elif item_type is list or item_type is tuple:
            result = (kind, tuple((part if part is None or type(part) in _ATOMIC_TYPES else ('ref', part_ref) if (part_ref := seen.get(id(part))) is not None else node(part)) for part in cast(list[object] | tuple[object, ...], item)), None, None)
        elif item_type is set or item_type is frozenset:
            result = (kind, tuple((part if part is None or type(part) in _ATOMIC_TYPES else ('ref', part_ref) if (part_ref := seen.get(id(part))) is not None else node(part)) for part in cast(set[object] | frozenset[object], item)), None)
        else:
            result = structure(item, item_type, kind)
        return result

    def structure(item: object, item_type: type[object], kind: tuple[str, str, object]) -> object:
        attributes = _attributes(item) if issubclass(item_type, _CONTAINER_TYPES) else {}
        state = node(attributes) if attributes else None
        if inspect.isfunction(item):
            result = ('function', node(item.__module__), item.__qualname__, item.__name__, node(item.__doc__), node(getattr(item, '__type_params__', ())), code_digest(item.__code__), node(item.__defaults__), node(item.__kwdefaults__), node(item.__closure__), node(vars(item)), node(item.__annotations__))
        elif item_type is CellType:
            result = ('cell', *(node(part) for part in children(item)))
        elif _booster(item):
            from lightgbm import Booster
            booster = cast(Booster, item)
            result = (kind, booster.model_to_string(num_iteration=-1), booster.num_trees(), booster.current_iteration(), booster.best_iteration)
        elif inspect.ismethod(item):
            result = ('method', node(item.__func__), node(item.__self__))
        elif isinstance(item, (staticmethod, classmethod)):
            result = (kind, 'descriptor', node(native_attribute(cast(object, item), '__func__')), node(_attributes(cast(object, item))))
        elif isinstance(item, property):
            result = (kind, 'property', *(node(native_attribute(item, name)) for name in ('fget', 'fset', 'fdel', '__doc__')), node(_attributes(cast(object, item))))
        elif item_type.__module__ == 'builtins' and (inspect.ismethoddescriptor(item) or inspect.isdatadescriptor(item)):
            result = ('descriptor', node(getattr(item, '__objclass__', None)), getattr(item, '__name__', None))
        elif type(item) is logging.Logger:
            result = ('logger', item.name, item.level, item.disabled, item.propagate, logging.Logger.getEffectiveLevel(item), item.manager.disable, node(logger_attributes(item)))
        elif inspect.isbuiltin(item) or isinstance(item, MethodWrapperType):
            owner = getattr(item, '__self__', None)
            result = ('builtin', getattr(item, '__module__', None), item.__qualname__, None if isinstance(owner, ModuleType) else node(owner))
        elif (metadata := _typing_metadata(item)) is not None:
            result = (kind, 'typing-parameter', node(metadata), node(_attributes(item)))
        elif isinstance(item, GenericAlias):
            metadata = {name: vars(GenericAlias)[name].__get__(item, item_type) for name in ('__origin__', '__args__', '__unpacked__') if name in vars(GenericAlias)}
            result = (kind, 'generic-alias', node(metadata), node(_attributes(cast(object, item))))
        elif isinstance(item, partial):
            result = (kind, 'partial', *(node(native_attribute(cast(object, item), name)) for name in ('func', 'args', 'keywords')), node(_attributes(cast(object, item))))
        elif isinstance(item, bytearray):
            result = (kind, bytes(memoryview(item)), node(_attributes(item)))
        elif isinstance(item, (str, int, float, bytes)):
            payload = str.__str__(item) if isinstance(item, str) else int.__int__(item) if isinstance(item, int) else float.__float__(item) if isinstance(item, float) else bytes.__repr__(item)
            result = (kind, payload, node(_attributes(item)))
        elif isinstance(item, Mapping):
            factory = node(native_attribute(cast(object, item), 'default_factory', defaultdict)) if isinstance(item, defaultdict) else None
            result = (kind, factory, tuple((node(key), node(part)) for key, part in (dict[object, object].items(cast(dict[object, object], item)) if isinstance(item, dict) else cast(Mapping[object, object], item).items())), state)
        elif isinstance(item, (tuple, list, deque)):
            result = (kind, tuple(node(part) for part in _native_values(cast(object, item))), native_attribute(cast(object, item), 'maxlen', deque) if isinstance(item, deque) else None, state)
        elif isinstance(item, (set, frozenset)):
            result = (kind, tuple(node(part) for part in _native_values(cast(object, item))), state)
        elif object_payload(item):
            payload_owner: type[object] = np.ndarray if isinstance(item, np.ndarray) else np.generic
            result = (kind, native_attribute(cast(object, item), 'shape', payload_owner), cast(np.dtype[np.generic], native_attribute(cast(object, item), 'dtype', payload_owner)).descr, node(npt.NDArray[np.generic].tolist(item) if isinstance(item, np.ndarray) else np.generic.item(item)), state)
        elif isinstance(item, np.ndarray):
            native = cast(object, npt.NDArray[np.generic].view(cast(npt.NDArray[np.generic], item), type=np.ndarray))
            result = (kind, pickle.dumps(native, protocol=5), state)
        elif isinstance(item, (pl.DataFrame, pl.Series)):
            result = (kind, pickle.dumps(item, protocol=5), state)
        elif kind[0].split('.')[0] == 'numpy' and kind[1] == '_ArrayFunctionDispatcher':
            result = (kind, node(vars(item_type)['_implementation'].__get__(item, item_type)), node(_attributes(item)))
        elif item_type is np.ufunc:
            result = (kind, pickle.dumps(item, protocol=5))
        elif isinstance(item, IOBase):
            raise ValueError('Holdout refuses opaque external prediction dependencies')
        elif item_type.__dictoffset__ or any('__slots__' in cls.__dict__ for cls in item_type.__mro__):
            if any(base is not object and inspect.isbuiltin(base.__dict__.get('__new__')) for base in item_type.__mro__):
                raise ValueError(f'Holdout refuses opaque native prediction state: {kind[0]}.{kind[1]}')
            result = (kind, node(_attributes(item)))
        else:
            result = (kind, pickle.dumps(item, protocol=5))
        return result
    try:
        result = node(value)
        aliases = tuple((cls.__module__, cls.__qualname__, parent.__module__, parent.__qualname__, name, seen[id(part)]) for cls in classes.values() for parent in cls.__mro__ for name, part in vars(parent).items() if id(part) in seen)
        return pickle.dumps((result, aliases) if class_aliases else result, protocol=5)
    finally:
        for cell in tuple(cast(tuple[CellType, ...], node.__closure__)[node.__code__.co_freevars.index(name)] for name in ('node', 'structure')):
            cell.cell_contents = None

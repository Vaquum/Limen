from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import partial
from hashlib import sha256
import inspect
from io import IOBase
import dis
import marshal
import pickle
import sys
from types import ModuleType
from typing import cast

import numpy as np
import polars as pl

__all__ = ['children', 'dependency_digest', 'fitted_witness', 'inference_functions', 'restore_iterations']


def _booster(value: object) -> bool:
    return type(value).__module__ == 'lightgbm.basic' and type(value).__name__ == 'Booster'


def children(value: object) -> tuple[object, ...]:
    if isinstance(value, Mapping):
        return tuple(cast(Mapping[object, object], value).values())
    if isinstance(value, (tuple, list)):
        return tuple(cast(Sequence[object], value))
    if isinstance(value, (type, ModuleType, np.ndarray, pl.DataFrame, pl.Series)) or inspect.isroutine(value) or _booster(value):
        return ()
    if isinstance(value, partial):
        return (value.func, value.args, value.keywords, vars(value))
    return tuple(_attributes(value).values())


def _attributes(value: object) -> dict[str, object]:
    slots = [slot for cls in type(value).__mro__ for slot in cls.__dict__.get('__slots__', ()) if slot not in ('__dict__', '__weakref__')]
    values = cast(dict[str, object], getattr(value, '__dict__', {}))
    if slots:
        values = {**values, **{slot: getattr(value, slot) for slot in slots if hasattr(value, slot)}}
    return values


def fitted_witness(value: object) -> bytes:
    seen: dict[int, int] = {}
    retained: list[object] = []

    def node(item: object) -> object:
        if item is None or type(item) in (str, bool, int, float, bytes):
            result = item
        elif isinstance(item, ModuleType):
            result = ('module', item.__name__)
        elif isinstance(item, type):
            result = ('class', item.__module__, item.__qualname__)
        elif id(item) in seen:
            result = ('ref', seen[id(item)])
        else:
            seen[id(item)] = len(seen)
            retained.append(item)
            result = structure(item)
        return result

    def structure(item: object) -> object:
        kind = (type(item).__module__, type(item).__qualname__)
        if _booster(item):
            from lightgbm import Booster
            booster = cast(Booster, item)
            result = (kind, booster.model_to_string(num_iteration=-1), booster.num_trees(), booster.current_iteration(), booster.best_iteration)
        elif inspect.ismethod(item):
            result = ('method', node(item.__func__), node(item.__self__))
        elif inspect.isbuiltin(item):
            owner = getattr(item, '__self__', None)
            result = ('builtin', item.__module__, item.__qualname__, None if isinstance(owner, ModuleType) else node(owner))
        elif inspect.isfunction(item):
            closure = tuple(cell.cell_contents for cell in item.__closure__ or ())
            result = ('function', item.__module__, item.__qualname__, marshal.dumps(item.__code__), node(item.__defaults__), node(item.__kwdefaults__), node(closure))
        elif isinstance(item, partial):
            result = ('partial', node(item.func), node(item.args), node(item.keywords), node(vars(item)))
        elif isinstance(item, Mapping):
            factory = node(item.default_factory) if isinstance(item, defaultdict) else None
            result = (kind, factory, tuple((node(key), node(part)) for key, part in cast(Mapping[object, object], item).items()))
        elif isinstance(item, (tuple, list)):
            result = (kind, tuple(node(part) for part in cast(Sequence[object], item)))
        elif isinstance(item, (np.ndarray, pl.DataFrame, pl.Series)):
            result = (kind, pickle.dumps(item, protocol=5))
        elif isinstance(item, IOBase):
            raise ValueError('Holdout refuses opaque external prediction dependencies')
        elif hasattr(item, '__dict__') or any('__slots__' in cls.__dict__ for cls in type(item).__mro__):
            result = (kind, node(_attributes(item)))
        else:
            result = (kind, pickle.dumps(item, protocol=5))
        return result
    return pickle.dumps(node(value), protocol=5)


def restore_iterations(source: object, memo: Mapping[int, object]) -> None:
    seen: set[int] = set()

    def visit(value: object) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        if _booster(value):
            from lightgbm import Booster
            original = cast(Booster, value)
            copied = cast(Booster, memo[id(value)])
            copied.best_iteration = original.best_iteration
        for part in children(value):
            visit(part)

    visit(source)


def inference_functions(values: Sequence[object]) -> tuple[list[object], list[object]]:
    seen: set[int] = set()
    evidence: list[object] = []
    functions: list[object] = []

    def visit(value: object) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        module = getattr(value, '__module__', type(value).__module__)
        if callable(value) and (module or type(value).__module__).startswith(('numpy.random', 'random')):
            functions.append(value)
        if inspect.ismethod(value):
            visit(value.__func__)
            visit(value.__self__)
        elif isinstance(value, partial):
            visit(value.func)
            visit(value.args)
            visit(value.keywords)
        elif inspect.isfunction(value):
            functions.append(value)
            evidence.append(fitted_witness(value))
            if value.__module__.split('.')[0] in sys.stdlib_module_names or value.__module__.startswith(('numpy.', 'polars.', 'sklearn.', 'lightgbm.', 'scipy.')):
                return
            visit(value.__defaults__)
            visit(value.__kwdefaults__)
            for cell in value.__closure__ or ():
                visit(cell.cell_contents)
            instructions = list(dis.get_instructions(value))
            for index, instruction in enumerate(instructions):
                if instruction.argval == 'fit_calibrator':
                    continue
                if instruction.opname != 'LOAD_GLOBAL' or instruction.argval not in value.__globals__:
                    continue
                item = cast(object, value.__globals__[instruction.argval])
                for attribute in instructions[index + 1:]:
                    if attribute.opname not in ('LOAD_ATTR', 'LOAD_METHOD'):
                        break
                    if not hasattr(item, str(attribute.argval)):
                        break
                    item = getattr(item, str(attribute.argval))
                if not isinstance(item, ModuleType):
                    evidence.append((value.__module__, value.__qualname__, instruction.argval, fitted_witness(item)))
                    visit(item)
        elif not isinstance(value, (ModuleType, type, np.ndarray, pl.DataFrame, pl.Series)) and not _booster(value):
            pending = ['predict', 'predict_proba', 'transform', '__call__']
            methods: set[str] = set()
            while pending:
                name = pending.pop()
                if name in methods:
                    continue
                methods.add(name)
                method = inspect.getattr_static(type(value), name, None)
                if isinstance(method, property):
                    method = method.fget
                if isinstance(method, (staticmethod, classmethod)):
                    method = method.__func__
                if inspect.isfunction(method):
                    visit(method)
                    pending.extend(method.__code__.co_names)
                elif method is not None and not inspect.isdatadescriptor(method) and not isinstance(method, type):
                    evidence.append((type(value).__module__, type(value).__qualname__, name, fitted_witness(method)))
                    visit(method)
            for part in children(value):
                visit(part)

    for value in values:
        visit(value)
    return functions, evidence


def dependency_digest(values: Sequence[object]) -> str:
    return sha256(pickle.dumps(inference_functions(values)[1], protocol=5)).hexdigest()

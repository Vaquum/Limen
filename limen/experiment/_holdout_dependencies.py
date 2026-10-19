from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import partial
from hashlib import sha256
import inspect
import marshal
import pickle
from types import ModuleType
from typing import cast

import numpy as np
import polars as pl

__all__ = ['children', 'dependency_digest', 'fitted_witness', 'restore_iterations']


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
    return tuple(cast(dict[str, object], vars(value)).values()) if hasattr(value, '__dict__') else ()


def fitted_witness(value: object) -> bytes:
    seen: dict[int, int] = {}

    def node(item: object) -> object:
        if item is None or type(item) in (str, bool, int, float, bytes):
            return item
        if isinstance(item, ModuleType):
            return ('module', item.__name__)
        if isinstance(item, type):
            return ('class', item.__module__, item.__qualname__)
        if id(item) in seen:
            return ('ref', seen[id(item)])
        seen[id(item)] = len(seen)
        kind = (type(item).__module__, type(item).__qualname__)
        if _booster(item):
            from lightgbm import Booster
            booster = cast(Booster, item)
            return (kind, booster.model_to_string(num_iteration=-1), booster.num_trees(), booster.current_iteration(), booster.best_iteration)
        if inspect.ismethod(item):
            return ('method', node(item.__func__), node(item.__self__))
        if inspect.isfunction(item):
            closure = tuple(cell.cell_contents for cell in item.__closure__ or ())
            return ('function', item.__module__, item.__qualname__, marshal.dumps(item.__code__), node(item.__defaults__), node(item.__kwdefaults__), node(closure))
        if isinstance(item, partial):
            return ('partial', node(item.func), node(item.args), node(item.keywords), node(vars(item)))
        if isinstance(item, Mapping):
            factory = node(item.default_factory) if isinstance(item, defaultdict) else None
            return (kind, factory, tuple((node(key), node(part)) for key, part in cast(Mapping[object, object], item).items()))
        if isinstance(item, (tuple, list)):
            return (kind, tuple(node(part) for part in cast(Sequence[object], item)))
        if isinstance(item, (np.ndarray, pl.DataFrame, pl.Series)):
            return (kind, pickle.dumps(item, protocol=5))
        if hasattr(item, '__dict__'):
            return (kind, node(vars(item)))
        return (kind, pickle.dumps(item, protocol=5))

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


def dependency_digest(values: Sequence[object]) -> str:
    functions: set[int] = set()
    objects: set[int] = set()
    evidence: list[object] = []

    def function(value: object) -> None:
        if inspect.ismethod(value):
            value = value.__func__
        if isinstance(value, (staticmethod, classmethod)):
            value = value.__func__
        if not inspect.isfunction(value) or id(value) in functions:
            return
        functions.add(id(value))
        evidence.append(fitted_witness(value))
        for name in value.__code__.co_names:
            if name not in value.__globals__:
                continue
            item = cast(object, value.__globals__[name])
            if inspect.isfunction(item):
                function(item)
            elif isinstance(item, (Mapping, list, tuple, set, np.ndarray)) or item is None or type(item) in (str, bool, int, float):
                evidence.append((value.__module__, value.__qualname__, name, fitted_witness(item)))

    def visit(value: object) -> None:
        if id(value) in objects:
            return
        objects.add(id(value))
        function(value)
        if not isinstance(value, (type, ModuleType, np.ndarray, pl.DataFrame, pl.Series)) and not _booster(value):
            for name in ('predict', 'predict_proba', 'transform'):
                function(inspect.getattr_static(type(value), name, None))
        for part in children(value):
            visit(part)

    for value in values:
        visit(value)
    return sha256(pickle.dumps(evidence, protocol=5)).hexdigest()

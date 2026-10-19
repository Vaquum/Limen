from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import lru_cache, partial
from hashlib import sha256
import inspect
from io import IOBase
import dis
import marshal
import logging
import pickle
import sys
from types import CodeType, ModuleType
from typing import cast

import numpy as np
import polars as pl

__all__ = ['FITTING_METHODS', 'children', 'code_digest', 'dependency_digest', 'fitted_witness', 'fitting_call', 'inference_functions', 'restore_iterations']


FITTING_METHODS = frozenset({'fit', 'fit_transform', 'partial_fit', 'train', 'fit_calibrator', 'grid_threshold_optimizer'})

def fitting_call(module: str, name: str) -> bool:
    return (name.rsplit('.', 1)[-1] in FITTING_METHODS
            or f'{module}.{name}' in ('limen.experiment._shim_fractional_diff.fit_order', 'limen.features.fractional_diff.find_min_d')
            or (module.startswith(('limen.scalers.', 'limen.targets.')) and name.rsplit('.', 1)[-1] == '__init__'))


@lru_cache(maxsize=2048)
def code_digest(code: CodeType) -> str:
    return sha256(marshal.dumps(code)).hexdigest()


@lru_cache(maxsize=2048)
def _instructions(code: CodeType) -> tuple[dis.Instruction, ...]:
    nested = [part for part in code.co_consts if isinstance(part, CodeType) and part.co_name not in FITTING_METHODS]
    return (*dis.get_instructions(code), *(instruction for part in nested for instruction in _instructions(part)))


def _booster(value: object) -> bool:
    return type(value).__module__ == 'lightgbm.basic' and type(value).__name__ == 'Booster'


def children(value: object) -> tuple[object, ...]:
    if isinstance(value, Mapping):
        return tuple(part for pair in cast(Mapping[object, object], value).items() for part in pair)
    if isinstance(value, (tuple, list)):
        return tuple(cast(Sequence[object], value))
    if isinstance(value, (type, ModuleType, np.ndarray, pl.DataFrame, pl.Series)) or inspect.isroutine(value) or _booster(value):
        return ()
    if isinstance(value, partial):
        return (value.func, value.args, value.keywords, vars(value))
    return tuple(_attributes(value).values())


def _attributes(value: object) -> dict[str, object]:
    values = dict(cast(dict[str, object], getattr(value, '__dict__', {})))
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
        elif inspect.ismethoddescriptor(item) or inspect.isdatadescriptor(item):
            result = ('descriptor', node(getattr(item, '__objclass__', None)), getattr(item, '__name__', None))
        elif isinstance(item, logging.Logger):
            result = ('logger', item.name)
        elif inspect.isbuiltin(item):
            owner = getattr(item, '__self__', None)
            result = ('builtin', item.__module__, item.__qualname__, None if isinstance(owner, ModuleType) else node(owner))
        elif inspect.isfunction(item):
            closure = tuple(cell.cell_contents for cell in item.__closure__ or ())
            result = ('function', item.__module__, item.__qualname__, code_digest(item.__code__), node(item.__defaults__), node(item.__kwdefaults__), node(closure))
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
        if value is None or isinstance(value, (str, bool, int, float, bytes, np.generic)):
            return
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


def inference_functions(values: Sequence[object], *, exclude: Sequence[object] = (), resolved: Sequence[object] = ()) -> tuple[list[object], list[object]]:
    seen: dict[int, object] = {}
    evidence: list[object] = []
    functions: list[object] = []
    omitted = {id(value) for value in exclude}
    sealed = {id(value) for value in resolved}
    attributes: set[str] = set()
    objects: list[object] = []
    processed: dict[int, set[str]] = {}

    def visit(value: object) -> None:
        if value is None or isinstance(value, (str, bool, int, float, bytes, np.generic)):
            return
        if type(value) is logging.Logger:
            evidence.append(fitted_witness(value))
            return
        if id(value) in seen or id(value) in omitted:
            return
        seen[id(value)] = value
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
            if fitting_call(value.__module__, value.__qualname__) or id(value) in sealed:
                return
            attributes.update(value.__code__.co_names)
            if value.__module__.split('.')[0] in sys.stdlib_module_names or value.__module__.startswith(('numpy.', 'polars.', 'sklearn.', 'lightgbm.', 'scipy.')):
                return
            visit(value.__defaults__)
            visit(value.__kwdefaults__)
            for cell in value.__closure__ or ():
                visit(cell.cell_contents)
            instructions = _instructions(value.__code__)
            references = dict(value.__globals__)
            references.update(zip(value.__code__.co_freevars, (cell.cell_contents for cell in value.__closure__ or ()), strict=True))
            imported: ModuleType | None = None
            chained_import = False
            binding: object = None
            for index, instruction in enumerate(instructions):
                name = str(instruction.argval)
                if instruction.opname in ('LOAD_ATTR', 'LOAD_METHOD'):
                    attributes.add(name)
                if instruction.opname == 'IMPORT_NAME':
                    chained_import = instructions[index - 1].argval is None
                    imported = sys.modules.get(name.split('.', 1)[0] if chained_import else name)
                    if imported is None:
                        raise ValueError(f'Holdout prediction import lacks loaded evidence: {name}')
                    binding = imported
                elif instruction.opname == 'IMPORT_FROM' and imported is not None:
                    binding = getattr(imported, name)
                    if chained_import and isinstance(binding, ModuleType):
                        imported = binding
                elif instruction.opname in ('STORE_FAST', 'STORE_DEREF') and binding is not None:
                    references[name], binding = binding, None
                if name == 'fit_calibrator' or instruction.opname not in ('LOAD_GLOBAL', 'LOAD_DEREF', 'LOAD_FAST') or name not in references:
                    continue
                available = value.__globals__ if instruction.opname == 'LOAD_GLOBAL' else references
                if name not in available:
                    continue
                item = cast(object, available[name])
                following = index + 1
                for attribute in instructions[following:]:
                    if attribute.opname not in ('LOAD_ATTR', 'LOAD_METHOD'):
                        break
                    if not hasattr(item, str(attribute.argval)):
                        break
                    item = getattr(item, str(attribute.argval))
                    following += 1
                if following < len(instructions) and instructions[following].opname in ('STORE_FAST', 'STORE_DEREF'):
                    references[str(instructions[following].argval)] = item
                if not isinstance(item, ModuleType):
                    try:
                        witness = fitted_witness(item)
                        evidence.append((value.__module__, value.__qualname__, instruction.argval, witness))
                        visit(item)
                    except (ValueError, TypeError, pickle.PickleError) as exc:
                        raise ValueError(f'Holdout dependency {value.__module__}.{value.__qualname__}:{name}: {exc}') from exc
        elif inspect.isroutine(value):
            evidence.append(fitted_witness(value))
        elif not isinstance(value, (ModuleType, np.ndarray, pl.DataFrame, pl.Series)) and not _booster(value):
            objects.append(value)
            methods(value)
            if not isinstance(value, type):
                for part in children(value):
                    visit(part)

    def methods(value: object) -> None:
        cls = value if isinstance(value, type) else type(value)
        pending = ['predict', 'predict_proba', 'transform', '__call__', '__getattr__', '__getattribute__']
        if isinstance(value, type):
            pending.extend(('__new__', '__init__'))
        pending.extend(name for parent in cls.__mro__ for name in parent.__dict__ if name in attributes)
        done = processed.setdefault(id(value), set())
        while pending:
            name = pending.pop()
            if name in done or fitting_call(cls.__module__, f'{cls.__qualname__}.{name}'):
                continue
            done.add(name)
            method = inspect.getattr_static(cls, name, None)
            if method is None:
                continue
            evidence.append((cls.__module__, cls.__qualname__, name, type(method).__module__, type(method).__qualname__))
            if isinstance(method, property):
                method = method.fget
            if isinstance(method, (staticmethod, classmethod)):
                method = method.__func__
            if inspect.isfunction(method):
                evidence.append((cls.__module__, cls.__qualname__, name, fitted_witness(method)))
                visit(method)
                if not fitting_call(method.__module__, method.__qualname__):
                    pending.extend(method.__code__.co_names)
            elif method is not None and not inspect.isdatadescriptor(method):
                evidence.append((cls.__module__, cls.__qualname__, name, fitted_witness(method)))
                visit(method)

    for value in values:
        visit(value)
    previous = (-1, -1)
    while previous != (len(attributes), len(objects)):
        previous = (len(attributes), len(objects))
        for value in tuple(objects):
            methods(value)
    return functions, evidence


def dependency_digest(values: Sequence[object], *, exclude: Sequence[object] = (), resolved: Sequence[object] = ()) -> str:
    return sha256(pickle.dumps(inference_functions(values, exclude=exclude, resolved=resolved)[1], protocol=5)).hexdigest()

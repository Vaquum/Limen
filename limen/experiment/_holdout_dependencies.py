from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Collection, Mapping, Sequence
from functools import lru_cache, partial
from hashlib import sha256
import inspect
from importlib.util import resolve_name
from io import IOBase
import dis
import marshal
import logging
import pickle
import sys
from types import CodeType, ModuleType
from typing import TypeGuard, cast

import numpy as np
import numpy.typing as npt
import polars as pl

__all__ = ['FITTING_METHODS', 'children', 'code_digest', 'dependency_digest', 'fitted_witness', 'fitting_call', 'inference_evidence', 'object_array']


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


def object_array(value: object) -> TypeGuard[npt.NDArray[np.generic]]:
    return isinstance(value, np.ndarray) and cast(npt.NDArray[np.generic], value).dtype.hasobject


def children(value: object) -> tuple[object, ...]:
    if isinstance(value, Mapping):
        return (*tuple(part for pair in cast(Mapping[object, object], value).items() for part in pair), *_attributes(cast(object, value)).values(), *((value.default_factory,) if isinstance(value, defaultdict) else ()))
    if isinstance(value, (tuple, list, set, frozenset, deque)):
        return (*cast(Collection[object], value), *_attributes(cast(object, value)).values())
    if object_array(value):
        return (value.tolist(), *_attributes(cast(object, value)).values())
    if isinstance(value, (type, ModuleType, pl.DataFrame, pl.Series)) or (inspect.isroutine(value) and type(value).__module__ == 'builtins') or _booster(value):
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
    seen: dict[int, tuple[int, object]] = {}

    def node(item: object) -> object:
        if item is None or type(item) in (str, bool, int, float, bytes):
            result = item
        elif isinstance(item, (ModuleType, type)):
            result = ('module', item.__name__) if isinstance(item, ModuleType) else ('class', item.__module__, item.__qualname__)
        elif id(item) in seen:
            result = ('ref', seen[id(item)][0])
        else:
            seen[id(item)] = (len(seen), item)
            result = structure(item)
        return result

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
            result = (kind, 'partial', node(item.func), node(item.args), node(item.keywords), node(vars(item)))
        elif isinstance(item, bytearray):
            result = (kind, bytes(memoryview(item)), node(_attributes(item)))
        elif isinstance(item, (str, int, float, bytes)):
            payload = str.__str__(item) if isinstance(item, str) else int.__int__(item) if isinstance(item, int) else float.__float__(item) if isinstance(item, float) else bytes.__repr__(item)
            result = (kind, payload, node(_attributes(item)))
        elif isinstance(item, Mapping):
            factory = node(item.default_factory) if isinstance(item, defaultdict) else None
            result = (kind, factory, tuple((node(key), node(part)) for key, part in cast(Mapping[object, object], item).items()), state)
        elif isinstance(item, (tuple, list, deque)):
            result = (kind, tuple(node(part) for part in cast(Collection[object], item)), item.maxlen if isinstance(item, deque) else None, state)
        elif isinstance(item, (set, frozenset)):
            result = (kind, tuple(sorted((node(part) for part in cast(Collection[object], item)), key=pickle.dumps)), state)
        elif object_array(item):
            result = (kind, item.shape, item.dtype.descr, node(item.tolist()), state)
        elif isinstance(item, (np.ndarray, pl.DataFrame, pl.Series)):
            result = (kind, pickle.dumps(item, protocol=5), state)
        elif kind[0].split('.')[0] == 'numpy' and kind[1] == '_ArrayFunctionDispatcher':
            result = (kind, node(vars(type(item))['_implementation'].__get__(item, type(item))), node(_attributes(item)))
        elif isinstance(item, IOBase):
            raise ValueError('Holdout refuses opaque external prediction dependencies')
        elif isinstance(getattr(item, '__dict__', None), dict) or any('__slots__' in cls.__dict__ for cls in type(item).__mro__):
            if any(base is not object and inspect.isbuiltin(base.__dict__.get('__new__')) for base in type(item).__mro__):
                raise ValueError('Holdout refuses opaque native prediction state')
            result = (kind, node(_attributes(item)))
        else:
            result = (kind, pickle.dumps(item, protocol=5))
        return result
    return pickle.dumps(node(value), protocol=5)


def inference_evidence(values: Sequence[object], *, exclude: Sequence[object] = (), resolved: Sequence[object] = ()) -> list[object]:
    seen: dict[int, object] = {}
    evidence: list[object] = []
    omitted, sealed = ({id(value) for value in exclude}, {id(value) for value in resolved})
    function_witness = lru_cache(maxsize=None)(fitted_witness)
    attributes: set[str] = set()
    objects: list[object] = []
    processed: dict[int, set[str]] = {}

    def visit(value: object) -> None:
        if value is None or type(value) in (str, bool, int, float, bytes) or isinstance(value, np.generic):
            return
        if type(value) is logging.Logger:
            evidence.append(fitted_witness(value))
            return
        if id(value) in seen or id(value) in omitted:
            return
        seen[id(value)] = value
        modules = (getattr(value, '__module__', None) or type(value).__module__, type(getattr(value, '__self__', None)).__module__, getattr(getattr(value, '__objclass__', None), '__module__', ''))
        if callable(value) and any(isinstance(module, str) and module.startswith(('numpy.random', 'random')) for module in modules):
            raise ValueError('Holdout inference has an unowned random dependency')
        if inspect.ismethod(value):
            visit(value.__func__)
            visit(value.__self__)
        elif isinstance(value, partial):
            provider = cast(object, value)
            for part in children(provider):
                visit(part)
            if type(provider) is not partial:
                objects.append(provider)
        elif inspect.isfunction(value):
            evidence.append(function_witness(value))
            if fitting_call(value.__module__, value.__qualname__) or id(value) in sealed:
                return
            attributes.update(value.__code__.co_names)
            visit(vars(value))
            if value.__module__.split('.')[0] in sys.stdlib_module_names or value.__module__.split('.')[0] in ('numpy', 'polars', 'sklearn', 'lightgbm', 'scipy'):
                return
            if any(name in value.__code__.co_names for name in ('random', 'default_rng', 'RandomState', 'Generator')):
                raise ValueError('Holdout inference has an unowned random dependency')
            visit(value.__defaults__)
            visit(value.__kwdefaults__)
            for cell in value.__closure__ or ():
                visit(cell.cell_contents)
            instructions = _instructions(value.__code__)
            references = {name: [part] for name, part in value.__globals__.items()}
            references.update((name, [cell.cell_contents]) for name, cell in zip(value.__code__.co_freevars, value.__closure__ or (), strict=True))
            imported: ModuleType | None = None
            chained_import = False
            binding: object = None
            for index, instruction in enumerate(instructions):
                name = str(instruction.argval)
                if instruction.opname in ('LOAD_ATTR', 'LOAD_METHOD') or (instruction.opname == 'LOAD_CONST' and isinstance(instruction.argval, str)):
                    attributes.add(name)
                if instruction.opname == 'IMPORT_NAME':
                    chained_import = instructions[index - 1].argval is None
                    level = int(instructions[index - 2].argval)
                    module_name = resolve_name('.' * level + name, value.__globals__.get('__package__', '')) if level else name
                    imported = sys.modules.get(module_name.split('.', 1)[0] if chained_import else module_name)
                    if imported is None:
                        raise ValueError(f'Holdout prediction import lacks loaded evidence: {name}')
                    binding = imported
                elif instruction.opname == 'IMPORT_FROM' and imported is not None:
                    binding = getattr(imported, name)
                    if chained_import and isinstance(binding, ModuleType):
                        imported = binding
                elif instruction.opname in ('STORE_FAST', 'STORE_DEREF', 'STORE_NAME') and binding is not None:
                    references.setdefault(name, []).append(binding)
                    binding = None
                if name == 'fit_calibrator' or instruction.opname not in ('LOAD_GLOBAL', 'LOAD_NAME', 'LOAD_DEREF', 'LOAD_FAST'):
                    continue
                for candidate in tuple(references.get(name, ())):
                    item = cast(object, candidate)
                    following = index + 1
                    for attribute in instructions[following:]:
                        if attribute.opname not in ('LOAD_ATTR', 'LOAD_METHOD') or not hasattr(item, str(attribute.argval)):
                            break
                        item = getattr(item, str(attribute.argval))
                        following += 1
                    if isinstance(item, ModuleType) and index > 0 and instructions[index - 1].argval == 'getattr':
                        for attribute in attributes.intersection(vars(item)):
                            visit(getattr(item, attribute))
                    if following < len(instructions) and instructions[following].opname in ('STORE_FAST', 'STORE_DEREF', 'STORE_NAME'):
                        aliases = references.setdefault(str(instructions[following].argval), [])
                        if all(part is not item for part in aliases):
                            aliases.append(item)
                    if not isinstance(item, ModuleType):
                        try:
                            visit(item)
                            witness = function_witness(item) if inspect.isfunction(item) else fitted_witness(item)
                            evidence.append((value.__module__, value.__qualname__, instruction.argval, witness))
                        except (ValueError, TypeError, pickle.PickleError) as exc:
                            raise ValueError(f'Holdout dependency {value.__module__}.{value.__qualname__}:{name}: {exc}') from exc
        elif inspect.isroutine(value) and type(value).__module__ == 'builtins':
            evidence.append(fitted_witness(value))
        elif object_array(value):
            visit(value.tolist())
            visit(_attributes(value))
            if type(value) is not np.ndarray:
                objects.append(value)
        elif not isinstance(value, (ModuleType, pl.DataFrame, pl.Series)) and type(value) is not np.ndarray:
            if isinstance(value, type) and type(value) is not type:
                visit(type(value))
            if type(value) not in (dict, list, tuple, set, frozenset, defaultdict):
                objects.append(value)
            if not isinstance(value, type):
                for part in children(value):
                    visit(part)

    def methods(value: object) -> None:
        cls = value if isinstance(value, type) else type(value)
        names = {'predict', 'predict_proba', 'transform', '__call__', '__get__', '__getattr__', '__getattribute__', '__getitem__', '__iter__', '__len__', '__contains__'} | attributes
        names.update(f'__{prefix}{operator}__' for prefix in ('', 'r', 'i') for operator in ('add', 'sub', 'mul', 'matmul', 'truediv', 'floordiv', 'mod', 'pow', 'lshift', 'rshift', 'and', 'xor', 'or'))
        if isinstance(value, type):
            names.update(('__new__', '__init__'))
        done = processed.setdefault(id(value), set())
        for parent in cls.__mro__:
            for name in names.intersection(parent.__dict__):
                key = f'{parent.__module__}.{parent.__qualname__}:{name}'
                if key in done or fitting_call(parent.__module__, f'{parent.__qualname__}.{name}'):
                    continue
                done.add(key)
                method = parent.__dict__[name]
                evidence.append((key, type(method).__module__, type(method).__qualname__))
                if isinstance(method, property):
                    method = method.fget
                if isinstance(method, (staticmethod, classmethod)):
                    method = method.__func__
                if method is not None:
                    evidence.append((key, function_witness(method) if inspect.isfunction(method) else fitted_witness(method)))
                    visit(method)

    for value in values:
        visit(value)
    previous = (-1, -1)
    while previous != (len(attributes), len(objects)):
        previous = (len(attributes), len(objects))
        for value in tuple(objects):
            methods(value)
    return evidence


def dependency_digest(values: Sequence[object], *, exclude: Sequence[object] = (), resolved: Sequence[object] = ()) -> str:
    return sha256(pickle.dumps(inference_evidence(values, exclude=exclude, resolved=resolved), protocol=5)).hexdigest()

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import lru_cache, partial
from hashlib import sha256
import inspect
from importlib.util import resolve_name
import dis
import logging
import pickle
import sys
from types import CodeType, FunctionType, ModuleType
from typing import cast

import numpy as np
import polars as pl

from limen.experiment._holdout_state import children, code_digest, fitted_witness, object_array

__all__ = ['FITTING_METHODS', 'call_functions', 'children', 'code_digest', 'dependency_digest', 'entropy_call', 'fitted_witness', 'fitting_call', 'inference_evidence', 'object_array', 'restore_iterations']


FITTING_METHODS = frozenset({'fit', 'fit_transform', 'partial_fit', 'train', 'fit_calibrator', 'grid_threshold_optimizer'})

def fitting_call(module: str, name: str) -> bool:
    return (name.rsplit('.', 1)[-1] in FITTING_METHODS
            or f'{module}.{name}' in ('limen.experiment._shim_fractional_diff.fit_order', 'limen.features.fractional_diff.find_min_d')
            or (module.startswith(('limen.scalers.', 'limen.targets.')) and name.rsplit('.', 1)[-1] == '__init__'))


def entropy_call(value: object) -> bool:
    name, module = getattr(value, '__name__', None), getattr(value, '__module__', None)
    return isinstance(name, str) and isinstance(module, str) and name in ('urandom', 'getrandom') and module in ('os', 'posix', 'nt')


@lru_cache(maxsize=2048)
def _instructions(code: CodeType) -> tuple[dis.Instruction, ...]:
    return (*dis.get_instructions(code), *(instruction for part in code.co_consts if isinstance(part, CodeType) and part.co_name not in FITTING_METHODS for instruction in _instructions(part)))


def inference_evidence(values: Sequence[object], *, exclude: Sequence[object] = (), resolved: Sequence[object] = (), capture: list[object] | None = None) -> list[object]:
    seen: dict[int, object] = {}
    evidence: list[object] = []
    omitted, sealed = ({id(value) for value in exclude}, {id(value) for value in resolved})
    def witness(identity: int) -> bytes:
        return fitted_witness(seen[identity])
    cached_witness = lru_cache(maxsize=None)(witness)
    attributes: set[str] = {'__float__', '__int__', '__index__', '__bool__', '__complex__', '__str__', '__bytes__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__neg__', '__pos__', '__abs__', '__invert__', '__round__', '__trunc__', '__floor__', '__ceil__', '__divmod__', '__rdivmod__', '__hash__', '__format__'} | {f'__{prefix}{operator}__' for prefix in ('', 'r', 'i') for operator in ('add', 'sub', 'mul', 'matmul', 'truediv', 'floordiv', 'mod', 'pow', 'lshift', 'rshift', 'and', 'xor', 'or')}
    module_attributes: set[str] = set()
    objects: list[object] = []
    processed: dict[int, set[str]] = {}

    def visit(value: object) -> None:
        if value is None or type(value) in (str, bool, int, float, bytes) or (type(value).__module__ == 'numpy' and issubclass(type(value), np.generic)):
            if isinstance(value, str):
                attributes.add(value)
            return
        if type(value) is logging.Logger:
            evidence.append(fitted_witness(value))
            return
        if id(value) in seen or id(value) in omitted:
            return
        seen[id(value)] = value
        modules = (getattr(value, '__module__', None) or type(value).__module__, type(getattr(value, '__self__', None)).__module__, getattr(getattr(value, '__objclass__', None), '__module__', ''))
        if callable(value) and (entropy_call(value) or any(isinstance(module, str) and module.startswith(('numpy.random', 'random', 'secrets')) for module in modules)):
            raise ValueError('Holdout inference has an unowned random dependency')
        if inspect.ismethod(value):
            visit(value.__func__)
            visit(value.__self__)
        elif isinstance(value, partial):
            for part in children(cast(object, value)):
                visit(part)
            if type(cast(object, value)) is not partial:
                objects.append(cast(object, value))
        elif inspect.isfunction(value):
            evidence.append(cached_witness(id(value)))
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
                    module_attributes.add(name)
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
                if (name == 'fit_calibrator' and (value.__module__, value.__qualname__) in (('limen.sfd.reference_architecture.logreg_binary', 'LogRegBinary.predict'), ('limen.sfd.reference_architecture.lightgbm_binary', 'LightGBMBinary.predict'))) or instruction.opname not in ('LOAD_GLOBAL', 'LOAD_NAME', 'LOAD_DEREF', 'LOAD_FAST'):
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
                    try:
                        visit(item)
                        witness = cached_witness(id(item)) if id(item) in seen else fitted_witness(item)
                        evidence.append((value.__module__, value.__qualname__, instruction.argval, witness))
                    except (ValueError, TypeError, pickle.PickleError) as exc:
                        raise ValueError(f'Holdout dependency {value.__module__}.{value.__qualname__}:{name}: {exc}') from exc
        elif inspect.isroutine(value) and type(value).__module__ == 'builtins':
            evidence.append(fitted_witness(value))
            owner = getattr(value, '__self__', None)
            if owner is not None and not isinstance(owner, ModuleType):
                visit(owner)
        elif object_array(value):
            for part in children(cast(object, value)):
                visit(part)
            if type(value) is not np.ndarray:
                objects.append(value)
        elif isinstance(value, ModuleType):
            objects.append(value)
        elif not isinstance(value, (pl.DataFrame, pl.Series)) and type(value) is not np.ndarray:
            if isinstance(value, type) and type(value) is not type:
                visit(type(value))
            if type(value) not in (dict, list, tuple, set, frozenset, defaultdict):
                objects.append(value)
            if not isinstance(value, type):
                for part in children(value):
                    visit(part)

    def methods(value: object) -> None:
        if isinstance(value, ModuleType):
            done = processed.setdefault(id(value), set())
            for name in module_attributes.intersection(vars(value)) - done:
                done.add(name)
                part = cast(object, vars(value)[name])
                visit(part)
                evidence.append((value.__name__, name, cached_witness(id(part)) if id(part) in seen else fitted_witness(part)))
            return
        cls = value if isinstance(value, type) else type(value)
        names = {'predict', 'predict_proba', 'transform', '__call__', '__get__', '__getattr__', '__getattribute__', '__getitem__', '__enter__', '__exit__', '__aenter__', '__aexit__', '__await__', '__aiter__', '__anext__', '__iter__', '__next__', '__len__', '__contains__'} | attributes
        if isinstance(value, type):
            names.update(('__new__', '__init__'))
        done = processed.setdefault(id(cls), set())
        for parent in cls.__mro__:
            for name in names.intersection(parent.__dict__):
                key = f'{parent.__module__}.{parent.__qualname__}:{name}'
                if key in done or fitting_call(parent.__module__, f'{parent.__qualname__}.{name}'):
                    continue
                done.add(key)
                method = parent.__dict__[name]
                evidence.append((key, type(method).__module__, type(method).__qualname__))
                if isinstance(method, (property, staticmethod, classmethod)):
                    visit(cast(object, method))
                    evidence.append((key, cached_witness(id(cast(object, method)))))
                if isinstance(method, property):
                    method = method.fget
                if isinstance(method, (staticmethod, classmethod)):
                    method = cast('staticmethod[..., object] | classmethod[object, ..., object]', method).__func__
                if method is not None:
                    visit(method)
                    evidence.append((key, cached_witness(id(method)) if id(method) in seen else fitted_witness(method)))

    for value in values:
        visit(value)
    previous = (-1, -1, -1)
    while previous != (len(attributes), len(objects), len(module_attributes)):
        previous = (len(attributes), len(objects), len(module_attributes))
        for value in {(id(item) if isinstance(item, (type, ModuleType)) else id(type(item)), isinstance(item, type)): item for item in tuple(objects)}.values():
            methods(value)
    if capture is not None:
        capture.extend(seen.values())
    return evidence


def dependency_digest(values: Sequence[object], *, exclude: Sequence[object] = (), resolved: Sequence[object] = (), capture: list[object] | None = None) -> str:
    return sha256(pickle.dumps(inference_evidence(values, exclude=exclude, resolved=resolved, capture=capture), protocol=5)).hexdigest()


def call_functions(value: object) -> tuple[FunctionType, ...]:
    if isinstance(value, (staticmethod, classmethod)):
        descriptor = cast('staticmethod[..., object] | classmethod[object, ..., object]', value)
        return call_functions(descriptor.__func__)
    if isinstance(value, partial):
        provider = cast(partial[object], value)
        call = inspect.getattr_static(cast(type[object], type(provider)), '__call__', None)
        return call_functions(provider.func) if call is partial[object].__call__ else call_functions(call)
    if isinstance(value, type):
        return (*call_functions(value.__init__), *call_functions(value.__new__), *call_functions(type(value).__call__))
    if inspect.ismethod(value):
        value = value.__func__
    if inspect.isfunction(value):
        unwrapped = inspect.unwrap(value)
        return (value, *call_functions(unwrapped)) if unwrapped is not value else (value,)
    call = inspect.getattr_static(cast(type[object], type(value)), '__call__', None)
    functions = call_functions(call.fget) if isinstance(call, property) else call_functions(cast(object, call)) if inspect.isfunction(call) or isinstance(call, (staticmethod, classmethod)) else ()
    unwrapped = inspect.unwrap(value) if callable(value) else value
    return (*functions, *call_functions(unwrapped)) if unwrapped is not value else functions


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

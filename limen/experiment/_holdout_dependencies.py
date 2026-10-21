from __future__ import annotations

from builtins import globals as _native_globals
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
from types import CellType, CodeType, GetSetDescriptorType, MappingProxyType, MemberDescriptorType, ModuleType
from typing import Literal, Union, cast

import numpy as np
import polars as pl

from limen.experiment._holdout_calls import call_functions, function_module as _function_module, native_attribute, entropy_call, namespace_calls, native_codes, native_leaf_dependencies, reader_names
from limen.experiment._holdout_state import children, code_digest, fitted_witness, object_payload

__all__ = ['FITTING_METHODS', 'call_functions', 'children', 'code_digest', 'dependency_digest', 'entropy_call', 'fitted_witness', 'fitting_call', 'inference_evidence', 'object_payload', 'restore_iterations']


FITTING_METHODS = frozenset({'fit', 'fit_transform', 'partial_fit', 'train', 'fit_calibrator', 'grid_threshold_optimizer'})
_LIBRARIES = ('numpy', 'polars', 'pandas', 'sklearn', 'lightgbm', 'scipy')
_ATOMIC_TYPES = (str, bool, int, float, bytes, type(Union), type(Literal))
_READER_TYPES: tuple[type[object], ...] = (list, tuple, dict)

def fitting_call(module: str, name: str) -> bool:
    return (name.rsplit('.', 1)[-1] in FITTING_METHODS
            or f'{module}.{name}' in ('limen.experiment._shim_fractional_diff.fit_order', 'limen.features.fractional_diff.find_min_d')
            or (module.startswith(('limen.scalers.', 'limen.targets.')) and name.rsplit('.', 1)[-1] == '__init__'))


@lru_cache(maxsize=2048)
def _instructions(code: CodeType) -> tuple[dis.Instruction, ...]:
    return (*dis.get_instructions(code), *(instruction for part in code.co_consts if isinstance(part, CodeType) and part.co_name not in FITTING_METHODS for instruction in _instructions(part)))


@lru_cache(maxsize=2048)
def _dependency_positions(code: CodeType) -> tuple[int, ...]:
    return tuple(index for index, instruction in enumerate(_instructions(code)) if instruction.opname in ('LOAD_GLOBAL', 'LOAD_NAME', 'LOAD_DEREF', 'LOAD_FAST', 'LOAD_ATTR', 'LOAD_METHOD', 'IMPORT_NAME', 'IMPORT_FROM', 'STORE_FAST', 'STORE_DEREF', 'STORE_NAME') or (instruction.opname == 'LOAD_CONST' and isinstance(instruction.argval, str)))


def _native_annotation(value: object) -> bool:
    if not isinstance(value, type):
        return False
    return (module := sys.modules.get(value.__module__)) is not None and vars(module).get(value.__qualname__) is value and ((root := value.__module__.split('.')[0]) in sys.stdlib_module_names or root in _LIBRARIES)


def inference_evidence(values: Sequence[object], *, exclude: Sequence[object] = (), resolved: Sequence[object] = (), capture: list[object] | None = None) -> list[object]:
    seen: dict[int, object] = {}
    evidence: list[object] = []
    omitted, sealed = ({id(value) for value in exclude}, {id(value) for value in resolved})
    attributes: set[str] = {'__float__', '__int__', '__index__', '__bool__', '__complex__', '__str__', '__repr__', '__bytes__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__neg__', '__pos__', '__abs__', '__invert__', '__round__', '__trunc__', '__floor__', '__ceil__', '__divmod__', '__rdivmod__', '__hash__', '__format__'} | {f'__{prefix}{operator}__' for prefix in ('', 'r', 'i') for operator in ('add', 'sub', 'mul', 'matmul', 'truediv', 'floordiv', 'mod', 'pow', 'lshift', 'rshift', 'and', 'xor', 'or')}
    module_attributes: set[str] = set()
    fitted_names: set[str] = set()
    module_dynamic: dict[int, bool] = {}
    objects: dict[tuple[int, bool], object] = {}
    processed: dict[int, set[str]] = {}
    inspected: set[tuple[int, bool, int, int]] = set()
    recorded_mro: set[int] = set()
    reflective_keys: set[tuple[str, str]] = set()
    globals_names: set[str] = set()
    reflected: dict[int, set[tuple[str, str]]] = {}
    recorded_bindings: set[tuple[int, str, int]] = set()
    method_names = {'predict', 'predict_proba', 'transform', '__call__', '__get__', '__getattr__', '__getattribute__', '__getitem__', '__missing__', '__reversed__', '__enter__', '__exit__', '__aenter__', '__aexit__', '__await__', '__aiter__', '__anext__', '__iter__', '__next__', '__len__', '__contains__', '__instancecheck__', '__subclasscheck__'} | attributes
    constructor_names, constructor_revision = method_names | {'__new__', '__init__'}, len(method_names)

    def visit(value: object, *, fitted_module: bool = True, annotation: bool = False) -> None:
        value_type = type(value)
        if value is None or value_type in _ATOMIC_TYPES or (annotation and _native_annotation(value)) or (value_type.__module__ == 'numpy' and issubclass(value_type, cast(type[object], np.generic)) and not object_payload(value)):
            if isinstance(value, str):
                attributes.add(value)
                method_names.add(value)
                fitted_names.add(value)
            return
        if value_type is logging.Logger or value_type in (GetSetDescriptorType, MemberDescriptorType):
            native_leaf_dependencies(value, seen, evidence, visit)
            return
        if isinstance(value, ModuleType):
            module_dynamic[id(value)] = module_dynamic.get(id(value), False) or fitted_module
        if (identity := id(value)) in seen or identity in omitted:
            return
        seen[identity] = value
        sequence = list[object].__iter__(cast(list[object], value)) if issubclass(value_type, _READER_TYPES[0]) else tuple[object, ...].__iter__(cast(tuple[object, ...], value)) if issubclass(value_type, _READER_TYPES[1]) else ()
        readers = dict[object, object].items(cast(dict[object, object], value)) if issubclass(value_type, _READER_TYPES[2]) else enumerate(sequence)
        globals_names.update(alias for key, part in readers if isinstance(key, (str, int)) and part is _native_globals for alias in reader_names(value, key))
        if callable(value):
            function = value.__func__ if inspect.ismethod(value) else value
            origin = _function_module(function, _LIBRARIES) if inspect.isfunction(function) else getattr(value, '__module__', None) or value_type.__module__
            modules = (origin, type(getattr(value, '__self__', None)).__module__, getattr(getattr(value, '__objclass__', None), '__module__', ''))
            if entropy_call(value) or any(isinstance(module, str) and module.startswith(('numpy.random', 'random', 'secrets')) for module in modules):
                raise ValueError('Holdout inference has an unowned random dependency')
            visit(getattr(value, '__name__', None), annotation=True)
        if inspect.ismethod(value):
            visit(value.__func__)
            visit(value.__self__)
        elif isinstance(value, partial):
            for part in children(cast(object, value), namespace_readers=globals_names):
                visit(part)
            if value_type is not partial:
                _ = objects.setdefault((id(value_type), False), cast(object, value))
        elif inspect.isfunction(value):
            visit(vars(value))
            visit(value.__annotations__, annotation=True)
            visit(value.__doc__, annotation=True)
            visit(getattr(value, '__type_params__', ()), annotation=True)
            visit(value.__defaults__)
            visit(value.__kwdefaults__)
            for cell in value.__closure__ or ():
                for part in children(cell, namespace_readers=globals_names):
                    visit(part)
            function_module, function_name = _function_module(value, _LIBRARIES), getattr(value.__code__, 'co_qualname', value.__code__.co_name)
            if fitting_call(function_module, function_name) or identity in sealed:
                return
            attributes.update(value.__code__.co_names)
            method_names.update(value.__code__.co_names)
            if function_module.split('.')[0] in sys.stdlib_module_names or function_module.split('.')[0] in _LIBRARIES:
                return
            if any(name in value.__code__.co_names for name in ('random', 'default_rng', 'RandomState', 'Generator')):
                raise ValueError('Holdout inference has an unowned random dependency')
            instructions = _instructions(value.__code__)
            references = {name: list(children(cell, namespace_readers=globals_names)[1:]) for name, cell in zip(value.__code__.co_freevars, value.__closure__ or (), strict=True)}
            references.update({name: [part] for name, part in zip(value.__code__.co_varnames[value.__code__.co_argcount - len(value.__defaults__ or ()):value.__code__.co_argcount], value.__defaults__ or (), strict=True)})
            references.update({name: [part] for name, part in (value.__kwdefaults__ or {}).items()})
            imported: ModuleType | None = None
            chained_import = False
            binding: object = None
            for index in _dependency_positions(value.__code__):
                instruction = instructions[index]
                name = str(instruction.argval)
                if instruction.opname in ('LOAD_FAST', 'LOAD_DEREF') and name not in references:
                    continue
                indirect_globals = instruction.opname == 'LOAD_CONST' and name in ('__globals__', '__builtins__') and any(part.opname in ('CALL', 'CALL_FUNCTION') for part in instructions[index + 1:index + 4])
                if name in ('__globals__', '__builtins__') and (instruction.opname in ('LOAD_ATTR', 'LOAD_METHOD') or indirect_globals):
                    lookup = [part for part in instructions[index + 1:index + 6] if part.opname != 'PRECALL']
                    if indirect_globals:
                        lookup = lookup[next(offset for offset, part in enumerate(lookup) if part.opname in ('CALL', 'CALL_FUNCTION')) + 1:]
                    if not lookup[1:] or lookup[0].opname != 'LOAD_CONST' or not isinstance(lookup[0].argval, str) or lookup[1].opname != 'BINARY_SUBSCR':
                        raise ValueError('Holdout requires a literal key for reflective function globals')
                    reflective_keys.add((name, lookup[0].argval))
                if instruction.opname in ('LOAD_GLOBAL', 'LOAD_NAME'):
                    namespace = value.__globals__ if name in value.__globals__ else value.__builtins__
                    if name in namespace:
                        references[name] = [namespace[name]]
                if instruction.opname in ('LOAD_ATTR', 'LOAD_METHOD') or (instruction.opname == 'LOAD_CONST' and isinstance(instruction.argval, str)):
                    attributes.add(name)
                    method_names.add(name)
                    if name not in ('__globals__', '__builtins__'):
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
                if (name == 'fit_calibrator' and (function_module, value.__qualname__) in (('limen.sfd.reference_architecture.logreg_binary', 'LogRegBinary.predict'), ('limen.sfd.reference_architecture.lightgbm_binary', 'LightGBMBinary.predict')) and value.__code__ in native_codes(function_module)) or instruction.opname not in ('LOAD_GLOBAL', 'LOAD_NAME', 'LOAD_DEREF', 'LOAD_FAST'):
                    continue
                for candidate in tuple(references.get(name, ())):
                    item = candidate
                    following = index + 1
                    while following < len(instructions):
                        attribute = instructions[following]
                        if attribute.opname in ('LOAD_ATTR', 'LOAD_METHOD') and hasattr(item, str(attribute.argval)):
                            item = getattr(item, str(attribute.argval))
                            following += 1
                        elif type(item) in (dict, MappingProxyType) and attribute.opname == 'LOAD_CONST' and isinstance(attribute.argval, str) and following + 1 < len(instructions) and instructions[following + 1].opname == 'BINARY_SUBSCR' and attribute.argval in cast(Mapping[object, object], item):
                            item = cast(Mapping[object, object], item)[attribute.argval]
                            following += 2
                        else:
                            break
                    if item is _native_globals and (following == len(instructions) or instructions[following].opname not in ('STORE_FAST', 'STORE_DEREF', 'STORE_NAME')) and not (following + 1 < len(instructions) and instructions[following].opname in ('LOAD_FAST', 'LOAD_DEREF') and instructions[following + 1].opname == 'STORE_ATTR'):
                        lookup = [part for part in instructions[following:following + 5] if part.opname != 'PRECALL']
                        if not lookup[2:] or lookup[0].opname not in ('CALL', 'CALL_FUNCTION') or lookup[0].arg != 0 or lookup[1].opname != 'LOAD_CONST' or not isinstance(lookup[1].argval, str) or lookup[2].opname != 'BINARY_SUBSCR':
                            raise ValueError('Holdout requires a literal key for prediction globals lookup')
                        if lookup[1].argval in value.__globals__:
                            part = cast(object, value.__globals__[lookup[1].argval])
                            visit(part)
                            evidence.append((function_module, value.__qualname__, lookup[1].argval, part))
                    if isinstance(item, ModuleType) and index > 0 and instructions[index - 1].argval == 'getattr':
                        for attribute in attributes.intersection(vars(item)):
                            visit(getattr(item, attribute))
                    if following < len(instructions) and instructions[following].opname in ('STORE_FAST', 'STORE_DEREF', 'STORE_NAME'):
                        aliases = references.setdefault(str(instructions[following].argval), [])
                        if all(part is not item for part in aliases):
                            aliases.append(item)
                    try:
                        visit(item)
                        binding_key = (identity, name, id(item))
                        if binding_key not in recorded_bindings:
                            recorded_bindings.add(binding_key)
                            evidence.append((value.__module__, value.__qualname__, instruction.argval, item))
                    except (ValueError, TypeError, pickle.PickleError) as exc:
                        raise ValueError(f'Holdout dependency {value.__module__}.{value.__qualname__}:{name}: {exc}') from exc
        elif value_type.__module__ == 'builtins' and value_type not in (dict, list, tuple, set, frozenset) and inspect.isroutine(value):
            owner = getattr(value, '__self__', None)
            if owner is not None and not isinstance(owner, ModuleType):
                visit(owner)
        elif object_payload(value):
            for part in children(cast(object, value), namespace_readers=globals_names):
                visit(part, annotation=annotation)
            if value_type is not np.ndarray:
                _ = objects.setdefault((id(value_type), False), value)
        elif isinstance(value, ModuleType):
            _ = objects.setdefault((identity, False), value)
        elif not isinstance(value, (pl.DataFrame, pl.Series)) and value_type is not np.ndarray:
            if isinstance(value, type) and value_type is not type:
                visit(value_type)
            if value_type not in (dict, list, tuple, set, frozenset, defaultdict) and not (annotation and value_type.__module__ == 'typing'):
                _ = objects.setdefault((identity if isinstance(value, type) else id(value_type), isinstance(value, type)), value)
            if not isinstance(value, type):
                for part in children(value, namespace_readers=globals_names):
                    visit(part, annotation=annotation)

    def methods(value: object) -> None:
        nonlocal constructor_names, constructor_revision
        if isinstance(value, ModuleType):
            done = processed.setdefault(id(value), set())
            for name in (module_attributes | (fitted_names if module_dynamic[id(value)] else set[str]())).intersection(vars(value)) - done:
                done.add(name)
                part = cast(object, vars(value)[name])
                visit(part, fitted_module=name in module_attributes)
                evidence.append((value.__name__, name, part))
            return
        cls = value if isinstance(value, type) else type(value)
        if isinstance(value, type) and constructor_revision != len(method_names):
            constructor_names, constructor_revision = method_names | {'__new__', '__init__'}, len(method_names)
        names = constructor_names if isinstance(value, type) else method_names
        if id(cls) not in recorded_mro:
            recorded_mro.add(id(cls))
            evidence.append(('mro', cls.__module__, cls.__qualname__, tuple((parent.__module__, parent.__qualname__) for parent in cls.__mro__)))
        for parent in cls.__mro__:
            stamp = (id(parent), isinstance(value, type), len(names), len(parent.__dict__))
            if stamp in inspected:
                continue
            inspected.add(stamp)
            done = processed.setdefault(id(parent), set())
            for name in names.intersection(parent.__dict__) - done:
                done.add(name)
                if fitting_call(parent.__module__, f'{parent.__qualname__}.{name}'):
                    continue
                key = f'{parent.__module__}.{parent.__qualname__}:{name}'
                method = parent.__dict__[name]
                evidence.append((key, method))
                if isinstance(method, (type, ModuleType)):
                    evidence.append((key, type(method).__module__, type(method).__qualname__))
                if isinstance(method, (property, staticmethod, classmethod)):
                    visit(cast(object, method))
                if isinstance(method, property):
                    method = native_attribute(method, 'fget')
                if isinstance(method, (staticmethod, classmethod)):
                    method = native_attribute(cast(object, method), '__func__')
                if method is _native_globals:
                    globals_names.add(name)
                if method is not None:
                    visit(method, annotation=name in ('__annotations__', '__type_params__', '__parameters__'))

    for value in values:
        visit(value)
    previous = (-1, -1, -1, -1, -1, -1, -1, -1)
    while previous != (len(attributes), len(objects), len(module_attributes), len(fitted_names), sum(module_dynamic.values()), len(reflective_keys), len(seen) if reflective_keys or globals_names else 0, len(globals_names)):
        previous = (len(attributes), len(objects), len(module_attributes), len(fitted_names), sum(module_dynamic.values()), len(reflective_keys), len(seen) if reflective_keys or globals_names else 0, len(globals_names))
        for value in tuple(objects.values()):
            methods(value)
        if reflective_keys or globals_names:
            functions = [value for value in seen.values() if inspect.isfunction(value)]
            functions.extend(function for config in exclude for name in ('calibration_func', 'threshold_func') for function in call_functions(getattr(config, name, None)))
            for function in functions:
                for name, key in namespace_calls(function.__code__):
                    if name in globals_names:
                        if key is None:
                            raise ValueError('Holdout requires a literal key for prediction globals lookup')
                        reflective_keys.add(('__globals__', key))
                done = reflected.setdefault(id(function), set())
                for namespace_name, key in sorted(reflective_keys - done):
                    done.add((namespace_name, key))
                    namespace = cast(dict[str, object], getattr(function, namespace_name))
                    if key not in namespace:
                        continue
                    part = namespace[key]
                    visit(part)
                    evidence.append((namespace_name, function.__module__, function.__qualname__, key, part))
    if capture is not None:
        capture.extend(seen.values())
    result: list[object] = [fitted_witness((tuple(values), evidence))]
    cast(tuple[CellType, ...], visit.__closure__)[visit.__code__.co_freevars.index('visit')].cell_contents = None
    return result


def dependency_digest(values: Sequence[object], *, exclude: Sequence[object] = (), resolved: Sequence[object] = (), capture: list[object] | None = None) -> str:
    return sha256(pickle.dumps(inference_evidence(values, exclude=exclude, resolved=resolved, capture=capture), protocol=5)).hexdigest()



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

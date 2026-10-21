from __future__ import annotations

from builtins import globals as _native_globals
from functools import lru_cache
import inspect
import sys
from types import CodeType, FunctionType, ModuleType
from typing import cast

from limen.experiment._holdout_calls import call_functions, function_module
from limen.experiment._holdout_namespaces import namespace_calls
from limen.experiment._holdout_state import children, fitted_witness

__all__ = ['declaration_witness']
_LIBRARIES = ('limen', 'numpy', 'polars', 'pandas', 'sklearn', 'lightgbm', 'scipy')


@lru_cache(maxsize=2048)
def _codes(code: CodeType) -> tuple[CodeType, ...]:
    return (code, *(nested for part in code.co_consts if isinstance(part, CodeType) for nested in _codes(part)))


def _native(function: FunctionType) -> bool:
    root = function_module(function, _LIBRARIES).split('.')[0]
    return root in sys.stdlib_module_names or root in _LIBRARIES


def declaration_witness(value: object) -> bytes:
    pending, seen, functions = [value], dict[int, object](), set[FunctionType]()
    bindings: list[object] = []
    owners: dict[int, type[object]] = {}
    modules: dict[int, ModuleType] = {}
    inspected: set[tuple[int, int]] = set()
    attributes: set[str] = {'__new__', '__init__', '__call__'}
    class_state: dict[tuple[type[object] | ModuleType, str], object] = {}
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen[id(item)] = item
        targets = call_functions(item)
        if isinstance(item, ModuleType):
            modules[id(item)] = item
        if isinstance(item, type) or (targets and not inspect.isfunction(item)):
            owner = item.__self__ if inspect.ismethod(item) else item
            cls = cast(type[object], owner if isinstance(owner, type) else type(owner))
            owners[id(cls)] = cls
            owners[id(type(cls))] = type(cls)
        for function in targets:
            if function in functions:
                continue
            functions.add(function)
            codes = _codes(function.__code__)
            names = {name for code in codes for name in code.co_names}
            attributes.update(names | {part for code in codes for part in code.co_consts if isinstance(part, str)})
            namespace = {name: (function.__globals__ if name in function.__globals__ else function.__builtins__)[name] for name in sorted(names) if name in function.__globals__ or name in function.__builtins__}
            parameters = dict(zip(function.__code__.co_varnames[function.__code__.co_argcount - len(function.__defaults__ or ()):function.__code__.co_argcount], function.__defaults__ or (), strict=True)) | (function.__kwdefaults__ or {})
            parameters.update({name: part for name, cell in zip(function.__code__.co_freevars, function.__closure__ or (), strict=True) for part in children(cell)[1:]})
            readers = {name for name, part in (namespace | parameters).items() if part is _native_globals}
            for name, key in namespace_calls(function.__code__):
                if name in readers:
                    if key is None:
                        raise ValueError('Holdout requires a literal key for factory globals lookup')
                    if key in function.__globals__:
                        namespace[key] = function.__globals__[key]
            bindings.append((function, namespace))
            pending.extend((function.__defaults__, function.__kwdefaults__, function.__module__, function.__qualname__, *(function.__closure__ or ())))
            pending.extend(part for part in namespace.values() if not _native(function) or isinstance(part, type))
        if isinstance(item, str):
            attributes.add(str.__str__(item))
        if not isinstance(item, ModuleType):
            pending.extend(children(item))
        parents = (*modules.values(), *(parent for cls in tuple(owners.values()) for parent in cls.__mro__))
        for parent in parents:
            stamp = (id(parent), len(attributes))
            if stamp in inspected:
                continue
            inspected.add(stamp)
            for name in sorted(attributes.intersection(vars(parent))):
                key = (parent, name)
                if key not in class_state:
                    part = vars(parent)[name]
                    class_state[key] = part
                    pending.extend((part,) if name in ('__new__', '__init__', '__call__') or any(not _native(function) for function in call_functions(part)) else children(part))
    return fitted_witness((value, bindings, tuple((cls, name, part) for (cls, name), part in class_state.items())))

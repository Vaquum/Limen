from __future__ import annotations

from builtins import globals as _native_globals
from functools import lru_cache
import inspect
import sys
from types import CodeType, FunctionType, MethodWrapperType, ModuleType
from typing import cast

from limen.experiment._holdout_calls import call_functions, factory_imports, function_module, native_attribute, reader_names, reader_types
from limen.experiment._holdout_namespaces import namespace_calls, reflective_reads
from limen.experiment._holdout_state import children, fitted_witness, native_values

__all__ = ['declaration_witness']
_LIBRARIES = ('limen', 'numpy', 'polars', 'pandas', 'sklearn', 'lightgbm', 'scipy', '_pytest')


@lru_cache(maxsize=2048)
def _codes(code: CodeType) -> tuple[CodeType, ...]:
    return (code, *(nested for part in code.co_consts if isinstance(part, CodeType) for nested in _codes(part)))


def _native(function: FunctionType) -> bool:
    return (root := function_module(function, _LIBRARIES).split('.')[0]) in sys.stdlib_module_names or root in _LIBRARIES


def declaration_witness(value: object) -> bytes:
    pending, seen, readers = [value], dict[int, object](), set[str]()
    bindings: dict[FunctionType, dict[str, object]] = {}
    reflected: set[tuple[str, str]] = set()
    owners: dict[int, type[object]] = {}
    modules: dict[int, ModuleType] = {}
    inspected: set[tuple[int, ...]] = set()
    attributes: set[str] = {'__new__', '__init__', '__call__'}
    class_state: dict[tuple[type[object] | ModuleType, str], object] = {}
    while pending:
        if (stamp := (len(bindings), len(readers), len(reflected), 0)) not in inspected:
            inspected.add(stamp)
            for function, namespace in tuple(bindings.items()):
                pending.extend(namespace.setdefault(f'{space}:{key}', mapping[key]) for space, key in sorted(reflected) if key in (mapping := getattr(function, space)) and f'{space}:{key}' not in namespace)
                for name, key in namespace_calls(function.__code__):
                    if name in readers:
                        if key is None:
                            raise ValueError('Holdout requires a literal key for factory globals lookup')
                        if key not in namespace and key in function.__globals__:
                            namespace[key] = function.__globals__[key]
                            pending.append(namespace[key])
        if id(item := pending.pop()) in seen:
            continue
        seen[id(item)] = item
        if isinstance(item, ModuleType):
            modules[id(item)] = item
        elif not inspect.isfunction(item):
            owner = item.__self__ if inspect.ismethod(item) else item
            cls = cast(type[object], owner if isinstance(owner, type) else type(owner))
            owners[id(cls)] = cls
            owners[id(type(cls))] = type(cls)
        pending.extend((getattr(item, '__self__', None),) if inspect.isbuiltin(item) or isinstance(item, MethodWrapperType) else ())
        if isinstance(item, reader_types):
            readers.update(alias for key, part in (dict[object, object].items(cast(dict[object, object], item)) if isinstance(item, dict) else enumerate(native_values(item))) if isinstance(key, (str, int)) and part is _native_globals for alias in reader_names(cast(object, item), key))
        for function in call_functions(cast(object, item)):
            if function in bindings:
                continue
            codes = _codes(function.__code__)
            names = {name for code in codes for name in code.co_names}
            attributes.update(names | {part for code in codes for part in code.co_consts if isinstance(part, str)})
            namespace = {name: (function.__globals__ if name in function.__globals__ else function.__builtins__)[name] for name in sorted(names) if name in function.__globals__ or name in function.__builtins__}
            namespace.update(factory_imports(function, codes) if not _native(function) else {})
            reflected.update(reflective_reads(function.__code__))
            parameters = cast(dict[str, object], dict(zip(function.__code__.co_varnames[function.__code__.co_argcount - len(function.__defaults__ or ()):function.__code__.co_argcount], function.__defaults__ or (), strict=True)) | (function.__kwdefaults__ or {}))
            parameters.update({name: part for name, cell in zip(function.__code__.co_freevars, function.__closure__ or (), strict=True) for part in children(cell)[1:]})
            readers.update(name for name, part in (namespace | parameters).items() if part is _native_globals)
            bindings[function] = namespace
            pending.extend((vars(function), function.__annotations__, function.__doc__, getattr(function, '__type_params__', ()), function.__defaults__, function.__kwdefaults__, function.__name__, function.__module__, function.__qualname__, *(function.__closure__ or ())))
            pending.extend(part for part in namespace.values() if not _native(function) or isinstance(part, type))
        if isinstance(item, str):
            attributes.add(str.__str__(item))
        if not isinstance(item, ModuleType):
            pending.extend(children(cast(object, item), namespace_readers=readers))
        stamp = (len(owners), len(modules), len(attributes))
        if stamp in inspected:
            continue
        inspected.add(stamp)
        parents = (*modules.values(), *(parent for cls in tuple(owners.values()) for parent in cls.__mro__))
        for parent in parents:
            stamp = (id(parent), len(attributes))
            if stamp in inspected:
                continue
            inspected.add(stamp)
            for name in sorted(attributes.intersection(vars(parent)) | {name for name, part in vars(parent).items() if name.startswith('__') and name.endswith('__') and any(not _native(function) for function in call_functions(cast(object, part)))}):
                if (key := (parent, name)) not in class_state:
                    class_state[key] = part = vars(parent)[name]
                    readers.update((name,) if (native_attribute(cast(object, part), '__func__') if isinstance(part, (staticmethod, classmethod)) else part) is _native_globals else ())
                    pending.extend((cast(object, part),) if name in ('__new__', '__init__', '__call__') or not (calls := call_functions(cast(object, part))) or any(not _native(function) for function in calls) else ())
    return fitted_witness((value, tuple(bindings.items()), tuple((cls, name, part) for (cls, name), part in class_state.items())))

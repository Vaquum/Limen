from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import partial
import inspect
from threading import RLock
from types import FrameType, FunctionType
from typing import NamedTuple, cast

__all__ = ['call_codes', 'call_functions', 'constructor_entry', 'matches']

_LOCK = RLock()


@contextmanager
def call_codes(functions: Sequence[FunctionType]) -> Iterator[None]:
    with _LOCK:
        originals = {function: function.__code__ for function in functions}
        try:
            for function, code in originals.items():
                function.__code__ = code.replace()
            yield
        finally:
            for function, code in originals.items():
                function.__code__ = code


class _ConstructorEntry(NamedTuple):
    target: object
    instance: bool
    index: int


def constructor_entry(value: type[object], active: frozenset[int] = frozenset()) -> _ConstructorEntry:
    if id(value) in active:
        return _ConstructorEntry(None, False, 0)
    active = active | {id(value)}
    new = inspect.getattr_static(value, '__new__', None)
    if type(new) is staticmethod:
        new = cast('staticmethod[..., object]', new).__func__
    if new is not object.__new__:
        index = _call_offset(new, active)
        return _ConstructorEntry(None, False, 0) if index is None else _ConstructorEntry(new, False, index)
    initializer = inspect.getattr_static(value, '__init__', None)
    supported = inspect.isfunction(initializer) or isinstance(initializer, (property, classmethod))
    return _ConstructorEntry(initializer if supported else None, not isinstance(initializer, classmethod), 0)


def _call_offset(value: object, active: frozenset[int]) -> int | None:
    if id(value) in active:
        return None
    if isinstance(value, type):
        call = inspect.getattr_static(cast(type[object], type(value)), '__call__', None)
        if call is type.__call__:
            entry = constructor_entry(cast(type[object], value), active)
            return entry.index + 1 if entry.target is not None else None
        offset = _call_offset(call, active | {id(value)})
        return None if offset is None else offset + int(inspect.isfunction(call))
    active = active | {id(value)}
    if inspect.isfunction(value):
        return 0
    if inspect.ismethod(value):
        return 1
    if isinstance(value, (staticmethod, classmethod)):
        native: type[object] = staticmethod if isinstance(value, staticmethod) else classmethod
        descriptor = cast('staticmethod[..., object] | classmethod[object, ..., object]', value)
        if inspect.getattr_static(type(descriptor), '__get__') is not inspect.getattr_static(native, '__get__'):
            return None
        offset = _call_offset(descriptor.__func__, active)
        return None if offset is None else offset + int(isinstance(descriptor, classmethod))
    call = inspect.getattr_static(cast(type[object], type(value)), '__call__', None)
    if isinstance(value, partial) and call is partial[object].__call__:
        provider = cast(partial[object], value)
        offset = _call_offset(provider.func, active)
        return None if offset is None else offset + len(provider.args)
    return 1 if inspect.isfunction(call) else _call_offset(call, active) if isinstance(call, (staticmethod, classmethod)) else None


def call_functions(value: object) -> tuple[FunctionType, ...]:
    if isinstance(value, (property, staticmethod, classmethod)):
        native: type[object] = property if isinstance(value, property) else staticmethod if isinstance(value, staticmethod) else classmethod
        descriptor = cast('property | staticmethod[..., object] | classmethod[object, ..., object]', value)
        if inspect.getattr_static(type(descriptor), '__get__') is not inspect.getattr_static(native, '__get__'):
            return ()
        return call_functions(descriptor.fget if isinstance(descriptor, property) else descriptor.__func__)
    if isinstance(value, partial):
        provider = cast(partial[object], value)
        call = inspect.getattr_static(cast(type[object], type(provider)), '__call__', None)
        return call_functions(provider.func) if call is partial[object].__call__ else call_functions(call)
    if isinstance(value, type):
        call = inspect.getattr_static(cast(type[object], type(value)), '__call__', None)
        return call_functions(constructor_entry(cast(type[object], value)).target) if call is type.__call__ else call_functions(call)
    if inspect.ismethod(value):
        value = value.__func__
    if inspect.isfunction(value):
        return (value,)
    call = inspect.getattr_static(cast(type[object], type(value)), '__call__', None)
    return call_functions(cast(object, call)) if inspect.isfunction(call) or isinstance(call, (property, staticmethod, classmethod)) else ()


def matches(function: FunctionType, frame: FrameType, owner: object | None, instance: bool, index: int) -> bool:
    if owner is not None:
        code = frame.f_code
        if index < code.co_argcount:
            receiver = frame.f_locals.get(code.co_varnames[index])
        elif code.co_flags & inspect.CO_VARARGS:
            arguments = frame.f_locals.get(code.co_varnames[code.co_argcount + code.co_kwonlyargcount], ())
            position = index - code.co_argcount
            receiver = cast(tuple[object, ...], arguments)[position] if isinstance(arguments, tuple) and len(arguments) > position else None
        else:
            return False
        if (type(receiver) if instance else receiver) is not owner:
            return False
    return frame.f_globals is function.__globals__ and all(
        frame.f_locals.get(name) is cell.cell_contents
        for name, cell in zip(function.__code__.co_freevars, function.__closure__ or (), strict=True))

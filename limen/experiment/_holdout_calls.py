from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import partial
import inspect
from threading import RLock
from types import FrameType, FunctionType, GetSetDescriptorType, MemberDescriptorType, MethodType
from typing import NamedTuple, cast

__all__ = ['call_codes', 'call_functions', 'constructor_entry', 'matches', 'native_attribute', 'receiver_position']

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


def native_attribute(value: object, name: str, owner: type[object] | None = None) -> object:
    owner = owner or (partial if isinstance(value, partial) else property if isinstance(value, property) else staticmethod if isinstance(value, staticmethod) else classmethod)
    descriptor = cast(GetSetDescriptorType | MemberDescriptorType, vars(owner)[name])
    native_value = cast(object, value)
    return descriptor.__get__(native_value, type(native_value))


def receiver_position(value: object) -> int | None:
    if inspect.ismethod(value):
        return _call_offset(value.__func__, frozenset())
    if isinstance(value, (property, classmethod)):
        return _call_offset(native_attribute(cast(object, value), 'fget' if isinstance(value, property) else '__func__'), frozenset())
    return 0 if inspect.isfunction(value) else None


class _ConstructorEntry(NamedTuple):
    target: object
    instance: bool
    position: int


def constructor_entry(value: type[object], active: frozenset[int] | None = None) -> _ConstructorEntry:
    active = frozenset() if active is None else active
    if id(value) in active:
        return _ConstructorEntry(None, False, 0)
    active = active | {id(value)}
    new = inspect.getattr_static(value, '__new__', None)
    if type(new) is staticmethod:
        new = native_attribute(cast(object, new), '__func__')
    if new is not object.__new__:
        index = _call_offset(new, active)
        return _ConstructorEntry(None, False, 0) if index is None else _ConstructorEntry(new, False, index)
    initializer = inspect.getattr_static(value, '__init__', None)
    supported = inspect.isfunction(initializer) or isinstance(initializer, (property, classmethod))
    index = receiver_position(cast(object, initializer)) if supported else None
    return _ConstructorEntry(None, False, 0) if index is None else _ConstructorEntry(initializer, not isinstance(initializer, classmethod), index)


def _class_offset(value: type[object], active: frozenset[int]) -> int | None:
    call = inspect.getattr_static(type(value), '__call__', None)
    if call is type.__call__:
        entry = constructor_entry(value, active)
        return entry.position + 1 if entry.target is not None else None
    offset = _call_offset(call, active | {id(value)})
    return None if offset is None else offset + int(inspect.isfunction(call))


def _descriptor_offset(descriptor: 'staticmethod[..., object] | classmethod[object, ..., object]', active: frozenset[int]) -> int | None:
    native: type[object] = staticmethod if isinstance(descriptor, staticmethod) else classmethod
    if inspect.getattr_static(type(descriptor), '__get__') is not inspect.getattr_static(native, '__get__'):
        return None
    target = native_attribute(descriptor, '__func__')
    if isinstance(descriptor, classmethod) and not inspect.isfunction(target) and inspect.getattr_static(type(target), '__get__', None) is not None:
        return None
    offset = _call_offset(target, active)
    return None if offset is None else offset + int(isinstance(descriptor, classmethod))


def _method_offset(value: MethodType, active: frozenset[int]) -> int | None:
    offset = _call_offset(value.__func__, active)
    return None if offset is None else offset + 1


def _call_offset(value: object, active: frozenset[int]) -> int | None:
    if id(value) in active:
        return None
    if isinstance(value, type):
        return _class_offset(cast(type[object], value), active)
    active = active | {id(value)}
    if inspect.isfunction(value) or inspect.ismethod(value):
        return _method_offset(value, active) if inspect.ismethod(value) else 0
    if isinstance(value, (staticmethod, classmethod)):
        return _descriptor_offset(cast('staticmethod[..., object] | classmethod[object, ..., object]', value), active)
    call = inspect.getattr_static(type(value), '__call__', None)
    if isinstance(value, partial) and call is partial[object].__call__:
        provider = cast(partial[object], value)
        offset = _call_offset(native_attribute(provider, 'func'), active)
        return None if offset is None else offset + len(cast(tuple[object, ...], native_attribute(provider, 'args')))
    return 1 if inspect.isfunction(call) else _call_offset(cast(object, call), active) if isinstance(call, (staticmethod, classmethod)) else None


def _descriptor_functions(descriptor: 'property | staticmethod[..., object] | classmethod[object, ..., object]') -> tuple[FunctionType, ...]:
    native: type[object] = property if isinstance(descriptor, property) else staticmethod if isinstance(descriptor, staticmethod) else classmethod
    if inspect.getattr_static(type(descriptor), '__get__') is not inspect.getattr_static(native, '__get__'):
        return ()
    target = native_attribute(descriptor, 'fget' if isinstance(descriptor, property) else '__func__')
    if isinstance(descriptor, classmethod) and not inspect.isfunction(target) and inspect.getattr_static(type(target), '__get__', None) is not None:
        return ()
    return call_functions(target)


def call_functions(value: object) -> tuple[FunctionType, ...]:
    if isinstance(value, (property, staticmethod, classmethod)):
        call = inspect.getattr_static(type(cast(object, value)), '__call__', None)
        return _bound_functions(call) if callable(value) and call is not inspect.getattr_static(staticmethod, '__call__') else _descriptor_functions(cast('property | staticmethod[..., object] | classmethod[object, ..., object]', value))
    if isinstance(value, partial):
        provider = cast(partial[object], value)
        call = inspect.getattr_static(cast(type[object], type(provider)), '__call__', None)
        return call_functions(native_attribute(provider, 'func')) if call is partial[object].__call__ else _bound_functions(call)
    if isinstance(value, type):
        call = inspect.getattr_static(cast(type[object], type(value)), '__call__', None)
        return call_functions(constructor_entry(cast(type[object], value)).target) if call is type.__call__ else _bound_functions(call)
    if inspect.ismethod(value):
        return call_functions(value.__func__)
    if inspect.isfunction(value):
        return (value,)
    call = inspect.getattr_static(type(value), '__call__', None)
    return _bound_functions(cast(object, call)) if inspect.isfunction(call) or inspect.ismethod(call) or isinstance(call, (property, staticmethod, classmethod)) else ()


def _bound_functions(value: object) -> tuple[FunctionType, ...]:
    return _descriptor_functions(cast('property | staticmethod[..., object] | classmethod[object, ..., object]', value)) if isinstance(value, (property, staticmethod, classmethod)) else call_functions(value)


def matches(function: FunctionType, frame: FrameType, owner: object | None, instance: bool, index: int) -> bool:
    if owner is not None:
        code = frame.f_code
        if index < code.co_argcount:
            receiver = frame.f_locals.get(code.co_varnames[index])
        elif code.co_flags & inspect.CO_VARARGS:
            arguments = frame.f_locals.get(code.co_varnames[code.co_argcount + code.co_kwonlyargcount], ())
            position = index - code.co_argcount
            values = cast(tuple[object, ...], arguments) if isinstance(arguments, tuple) else ()
            receiver = values[position] if len(values) > position else None
        else:
            return False
        if (type(receiver) if instance else receiver) is not owner:
            return False
    return frame.f_globals is function.__globals__ and all(
        frame.f_locals.get(name) is cell.cell_contents
        for name, cell in zip(function.__code__.co_freevars, function.__closure__ or (), strict=True))

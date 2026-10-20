from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import partial
import inspect
from threading import RLock
from types import FrameType, FunctionType
from typing import cast

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


def constructor_entry(value: type[object]) -> tuple[object, bool]:
    new = inspect.getattr_static(value, '__new__', None)
    if type(new) is staticmethod:
        new = cast('staticmethod[..., object]', new).__func__
    if new is not object.__new__:
        return new, False
    initializer = inspect.getattr_static(value, '__init__', None)
    return (None, False) if isinstance(initializer, staticmethod) else (initializer, not isinstance(initializer, classmethod))


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
        return call_functions(constructor_entry(cast(type[object], value))[0]) if call is type.__call__ else call_functions(call)
    if inspect.ismethod(value):
        value = value.__func__
    if inspect.isfunction(value):
        return (value,)
    call = inspect.getattr_static(cast(type[object], type(value)), '__call__', None)
    return call_functions(cast(object, call)) if inspect.isfunction(call) or isinstance(call, (property, staticmethod, classmethod)) else ()


def matches(function: FunctionType, frame: FrameType, owner: object | None, instance: bool) -> bool:
    if owner is not None:
        code = frame.f_code
        if code.co_argcount:
            receiver = frame.f_locals.get(code.co_varnames[0])
        elif code.co_flags & inspect.CO_VARARGS:
            arguments = frame.f_locals.get(code.co_varnames[code.co_kwonlyargcount], ())
            receiver = cast(tuple[object, ...], arguments)[0] if isinstance(arguments, tuple) and arguments else None
        else:
            return False
        if (type(receiver) if instance else receiver) is not owner:
            return False
    return frame.f_globals is function.__globals__ and all(
        frame.f_locals.get(name) is cell.cell_contents
        for name, cell in zip(function.__code__.co_freevars, function.__closure__ or (), strict=True))

from __future__ import annotations

from inspect import getattr_static
from types import FunctionType
from typing import cast

from limen.experiment._parallel_state import call_implementation


def source_class(cls: type) -> type:
    return cast(type, vars(cls).get('_parallel_source_class', cls))


def bind_call(instance: object, implementation: object) -> None:
    cls = source_class(type(instance))
    if type(cls) is not type or not isinstance(implementation, FunctionType) or implementation.__code__ is not getattr(call_implementation(instance), '__code__', None):
        raise ValueError('Parallel callable implementation or metaclass state changed')
    raw = getattr_static(cls, '__call__')
    descriptor: object = classmethod(implementation) if isinstance(raw, classmethod) else staticmethod(implementation) if isinstance(raw, staticmethod) else implementation
    attributes: dict[str, object] = {'__slots__': (), '__module__': cls.__module__, '__qualname__': cls.__qualname__,
                  '_parallel_source_class': cls, '__call__': descriptor}
    object.__setattr__(instance, '__class__', type(cls.__name__, (cls,), attributes))


def restore_function_metadata(function: FunctionType, source: FunctionType, wrapped: object) -> None:
    function.__module__, function.__qualname__, function.__annotations__ = source.__module__, source.__qualname__, source.__annotations__.copy()
    for name, value in {'__wrapped__': wrapped, '__parallel_round_contract__': vars(source).get('__parallel_round_contract__')}.items():
        if value is not None:
            vars(function)[name] = value

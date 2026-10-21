from __future__ import annotations

from inspect import getattr_static
from types import FunctionType
from typing import cast

from limen.experiment._parallel_graph import isolated_encoding
from limen.experiment._parallel_identity import execution_class
from limen.experiment._parallel_state import call_implementation, mutable_function

_CAPTURED_CALL_PARTS = 2


def source_class(cls: type) -> type:
    return cast(type, vars(cls).get('_parallel_source_class', cls))


def bind_call(instance: object, captured: object) -> None:
    if not isinstance(captured, tuple) or len(cast(tuple[object, ...], captured)) != _CAPTURED_CALL_PARTS:
        raise ValueError('Invalid captured parallel call')
    implementation, contract = cast(tuple[object, object], captured)
    cls = source_class(type(instance))
    if type(cls) is not type or not isinstance(implementation, FunctionType) or implementation.__code__ is not getattr(call_implementation(instance), '__code__', None):
        raise ValueError('Parallel callable implementation or metaclass state changed')
    raw = getattr_static(cls, '__call__')
    descriptor: object = classmethod(implementation) if isinstance(raw, classmethod) else staticmethod(implementation) if isinstance(raw, staticmethod) else implementation
    if hasattr(instance, '__dict__'):
        vars(instance)['__parallel_round_contract__'] = contract
    from limen.experiment._parallel_encoding import digest
    with isolated_encoding():
        identity = (digest(implementation), id(implementation) if mutable_function(implementation) else 0, 0 if hasattr(instance, '__dict__') else id(contract))
    attributes: dict[str, object] = {'__slots__': (), '__module__': cls.__module__, '__qualname__': cls.__qualname__,
                  '_parallel_source_class': cls, '__call__': descriptor, '__parallel_round_contract__': contract}
    object.__setattr__(instance, '__class__', execution_class(cls, identity, attributes))

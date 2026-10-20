from __future__ import annotations

from collections.abc import Callable
from functools import partial
from inspect import Parameter, ismethod, signature
from types import FunctionType, MemberDescriptorType
from typing import cast

from limen.experiment._parallel_graph import _EMPTY_IDS, _value_only

ObjectState = tuple[dict[str, object], list[tuple[type, str, object]], tuple[object, ...] | None, list[object] | None]


def slot_state(value: object) -> list[tuple[type, str, object]]:
    result: list[tuple[type, str, object]] = []
    missing = object()
    for cls in (cls for cls in type(value).__mro__ if cls is not partial):
        for name, descriptor in vars(cls).items():
            if isinstance(descriptor, MemberDescriptorType):
                try:
                    item = descriptor.__get__(value, type(value))
                except AttributeError:
                    item = missing
                if item is not missing:
                    result.append((cls, name, item))
    return result


def component_state(value: object) -> dict[str, object]:
    if ismethod(value):
        return {} if isinstance(value.__self__, type) else component_state(value.__self__)
    state = dict(vars(value)) if hasattr(value, '__dict__') else {}
    state |= {name: item for _, name, item in reversed(slot_state(value))}
    if issubclass(type(value), partial):
        bound = object_state(value)[2]
        assert bound is not None
        state |= component_state(bound[0])
        binding = signature(cast(Callable[..., object], bound[0])).bind_partial(
            *cast(tuple[object, ...], bound[1]), **cast(dict[str, object], bound[2] or {}))
        state |= {name: item for name, item in cast(dict[str, object], binding.arguments).items() if binding.signature.parameters[name].kind != Parameter.VAR_KEYWORD} | binding.kwargs
    native = native_state(value)
    return state | ({'__parallel_native_state__': native} if native is not None else {})


def native_state(value: object) -> list[object] | None:
    return (list[object](dict[object, object].items(cast(dict[object, object], value))) if issubclass(type(value), dict)
            else list[object](list[object].__iter__(cast(list[object], value))) if issubclass(type(value), list) else None)


def object_state(value: object) -> ObjectState:
    attributes = {key: item for key, item in vars(value).items() if key != '__parallel_round_contract__'} if hasattr(value, '__dict__') else {}
    reducer = cast(Callable[[object], tuple[object, ...]], vars(partial)['__reduce__'])
    reduced = reducer(value) if issubclass(type(value), partial) else None
    bound = cast(tuple[object, ...], reduced[2])[:3] if reduced is not None else None
    return attributes, slot_state(value), bound, native_state(value)


def restore_object(cls: type, state: ObjectState) -> object:
    attributes, slots, bound, native = state
    is_mapping, is_sequence = issubclass(cls, dict), issubclass(cls, list)
    if bound is None:
        if is_mapping:
            result: object = dict[object, object].__new__(cast(type[dict[object, object]], cls))
            dict[object, object].update(result, cast(list[tuple[object, object]], native))
        elif is_sequence:
            result = list[object].__new__(cast(type[list[object]], cls))
            list[object].extend(result, cast(list[object], native))
        else:
            result = cast(object, object.__new__(cls))
        if attributes:
            vars(result).update(attributes)
    else:
        result = partial.__new__(cast(type[partial[object]], cls), cast(Callable[..., object], bound[0]))
        setter = cast(Callable[[object, tuple[object, ...]], None], vars(partial)['__setstate__'])
        setter(result, (*bound, attributes))
    for owner, name, item in slots:
        vars(owner)[name].__set__(result, item)
    return result


def call_implementation(value: object) -> object:
    implementation = type(value).__call__ if callable(value) else None
    return None if implementation is vars(partial)['__call__'] else implementation


def restore_function_metadata(function: FunctionType, source: FunctionType, wrapped: object) -> None:
    function.__module__, function.__qualname__, function.__annotations__ = source.__module__, source.__qualname__, source.__annotations__.copy()
    for name, value in {'__wrapped__': wrapped, '__parallel_round_contract__': vars(source).get('__parallel_round_contract__')}.items():
        if value is not None:
            vars(function)[name] = value


def mutable_function(function: FunctionType, active: frozenset[int] = _EMPTY_IDS) -> bool:
    if id(function) in active:
        return True
    active |= {id(function)}
    def mutable(value: object) -> bool:
        return mutable_function(value, active) if isinstance(value, FunctionType) else any(mutable(item) for item in cast(tuple[object, ...], value)) if isinstance(value, tuple) else not _value_only(value, functions=False)
    return bool(function.__closure__) or mutable(function.__defaults__) or any(mutable(item) for item in (function.__kwdefaults__ or {}).values())

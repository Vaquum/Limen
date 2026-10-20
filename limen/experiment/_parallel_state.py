from __future__ import annotations

from collections.abc import Callable
from functools import partial
from types import MemberDescriptorType
from typing import cast

ObjectState = tuple[dict[str, object], list[tuple[type, str, object]], tuple[object, ...] | None]


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
    state = dict(vars(value)) if hasattr(value, '__dict__') else {}
    return state | {name: item for _, name, item in reversed(slot_state(value))}


def object_state(value: object) -> ObjectState:
    attributes = {key: item for key, item in vars(value).items() if key != '__parallel_round_contract__'} if hasattr(value, '__dict__') else {}
    reduced = cast(tuple[object, ...], partial.__reduce__(value)) if isinstance(value, partial) else None
    bound = cast(tuple[object, ...], reduced[2])[:3] if reduced is not None else None
    return attributes, slot_state(value), bound


def restore_object(cls: type, state: ObjectState) -> object:
    attributes, slots, bound = state
    if bound is None:
        result: object = object.__new__(cls)
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
    return None if implementation is partial.__call__ else implementation

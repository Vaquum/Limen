from collections.abc import Callable, Mapping
import inspect
from typing import Protocol

import polars as pl

__all__: list[str] = []


class DataOwner(Protocol):
    data: object


def source_frame(value: object, method: object, *, origin: str = 'result') -> pl.DataFrame:
    if not isinstance(value, pl.DataFrame):
        name = getattr(method, '__qualname__', type(method).__qualname__)
        raise TypeError(f"DataSourceResolver Source '{name}' {origin} resolved to {type(value).__name__}; expected a materialized polars.DataFrame")
    return value


def source_owner(method: Callable[..., object], params: Mapping[str, object]) -> type[object] | None:
    if inspect.isfunction(method) and '.' in method.__qualname__:
        cls = getattr(inspect.getmodule(method), method.__qualname__.rsplit('.', 1)[0], None)
        if inspect.isclass(cls) and inspect.getattr_static(cls, method.__name__, None) is method:
            receiver = next(iter(inspect.signature(method, follow_wrapped=False).parameters.values()), None)
            if (receiver is None or receiver.name not in params
                    or receiver.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL)):
                return cls
    return None


def source_parameters(method: Callable[..., object], params: Mapping[str, object], dates: Mapping[str, object] | None) -> dict[str, object]:
    values = dict(params)
    signature = inspect.signature(method)
    names = signature.parameters
    variadic = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in names.values())
    if dates is not None and ('start_date_limit' in names or variadic):
        values['start_date_limit'] = dates['train_start']
    if dates is not None and ('end_date_limit' in names or variadic):
        values['end_date_limit'] = dates['test_end']
    return values


def trade_dispatch(method: Callable[..., object]) -> bool:
    return (inspect.isfunction(method) and '.' in method.__qualname__ and '<locals>' not in method.__qualname__
            and next(iter(inspect.signature(method).parameters), None) == 'self')

import math
import re
from collections.abc import Iterable
from numbers import Integral, Real
from typing import cast

__all__: list[str] = []


def integers(*, lower: int | None = None, upper: int | None = None, **values: object) -> None:
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f'{name} must be an integer')
        if lower is not None and value < lower:
            raise ValueError(f'{name} must be at least {lower}')
        if upper is not None and value > upper:
            raise ValueError(f'{name} must be at most {upper}')


def numbers(*, lower: float | None = None, upper: float | None = None,
            positive: bool = False, **values: object) -> None:
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f'{name} must be a finite real number')
        if not isinstance(value, Integral) and not math.isfinite(value):
            raise ValueError(f'{name} must be finite')
        if positive and value <= 0:
            raise ValueError(f'{name} must be positive')
        if lower is not None and value < lower:
            raise ValueError(f'{name} must be at least {lower}')
        if upper is not None and value > upper:
            raise ValueError(f'{name} must be at most {upper}')


def names(**values: object) -> None:
    for name, value in values.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f'{name} must be a non-empty column name')


def booleans(**values: object) -> None:
    for name, value in values.items():
        if not isinstance(value, bool):
            raise ValueError(f'{name} must be a bool')


def integer_sequence(name: str, values: object, *, minimum: int = 0,
                     maximum: int | None = None) -> list[int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(f'{name} must be an iterable of integers')
    result = list(cast(Iterable[object], values))
    for value in result:
        integers(lower=minimum, upper=maximum, **{name: value})
    return cast(list[int], result)


def column_sequence(name: str, values: object) -> None:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError(f'{name} must be a sequence of column names')
    for value in values:
        names(**{name: value})


def ordered(lower_name: str, lower: float, upper_name: str, upper: float) -> None:
    if lower > upper:
        raise ValueError(f'{lower_name} must be less than or equal to {upper_name}')


def choice(name: str, value: object, choices: tuple[str, ...]) -> None:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f'{name} must be one of {choices}')


def interval(value: object) -> None:
    if not isinstance(value, str) or re.fullmatch(r'[1-9][0-9]*[smhd]', value) is None:
        raise ValueError('kline_interval must be a positive integer followed by s, m, h or d')

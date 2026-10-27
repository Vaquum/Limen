import math
import re
from collections.abc import Iterable, Sequence
from numbers import Integral, Real
from typing import cast

__all__: list[str] = []


def integers(*, lower: int | None = None, upper: int | None = None,
             context: str = '', **values: object) -> None:
    for parameter, value in values.items():
        name = f'{context} {parameter}'.strip()
        _integer(name, value, lower, upper)


def _integer(name: str, value: object, lower: int | None, upper: int | None) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer')
    numeric = int(value)
    if lower is not None and numeric < lower:
        raise ValueError(f'{name} must be non-negative' if lower == 0 else f'{name} must be at least {lower}')
    if upper is not None and numeric > upper:
        raise ValueError(f'{name} must be at most {upper}')


def numbers(*, lower: float | None = None, upper: float | None = None,
            positive: bool = False, context: str = '', **values: object) -> None:
    for parameter, value in values.items():
        name = f'{context} {parameter}'.strip()
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f'{name} must be a finite real number')
        if not isinstance(value, Integral) and not math.isfinite(value):
            raise ValueError(f'{name} must be finite')
        numeric = int(value) if isinstance(value, Integral) else float(value)
        if positive and numeric <= 0:
            raise ValueError(f'{name} must be positive')
        if lower is not None and numeric < lower:
            raise ValueError(f'{name} must be non-negative' if lower == 0 else f'{name} must be at least {lower}')
        if upper is not None and numeric > upper:
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
        _integer(name, value, minimum, maximum)
    return cast(list[int], result)


def column_sequence(name: str, values: object) -> None:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError(f'{name} must be a sequence of column names')
    for value in cast(Sequence[object], values):
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

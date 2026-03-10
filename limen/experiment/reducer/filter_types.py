'''
Declarative filter type constants and builders for named filters.

Each constant defines a filter_type string used in intervention dicts.
FILTER_BUILDERS maps these to factory functions that create filter callables
from filter_params dicts. Keys are extracted eagerly so missing params
fail at filter creation time, not during MSQ iteration.

'''

import hashlib
from collections.abc import Callable
from typing import Any

FILTER_EXCLUDE_VALUE = 'exclude_value'
FILTER_KEEP_VALUES = 'keep_values'
FILTER_KEEP_BETWEEN = 'keep_between'
FILTER_SAMPLE = 'sample'


def _build_exclude_value(fp: dict[str, Any]) -> Callable[[dict[str, Any]], bool]:

    param, value = fp['param'], fp['value']
    return lambda c: c[param] == value


def _build_keep_values(fp: dict[str, Any]) -> Callable[[dict[str, Any]], bool]:

    param, values = fp['param'], fp['values']
    return lambda c: c[param] not in values


def _build_keep_between(fp: dict[str, Any]) -> Callable[[dict[str, Any]], bool]:

    param, lower, upper = fp['param'], fp['lower'], fp['upper']
    return lambda c: not (lower <= c[param] <= upper)


def _build_sample(fp: dict[str, Any]) -> Callable[[dict[str, Any]], bool]:

    param, value, fraction = fp['param'], fp['value'], fp['fraction']
    threshold = round(fraction * 10_000)
    return lambda c: (
        c.get(param) == value
        and int.from_bytes(hashlib.blake2b(
            str(sorted(c.items())).encode(), digest_size=8,
        ).digest(), 'big') % 10_000 >= threshold
    )


FILTER_BUILDERS: dict[str, Callable[[dict[str, Any]], Callable[[dict[str, Any]], bool]]] = {
    FILTER_EXCLUDE_VALUE: _build_exclude_value,
    FILTER_KEEP_VALUES: _build_keep_values,
    FILTER_KEEP_BETWEEN: _build_keep_between,
    FILTER_SAMPLE: _build_sample,
}

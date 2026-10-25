from __future__ import annotations

from collections.abc import Mapping
import json
import math

import numpy as np


def _value(value: object, path: str) -> object:
    if isinstance(value, np.generic):
        return ['numpy', str(value.dtype), _value(value.item(), path)]
    if value is None or type(value) in (bool, int, str):
        return [type(value).__name__, value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f'factorize parameter {path}: nonfinite value')
        return ['float', value.hex()]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [_value(item, f'{path}[{index}]') for index, item in enumerate(value)]]
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError(f'factorize parameter {path}: mapping keys must be strings')
        return ['dict', [[key, _value(value[key], path + '.' + str(key))] for key in sorted(value)]]
    raise ValueError(f'factorize parameter {path}: unsupported {type(value).__name__}')


def parameter_key(params: Mapping[str, object]) -> str:
    return json.dumps(_value(params, 'params'), separators=(',', ':'))

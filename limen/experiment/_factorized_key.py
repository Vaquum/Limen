from __future__ import annotations

from collections.abc import Mapping
import json
import math
from typing import Protocol, cast

import numpy as np
from ruamel.yaml.scalarbool import ScalarBoolean
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarint import ScalarInt
from ruamel.yaml.scalarstring import ScalarString


class _Scalar(Protocol):
    @property
    def dtype(self) -> object: ...
    def item(self) -> object: ...


def _value(value: object, path: str) -> object:
    if isinstance(value, (ScalarBoolean, ScalarFloat, ScalarInt, ScalarString)):
        value = bool(value) if isinstance(value, ScalarBoolean) else float(value) if isinstance(value, ScalarFloat) else int(value) if isinstance(value, ScalarInt) else str(value)
    if isinstance(value, np.generic):
        scalar = cast(_Scalar, value)
        return ['numpy', str(scalar.dtype), _value(scalar.item(), path)]
    if value is None or type(value) in (bool, int, str):
        return [type(value).__name__, value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f'factorize parameter {path}: nonfinite value')
        return ['float', value.hex()]
    if isinstance(value, (list, tuple)):
        return ['list' if isinstance(value, list) else 'tuple', [_value(item, f'{path}[{index}]') for index, item in enumerate(cast(list[object] | tuple[object, ...], value))]]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in cast(Mapping[object, object], value)):
            raise ValueError(f'factorize parameter {path}: mapping keys must be strings')
        mapping = cast(Mapping[str, object], value)
        return ['dict', [[key, _value(mapping[key], path + '.' + key)] for key in sorted(mapping)]]
    raise ValueError(f'factorize parameter {path}: unsupported {type(value).__name__}')


def parameter_key(params: Mapping[str, object]) -> str:
    return json.dumps(_value(params, 'params'), separators=(',', ':'))

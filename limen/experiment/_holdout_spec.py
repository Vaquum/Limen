from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from hashlib import sha256
import json
import math
import numbers
from typing import cast

from limen.experiment._holdout_declaration import declaration_witness

__all__ = ['digest', 'encoded', 'normalized']


def normalized(value: object) -> object:
    result: object
    if callable(value):
        result = {'callable': f'{getattr(value, "__module__", type(value).__module__)}.{getattr(value, "__qualname__", type(value).__qualname__)}',
                  'implementation': sha256(declaration_witness(value)).hexdigest()}
    elif value is None or type(value) in (str, bool, int):
        result = value
    elif isinstance(value, numbers.Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('Holdout specification must be finite')
    elif isinstance(value, (date, datetime)):
        result = value.isoformat()
    elif isinstance(value, Mapping):
        items = cast(Mapping[object, object], value)
        if not all(isinstance(key, str) for key in items):
            raise ValueError('Holdout specification requires string keys')
        result = {str(key): normalized(item) for key, item in items.items()}
    elif isinstance(value, (list, tuple)):
        result = [normalized(item) for item in cast(list[object] | tuple[object, ...], value)]
    elif is_dataclass(value) and not isinstance(value, type) and type(value).__module__.startswith('limen.'):
        result = {field.name: normalized(getattr(value, field.name)) for field in fields(value)}
    else:
        raise ValueError(f'Unsupported holdout specification value: {type(value).__name__}')
    return result


def encoded(value: object) -> bytes:
    return json.dumps(normalized(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value: object) -> str:
    return sha256(encoded(value)).hexdigest()


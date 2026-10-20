from __future__ import annotations

import hashlib
import inspect
import json
import math
from ast import literal_eval
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from functools import partial
from importlib import import_module
from io import BytesIO
from pathlib import Path
from typing import cast

import polars as pl

from limen.experiment._parallel_identity import module_identity
from limen.experiment._parallel_state import ObjectState, call_implementation, component_state, object_state, restore_object

VERSION = 1


def reference(value: object) -> list[object]:
    module, name = getattr(value, '__module__', None), getattr(value, '__qualname__', None)
    if not isinstance(module, str) or module == '__main__' or not isinstance(name, str) or '<' in name:
        raise ValueError('Parallel components require importable module-qualified identities')
    resolved: object = import_module(module)
    for part in name.split('.'):
        resolved = getattr(resolved, part)
    resolved = resolved.__func__ if inspect.ismethod(resolved) else resolved
    if resolved is not value:
        raise ValueError('Parallel component differs from its importable identity')
    return [module_identity(module), name]


_EMPTY_IDS: frozenset[int] = frozenset()


def encode(value: object, active: frozenset[int] = _EMPTY_IDS) -> list[object]:
    if callable(value):
        return _encode_composite(value, active)
    if value is None:
        node: list[object] = ['null']
    elif isinstance(value, str):
        node = ['str', str(value)]
    elif type(value) is bool:
        node = ['bool', value]
    elif isinstance(value, int):
        node = ['int', str(value)]
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Parallel encoding rejects nonfinite numbers')
        node = ['float', value.hex()]
    elif isinstance(value, (date, datetime)):
        node = [type(value).__name__, value.isoformat()]
    elif isinstance(value, Path):
        node = ['path', str(value)]
    else:
        node = _encode_composite(value, active)
    return node


def _encode_composite(value: object, active: frozenset[int]) -> list[object]:
    if id(value) in active:
        raise ValueError('Parallel encoding rejects cycles')
    active |= {id(value)}
    if callable(value):
        return _encode_component(value, active)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if all(isinstance(key, str) for key in mapping):
            return ['map', [[key, encode(mapping[key], active)] for key in sorted(cast(Mapping[str, object], mapping))]]
        pairs = [[encode(key, active), encode(item, active)] for key, item in mapping.items()]
        return ['keyed_map', sorted(pairs, key=lambda pair: json.dumps(pair[0], separators=(',', ':'))) ]
    if isinstance(value, (list, tuple)):
        return ['tuple' if isinstance(value, tuple) else 'list',
                [encode(item, active) for item in cast(list[object] | tuple[object, ...], value)]]
    return _encode_component(value, active)



def _encode_component(value: object, active: frozenset[int]) -> list[object]:
    if inspect.ismethod(value):
        return ['method', encode(value.__self__, active), value.__name__, encode(value.__func__, active)]
    if inspect.isfunction(value) or isinstance(value, type):
        return ['import', reference(value)]
    if is_dataclass(value) and not isinstance(value, (type, partial)):
        names = {field.name for field in fields(value)}
        if set(component_state(value)) - names - {'__parallel_round_contract__'}:
            raise ValueError('Parallel transport rejects undeclared runtime state')
        return ['dataclass', reference(type(value)),
                [[field.name, encode(getattr(value, field.name), active)] for field in fields(value)], encode(call_implementation(value), active)]
    if callable(value):
        return ['instance', reference(type(value)), encode(object_state(value), active), encode(call_implementation(value), active)]
    raise ValueError(f'Parallel encoding rejects unsupported or fitted state: {type(value).__name__}')


def _import(node: object) -> object:
    identity, name = cast(list[object], node)
    module_name = cast(list[str], identity)[0]
    if module_identity(module_name) != identity:
        raise ValueError('Parallel imported source or dependency identity changed')
    value: object = import_module(module_name)
    for part in cast(str, name).split('.'):
        value = getattr(value, part)
    return value.__func__ if inspect.ismethod(value) else value


_ARITY = {'null': 1, 'str': 2, 'bool': 2, 'int': 2, 'float': 2, 'date': 2, 'datetime': 2,
          'path': 2, 'list': 2, 'tuple': 2, 'map': 2, 'keyed_map': 2, 'import': 2,
          'method': 4, 'dataclass': 4, 'instance': 4}


def decode(node: object) -> object:
    if not isinstance(node, list) or not node or not isinstance(node[0], str):
        raise ValueError('Invalid version1 parallel parameter encoding')
    parts = cast(list[object], node)
    tag = cast(str, parts[0])
    if tag not in _ARITY or len(parts) != _ARITY[tag]:
        raise ValueError('Invalid version1 parallel parameter payload')
    if tag == 'null':
        result: object = None
    elif tag in ('str', 'bool') and type(parts[1]) is (str if tag == 'str' else bool):
        result = parts[1]
    elif tag in ('int', 'float') and isinstance(value := parts[1], str):
        result = int(value) if tag == 'int' else float.fromhex(value)
    elif tag in ('date', 'datetime', 'path') and isinstance(value := parts[1], str):
        result = date.fromisoformat(value) if tag == 'date' else datetime.fromisoformat(value) if tag == 'datetime' else Path(value)
    else:
        result = _decode_composite(parts)
    if encode(result) != parts:
        raise ValueError('Noncanonical parallel parameter encoding')
    return result


def _decode_composite(parts: list[object]) -> object:
    tag, value = parts[:2]
    if tag in ('list', 'tuple') and isinstance(value, list):
        items = [decode(item) for item in cast(list[object], value)]
        return tuple(items) if tag == 'tuple' else items
    if tag in ('map', 'keyed_map') and isinstance(value, list):
        return {cast(str, pair[0]) if tag == 'map' else decode(pair[0]): decode(pair[1]) for pair in cast(list[list[object]], value)}
    if tag == 'import':
        return _import(value)
    if tag == 'method':
        method = getattr(decode(value), cast(str, parts[2]))
        if not inspect.ismethod(method) or method.__func__ is not decode(parts[3]):
            raise ValueError('Parallel bound method implementation changed')
        return method
    return _decode_instance(parts)


def _decode_instance(parts: list[object]) -> object:
    tag, value, state, implementation = parts
    if tag == 'dataclass':
        cls = cast(type, _import(value))
        params = {cast(str, key): decode(item) for key, item in cast(list[list[object]], state)}
        if set(params) != {field.name for field in fields(cls)}:
            raise ValueError('Invalid parallel dataclass fields')
        instance = cast(object, object.__new__(cls))
        for field in fields(cls):
            object.__setattr__(instance, field.name, params[field.name])
    elif tag == 'instance':
        instance = restore_object(cast(type, _import(value)), cast(ObjectState, decode(state)))
    else:
        raise ValueError('Invalid or unsupported version1 parallel parameter tag')
    if encode(call_implementation(instance)) != implementation:
        raise ValueError('Parallel callable implementation changed')
    return instance


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(encode(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def parameter_projection(node: list[object]) -> object:
    tag = node[0]
    if tag in ('null', 'str', 'bool', 'int', 'float'):
        return decode(node)
    if tag == 'map':
        return {cast(str, key): parameter_projection(cast(list[object], value)) for key, value in cast(list[list[object]], node[1])}
    if tag in ('tuple', 'list'):
        return [parameter_projection(cast(list[object], value)) for value in cast(list[object], node[1])]
    return node


def checkpoint_payload(data: Mapping[str, object], version: int | None, *, restore: bool = False) -> dict[str, object]:
    result = dict(data)
    if version is None:
        return result
    if type(version) is not int or version != VERSION:
        raise ValueError('Cannot resume: missing or inconsistent parallel evidence version')
    result['parameter_encoding'] = VERSION
    for key in ('msq_state', 'domain_state', 'failed_round_params'):
        if key in result:
            result[key] = decode(result[key]) if restore else encode(result[key])
    return result


def source_digest(frames: tuple[pl.DataFrame | None, ...]) -> str:
    identity: list[object] = []
    for role, frame in zip(('learning', 'execution', 'funding'), frames, strict=True):
        if frame is None:
            identity.append((role, None))
        else:
            stream = BytesIO()
            frame.rechunk().write_ipc(stream, compression='uncompressed')
            identity.append((role, pl.__version__, list(zip(frame.columns, map(str, frame.dtypes), strict=True)), hashlib.sha256(stream.getvalue()).hexdigest()))
    return digest(identity)


def parameter_matches(cell: str | None, value: object, *, legacy: bool = False) -> bool:
    if cell is None or value is None:
        return cell == ''
    if type(value) is bool:
        return cell.lower() == str(value).lower()
    try:
        if type(value) is int:
            return Decimal(cell) == value
        if type(value) is float:
            return float(cell) == value
        if isinstance(value, (list, tuple, dict)):
            parsed = literal_eval(cell)
            return json.loads(json.dumps(parsed)) == value if legacy else encode(parsed) == encode(cast(object, value))
    except (ValueError, SyntaxError, InvalidOperation) as exc:
        raise ValueError('Cannot resume parallel run with malformed CSV parameter evidence') from exc
    return cell == str(value)

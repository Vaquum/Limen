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
from functools import cache, partial
from importlib import import_module, metadata
from io import BytesIO
from pathlib import Path
from typing import cast

import polars as pl

VERSION = 1


@cache
def _distributions() -> dict[str, list[str]]:
    return metadata.packages_distributions()


def module_identity(name: str) -> list[object]:
    spec = inspect.getmodule(import_module(name))
    path = None if spec is None else spec.__file__
    if path is None:
        raise ValueError(f'Parallel import has no source file: {name}')
    top = name.split('.')[0]
    distributions = _distributions().get(top, []) if top not in ('limen', 'tests') else []
    return [name, hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            [[name, metadata.version(name)] for name in sorted(distributions)]]


def reference(value: object) -> list[object]:
    module, name = getattr(value, '__module__', None), getattr(value, '__qualname__', None)
    if not isinstance(module, str) or module == '__main__' or not isinstance(name, str) or '<' in name:
        raise ValueError('Parallel components require importable module-qualified identities')
    resolved: object = import_module(module)
    for part in name.split('.'):
        resolved = getattr(resolved, part)
    if resolved is not value:
        raise ValueError('Parallel component differs from its importable identity')
    return [module_identity(module), name]


def encode(value: object, active: frozenset[int] = frozenset()) -> list[object]:
    if value is None:
        return ['null']
    if isinstance(value, str):
        return ['str', str(value)]
    if type(value) is bool:
        return ['bool', value]
    if type(value) is int:
        return ['int', str(value)]
    if type(value) is float:
        if not math.isfinite(cast(float, value)):
            raise ValueError('Parallel encoding rejects nonfinite numbers')
        return ['float', cast(float, value).hex()]
    if isinstance(value, (date, datetime)):
        return [type(value).__name__, value.isoformat()]
    if isinstance(value, Path):
        return ['path', str(value)]
    if id(value) in active:
        raise ValueError('Parallel encoding rejects cycles')
    active |= {id(value)}
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if any(not isinstance(key, str) for key in mapping):
            raise ValueError('Parallel mappings require string keys')
        return ['map', [[key, encode(mapping[key], active)] for key in sorted(cast(Mapping[str, object], mapping))]]
    if isinstance(value, (list, tuple)):
        return ['tuple' if isinstance(value, tuple) else 'list',
                [encode(item, active) for item in cast(list[object] | tuple[object, ...], value)]]
    if isinstance(value, partial):
        return ['partial', encode(value.func, active), encode(value.args, active), encode(value.keywords, active)]
    if inspect.ismethod(value):
        return ['method', encode(value.__self__, active), value.__name__]
    if inspect.isfunction(value) or isinstance(value, type):
        return ['import', reference(value)]
    if is_dataclass(value) and not isinstance(value, type):
        return ['dataclass', reference(type(value)),
                [[field.name, encode(getattr(value, field.name), active)] for field in fields(value)]]
    if callable(value) and hasattr(value, '__dict__'):
        return ['instance', reference(type(value)), encode(vars(value), active)]
    raise ValueError(f'Parallel encoding rejects unsupported or fitted state: {type(value).__name__}')


def _import(node: object) -> object:
    identity, name = cast(list[object], node)
    module_name = cast(list[str], identity)[0]
    if module_identity(module_name) != identity:
        raise ValueError('Parallel imported source or dependency identity changed')
    value: object = import_module(module_name)
    for part in cast(str, name).split('.'):
        value = getattr(value, part)
    return value


def decode(node: object) -> object:
    if not isinstance(node, list) or not node or not isinstance(node[0], str):
        raise ValueError('Invalid version1 parallel parameter encoding')
    parts = cast(list[object], node)
    tag = parts[0]
    if tag == 'null' and len(parts) == 1:
        return None
    if len(parts) < 2:
        raise ValueError('Invalid version1 parallel parameter payload')
    value = parts[1]
    if tag in ('str', 'bool') and type(value) is (str if tag == 'str' else bool):
        return value
    if tag in ('int', 'float') and isinstance(value, str):
        result = int(value) if tag == 'int' else float.fromhex(value)
        if encode(result) != parts:
            raise ValueError('Noncanonical parallel number')
        return result
    if tag in ('date', 'datetime', 'path') and isinstance(value, str):
        return date.fromisoformat(value) if tag == 'date' else datetime.fromisoformat(value) if tag == 'datetime' else Path(value)
    if tag in ('list', 'tuple') and isinstance(value, list):
        result = [decode(item) for item in cast(list[object], value)]
        return tuple(result) if tag == 'tuple' else result
    if tag == 'map' and isinstance(value, list):
        pairs = cast(list[list[object]], value)
        result = {cast(str, pair[0]): decode(pair[1]) for pair in pairs}
        if encode(result) != parts:
            raise ValueError('Noncanonical parallel mapping')
        return result
    if tag == 'import':
        return _import(value)
    if tag == 'partial' and len(parts) == 4:
        return partial(cast(type, decode(value)), *cast(tuple[object, ...], decode(parts[2])),
                       **cast(dict[str, object], decode(parts[3])))
    if tag == 'method' and len(parts) == 3:
        return getattr(decode(value), cast(str, parts[2]))
    if tag == 'dataclass' and len(parts) == 3:
        return cast(type, _import(value))(**{cast(str, key): decode(item) for key, item in cast(list[list[object]], parts[2])})
    if tag == 'instance' and len(parts) == 3:
        instance = object.__new__(cast(type, _import(value)))
        vars(instance).update(cast(dict[str, object], decode(parts[2])))
        return instance
    raise ValueError('Invalid or unsupported version1 parallel parameter tag')


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(encode(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


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
            frame.write_ipc(stream, compression='uncompressed')
            identity.append((role, pl.__version__, list(zip(frame.columns, map(str, frame.dtypes), strict=True)), hashlib.sha256(stream.getvalue()).hexdigest()))
    return digest(identity)


def parameter_matches(cell: str | None, value: object) -> bool:
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
            return encode(literal_eval(cell)) == encode(value)
    except (ValueError, SyntaxError, InvalidOperation) as exc:
        raise ValueError('Cannot resume parallel run with malformed CSV parameter evidence') from exc
    return cell == str(value)

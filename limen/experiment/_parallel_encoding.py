from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from functools import partial
from importlib import import_module
from io import BytesIO
from pathlib import Path
from types import CellType, FunctionType, MethodType
from typing import cast

import polars as pl

from limen.experiment._parallel_identity import declaration_changed, identity_snapshot, module_identity, restore_function_metadata
from limen.experiment._parallel_callable import bind_call, source_class
from limen.experiment._parallel_graph import EMPTY_IDS as _EMPTY_IDS, copied_source, decode_graph, encode_graph, has_decoded_identity, isolated_encoding
from limen.experiment._parallel_state import ObjectState, call_implementation, component_state, mutable_function, object_state, restore_object

VERSION = 1
_CAPTURED_CALL_ARITY = 2


def reference(value: object) -> list[object]:
    value = source_class(value) if isinstance(value, type) else value
    module, name = getattr(value, '__module__', None), getattr(value, '__qualname__', None)
    if not isinstance(module, str) or module == '__main__' or not isinstance(name, str) or '<' in name:
        raise ValueError('Parallel components require importable module-qualified identities')
    resolved: object = import_module(module)
    for part in name.split('.'):
        resolved = getattr(resolved, part)
    resolved = resolved.__func__ if inspect.ismethod(resolved) else resolved
    while inspect.isfunction(resolved) and inspect.isfunction(value) and resolved.__code__ is not value.__code__ and hasattr(resolved, '__wrapped__'):
        name, resolved = name + '.__wrapped__', vars(resolved)['__wrapped__']
    if resolved is not value and not (inspect.isfunction(value) and inspect.isfunction(resolved)
                                     and value.__code__ is resolved.__code__ and value.__globals__ is resolved.__globals__):
        raise ValueError('Parallel component differs from its importable identity')
    return [module_identity(module), name]


def encode(value: object, active: frozenset[int] = _EMPTY_IDS, *, ordered: bool = False) -> list[object]:
    if callable(value):
        return encode_graph(value, lambda: _encode_composite(value, active, ordered=ordered))
    if isinstance(value, CellType):
        return encode_graph(value, lambda: ['cell', encode(value.cell_contents, active, ordered=ordered)])
    if isinstance(value, defaultdict):
        return encode_graph(cast(object, value), lambda: _encode_component(cast(object, value), active | {id(cast(object, value))}, owner_only=True))
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
        return encode_graph(value, lambda: _encode_composite(value, active, ordered=ordered))
    return node


def _encode_composite(value: object, active: frozenset[int], *, ordered: bool) -> list[object]:
    if id(value) in active:
        raise ValueError('Parallel encoding rejects cycles')
    active |= {id(value)}
    if callable(value):
        return _encode_component(value, active)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if ordered:
            return ['state_map', [[encode(key, active, ordered=True), encode(item, active, ordered=True)] for key, item in mapping.items()]]
        if all(isinstance(key, str) for key in mapping):
            return ['map', [[key, encode(mapping[key], active)] for key in sorted(cast(Mapping[str, object], mapping))]]
        with isolated_encoding():
            keys = sorted(mapping, key=lambda key: json.dumps(encode(key), separators=(',', ':')))
        return ['keyed_map', [[encode(key, active), encode(mapping[key], active)] for key in keys]]
    if isinstance(value, (list, tuple)):
        return ['tuple' if isinstance(value, tuple) else 'list',
                [encode(item, active, ordered=ordered) for item in cast(list[object] | tuple[object, ...], value)]]
    return _encode_component(value, active)



def _encode_component(value: object, active: frozenset[int], *, owner_only: bool = False) -> list[object]:
    if inspect.ismethod(value):
        owner = encode_graph(value.__self__, lambda: _encode_component(value.__self__, active | {id(value.__self__)}, owner_only=True))
        return ['method', owner, value.__name__, encode(value.__func__, active)]
    if inspect.isfunction(value):
        return ['function', reference(value), encode(value.__defaults__, active, ordered=True), encode(value.__kwdefaults__, active, ordered=True), encode(value.__closure__, active, ordered=True), encode(vars(value).get('__wrapped__'), active, ordered=True)]
    if isinstance(value, type):
        return ['import', reference(value)]
    if is_dataclass(value) and not isinstance(value, (type, partial)):
        names = {field.name for field in fields(value)}
        if set(component_state(value)) - names - {'__parallel_round_contract__'}:
            raise ValueError('Parallel transport rejects undeclared runtime state')
        return ['dataclass', reference(type(value)),
                [[field.name, encode(getattr(value, field.name), active, ordered=True)] for field in fields(value)], _call_node(value, active, owner_only)]
    if callable(value) or owner_only:
        return ['instance', reference(type(value)), encode(object_state(value), active, ordered=True), _call_node(value, active, owner_only)]
    raise ValueError(f'Parallel encoding rejects unsupported or fitted state: {type(value).__name__}')


def _call_node(value: object, active: frozenset[int], owner_only: bool) -> list[object]:
    implementation = call_implementation(value)
    return ['captured_call', encode((implementation.__func__ if inspect.ismethod(implementation) else implementation, getattr(value, '__parallel_round_contract__', None)), active)] if source_class(type(value)) is not type(value) else _owner_call(type(value)) if owner_only and implementation is not None else encode(None if owner_only else implementation, active)


def _owner_call(cls: type) -> list[object]:
    implementation = cls.__call__
    return ['owner_call', reference(cls), module_identity(implementation.__module__), implementation.__qualname__]


def import_reference(node: object) -> object:
    identity, name = cast(list[object], node)
    if module_identity(module_name := cast(list[str], identity)[0]) != identity:
        raise ValueError('Parallel imported source or dependency identity changed')
    value: object = import_module(module_name)
    for part in cast(str, name).split('.'):
        value = getattr(value, part)
    return value.__func__ if inspect.ismethod(value) else value


_ARITY = {'null': 1, 'str': 2, 'bool': 2, 'int': 2, 'float': 2, 'date': 2, 'datetime': 2, 'path': 2, 'list': 2, 'tuple': 2, 'map': 2, 'keyed_map': 2, 'state_map': 2, 'import': 2,
          'function': 6, 'method': 4, 'dataclass': 4, 'instance': 4, 'cell': 2, 'owner_call': 4}


def decode(node: object, *, ordered: bool = False, isolate_functions: bool = False) -> object:
    with identity_snapshot():
        return decode_graph(node, lambda: _decode(node, ordered=ordered, isolate_functions=isolate_functions), lambda value: encode(value, ordered=ordered))


def _decode(node: object, *, ordered: bool, isolate_functions: bool) -> object:
    if not isinstance(node, list) or not node or not isinstance(node[0], str):
        raise ValueError('Invalid version1 parallel parameter encoding')
    parts = cast(list[object], node)
    tag = cast(str, parts[0])
    if tag not in _ARITY or len(parts) != _ARITY[tag]:
        raise ValueError('Invalid version1 parallel parameter payload')
    if tag == 'null':
        return None
    if tag in ('str', 'bool') and type(parts[1]) is (str if tag == 'str' else bool):
        return parts[1]
    if tag in ('int', 'float') and isinstance(value := parts[1], str):
        return int(value) if tag == 'int' else float.fromhex(value)
    if tag in ('date', 'datetime', 'path') and isinstance(value := parts[1], str):
        return date.fromisoformat(value) if tag == 'date' else datetime.fromisoformat(value) if tag == 'datetime' else Path(value)
    return _decode_composite(parts, ordered=ordered, isolate_functions=isolate_functions)


def _decode_composite(parts: list[object], *, ordered: bool, isolate_functions: bool) -> object:
    tag, value = parts[:2]
    if tag in ('list', 'tuple') and isinstance(value, list):
        items = [decode(item, ordered=ordered, isolate_functions=isolate_functions) for item in cast(list[object], value)]
        return tuple(items) if tag == 'tuple' else items
    if tag in ('map', 'keyed_map', 'state_map') and isinstance(value, list):
        return {cast(str, pair[0]) if tag == 'map' else decode(pair[0], ordered=ordered, isolate_functions=isolate_functions): decode(pair[1], ordered=ordered, isolate_functions=isolate_functions) for pair in cast(list[list[object]], value)}
    if tag in ('cell', 'owner_call', 'import'):
        return _decode_reference(parts, ordered=ordered, isolate_functions=isolate_functions)
    if tag == 'function':
        source, original = import_reference(value), copied_source()
        if not inspect.isfunction(source):
            raise ValueError('Parallel function implementation changed')
        defaults, keywords, closure, wrapped = [decode(item, ordered=True, isolate_functions=isolate_functions) for item in parts[2:]]
        function = source
        if isolate_functions or has_decoded_identity(source) or mutable_function(source) or declaration_changed(source, original, encode) or encode(source) != parts:
            function = FunctionType(source.__code__, source.__globals__, source.__name__, cast(tuple[object, ...] | None, defaults), cast(tuple[CellType, ...] | None, closure))
            function.__kwdefaults__ = cast(dict[str, object] | None, keywords)
            restore_function_metadata(function, source, wrapped)
        return function
    if tag == 'method':
        owner = (decode_graph(cast(object, value), lambda: _decode_instance(cast(list[object], value), isolate_functions=isolate_functions, owner_only=True), lambda owner: encode(owner))
                 if isinstance(value, list) and value and value[0] in ('dataclass', 'instance') else decode(cast(object, value), isolate_functions=isolate_functions))
        function = decode(parts[3], isolate_functions=isolate_functions)
        if not inspect.isfunction(function):
            raise ValueError('Parallel bound method implementation changed')
        return MethodType(function, owner)
    return _decode_instance(parts, isolate_functions=isolate_functions)


def _decode_reference(parts: list[object], *, ordered: bool, isolate_functions: bool) -> object:
    tag, value = parts[:2]
    if tag == 'cell':
        return CellType(decode(value, ordered=ordered, isolate_functions=isolate_functions))
    if tag == 'owner_call':
        cls = cast(type, import_reference(value))
        if _owner_call(cls) != parts:
            raise ValueError('Parallel callable implementation changed')
        return cls.__call__
    return import_reference(value)


def _decode_instance(parts: list[object], *, isolate_functions: bool, owner_only: bool = False) -> object:
    tag, value, state, implementation = parts
    if tag == 'dataclass':
        cls = cast(type, import_reference(value))
        params = {cast(str, key): decode(item, ordered=True, isolate_functions=isolate_functions) for key, item in cast(list[list[object]], state)}
        if set(params) != {field.name for field in fields(cls)}:
            raise ValueError('Invalid parallel dataclass fields')
        instance = cast(object, object.__new__(cls))
        for field in fields(cls):
            object.__setattr__(instance, field.name, params[field.name])
    elif tag == 'instance':
        instance = restore_object(cast(type, import_reference(value)), cast(ObjectState, decode(state, ordered=True, isolate_functions=isolate_functions)))
    else:
        raise ValueError('Invalid or unsupported version1 parallel parameter tag')
    call: list[object] = cast(list[object], implementation) if isinstance(implementation, list) else []
    frozen = len(call) == _CAPTURED_CALL_ARITY and call[0] == 'captured_call'
    captured = decode(call[1] if frozen else cast(object, implementation), isolate_functions=isolate_functions)
    if frozen:
        bind_call(instance, captured, digest)
    elif not owner_only and encode(call_implementation(instance)) != encode(captured):
        raise ValueError('Parallel callable implementation changed')
    return instance


@identity_snapshot()
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
            frame.rechunk().write_ipc(stream, compression='uncompressed')
            identity.append((role, pl.__version__, list(zip(frame.columns, map(str, frame.dtypes), strict=True)), hashlib.sha256(stream.getvalue()).hexdigest()))
    return digest(identity)

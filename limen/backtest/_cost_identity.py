from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import cast

import polars as pl

from limen.backtest.trade_contract import FundingPolicy, JsonValue, ProductSpec, SourceBinding, TradeInputs, TradePolicy

_RECORDS = (TradeInputs, TradePolicy, ProductSpec, SourceBinding, FundingPolicy)
_PRIMITIVES = (pl.Boolean, pl.String, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
               pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64)


def _frame(frame: pl.DataFrame, active: set[int]) -> JsonValue:
    schema: list[JsonValue] = []
    columns: list[list[JsonValue]] = []
    for name, dtype in frame.schema.items():
        temporal = True
        tag: JsonValue
        if dtype == pl.Date:
            tag = ['Date', 'd', None]
        elif dtype == pl.Time:
            tag = ['Time', 'ns', None]
        elif isinstance(dtype, pl.Datetime):
            tag = ['Datetime', dtype.time_unit, dtype.time_zone]
        elif isinstance(dtype, pl.Duration):
            tag = ['Duration', dtype.time_unit, None]
        elif dtype.base_type() in _PRIMITIVES:
            tag, temporal = str(dtype), False
        else:
            raise ValueError(f'Unsupported canonical frame dtype: {dtype}')
        values = frame[name].cast(pl.Int64) if temporal else frame[name]
        schema.append([name, tag])
        columns.append([_node(value, active) for value in values])
    rows: list[JsonValue] = [[column[row] for column in columns] for row in range(frame.height)]
    return ['frame', schema, rows]


def _scalar(value: object) -> JsonValue:
    result: JsonValue = None
    if value is None:
        result = ['null']
    elif type(value) is bool:
        result = ['bool', value]
    elif type(value) is str:
        result = ['str', value]
    elif type(value) is int:
        result = ['int', str(value)]
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError('Nonfinite canonical float')
        result = ['float', value.hex()]
    elif isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError('Nonfinite canonical Decimal')
        result = ['decimal', str(value)]
    return result


def _node(value: object, active: set[int]) -> JsonValue:
    scalar = _scalar(value)
    if scalar is not None:
        return scalar
    identity = id(value)
    if identity in active:
        raise ValueError('Cyclic canonical value')
    active.add(identity)
    try:
        if isinstance(value, pl.DataFrame):
            return _frame(value, active)
        if isinstance(value, Mapping):
            mapping = cast(Mapping[object, object], value)
            if any(type(key) is not str for key in mapping):
                raise TypeError('Canonical mapping keys must be strings')
            ordered = cast(Mapping[str, object], mapping)
            return ['map', [[key, _node(ordered[key], active)] for key in sorted(ordered)]]
        if type(value) in (list, tuple):
            return [type(value).__name__, [_node(item, active) for item in cast(list[object] | tuple[object, ...], value)]]
        if type(value) in _RECORDS:
            record = cast(TradeInputs | TradePolicy | ProductSpec | SourceBinding | FundingPolicy, value)
            name = type(record).__module__ + '.' + type(record).__qualname__
            return ['record', name, [[field.name, _node(getattr(record, field.name), active)]
                                     for field in dataclasses.fields(record)]]
        raise TypeError(f'Unsupported canonical value: {type(value).__name__}')
    finally:
        active.remove(identity)


def canonical(value: object) -> bytes:
    return json.dumps(_node(value, set()), ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _constant(value: str) -> object:
    raise ValueError(f'Nonfinite JSON value: {value}')


def read_json(path: str | Path) -> object:
    value: object = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_pairs, parse_constant=_constant)
    return value


def publish(path: str | Path, value: object) -> None:
    target = Path(path)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if target.exists():
        if target.read_bytes() == payload:
            return
        raise FileExistsError(f'Different execution-cost artifact already exists: {target}')
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.execution-cost-', delete=False) as stream:
        temporary = Path(stream.name)
        _ = stream.write(payload)
    try:
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise
    finally:
        temporary.unlink()


__all__: list[str] = []

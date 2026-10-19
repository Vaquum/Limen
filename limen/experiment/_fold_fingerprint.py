from __future__ import annotations

from collections.abc import Iterable, Sequence
from hashlib import sha256
from itertools import pairwise
import math
import struct
from typing import cast

import polars as pl

__all__: list[str] = []


class FoldEvidenceError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code

    def record(self) -> dict[str, object]:
        return {'version': 1, 'unavailable_reason': {'code': self.code, 'detail': str(self)}}


_FORMATS = {'i8': 'b', 'i16': 'h', 'i32': 'i', 'i64': 'q',
            'u8': 'B', 'u16': 'H', 'u32': 'I', 'u64': 'Q',
            'f32': 'f', 'f64': 'd', 'date': 'i', 'datetime_utc_ns': 'q', 'bool': '?'}
_TAGS: dict[object, str] = {pl.Boolean: 'bool', pl.Int8: 'i8', pl.Int16: 'i16', pl.Int32: 'i32', pl.Int64: 'i64',
         pl.UInt8: 'u8', pl.UInt16: 'u16', pl.UInt32: 'u32', pl.UInt64: 'u64',
         pl.Float32: 'f32', pl.Float64: 'f64', pl.String: 'utf8', pl.Date: 'date'}


def schema(frame: pl.DataFrame) -> list[list[str]]:
    fields: list[list[str]] = []
    for name, dtype in frame.schema.items():
        if isinstance(dtype, pl.Datetime) and dtype.time_zone == 'UTC':
            tag = 'datetime_utc_ns'
        else:
            tag = _TAGS.get(dtype)
        if tag is None:
            raise FoldEvidenceError('dtype', f'Unsupported fold fingerprint dtype: {name}={dtype}')
        fields.append([name, tag])
    return fields


def _text(value: str) -> bytes:
    encoded = value.encode('utf-8')
    return struct.pack('>I', len(encoded)) + encoded


def _scalar(value: object, tag: str) -> bytes:
    if value is None:
        return b'N'
    if isinstance(value, float) and not math.isfinite(value):
        return b'Q' if math.isnan(value) else b'P' if value > 0 else b'M'
    return b'V' + (_text(str(value)) if tag == 'utf8' else struct.pack('>' + _FORMATS[tag], value))


def fingerprint(frame: pl.DataFrame) -> str:
    fields = schema(frame)
    digest = sha256(b'limen.fold.source.v1\0')
    digest.update(struct.pack('>QI', frame.height, frame.width))
    for name, tag in fields:
        digest.update(_text(name) + _text(tag))
    normalized = frame.with_columns([
        pl.col(name).dt.epoch('ns') if tag == 'datetime_utc_ns' else pl.col(name).cast(pl.Int32)
        for name, tag in fields if tag in ('datetime_utc_ns', 'date')
    ])
    for row in normalized.iter_rows():
        digest.update(b''.join(_scalar(value, tag) for value, (_, tag) in zip(row, fields, strict=True)))
    return digest.hexdigest()


def membership(groups: Sequence[Sequence[int]]) -> str:
    digest = sha256(b'limen.fold.rows.v1\0' + struct.pack('>Q', len(groups)))
    for group in groups:
        digest.update(struct.pack('>Q', len(group)))
        digest.update(b''.join(struct.pack('>Q', index) for index in group))
    return digest.hexdigest()


def clock(frame: pl.DataFrame) -> list[int]:
    if 'datetime' not in frame.columns:
        raise FoldEvidenceError('clock', 'Fold source requires UTC datetime')
    times = frame['datetime']
    if not isinstance(times.dtype, pl.Datetime) or times.dtype.time_zone != 'UTC' or times.null_count():
        raise FoldEvidenceError('clock', 'Fold source requires nonnull UTC datetime')
    values = cast(list[int], times.dt.epoch('ns').to_list())
    if not values or any(value % 1000 for value in values):
        raise FoldEvidenceError('clock', 'Fold source requires nonempty microsecond-representable timestamps')
    return values


def interval(frame: pl.DataFrame, declared_seconds: object = None) -> int:
    declared: list[int] = []
    seconds_values = ([] if declared_seconds is None else [declared_seconds]) + ([] if 'base_interval' not in frame.columns else frame['base_interval'].unique().to_list())
    for seconds in seconds_values:
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0 or seconds * 1_000_000_000 != int(seconds * 1_000_000_000):
            raise FoldEvidenceError('clock', 'Fold interval must be positive, constant, exact seconds')
        declared.append(int(seconds * 1_000_000_000))
    unique = sorted(set(clock(frame)))
    gaps = [b - a for a, b in pairwise(unique)]
    if declared and len(set(declared)) != 1:
        raise FoldEvidenceError('clock', 'Conflicting native interval declarations')
    step = declared[0] if declared else min(gaps, default=0)
    if step <= 0 or any(gap % step for gap in gaps):
        raise FoldEvidenceError('clock', 'Unresolvable native interval')
    return step


def source(frame: pl.DataFrame, declared_seconds: object = None) -> dict[str, object]:
    times = clock(frame)
    return {'rows': frame.height, 'schema': schema(frame), 'digest': fingerprint(frame),
            'first_ns': min(times), 'last_ns': max(times), 'interval_ns': interval(frame, declared_seconds)}


def segments(indices: Iterable[int]) -> list[list[int]]:
    result: list[list[int]] = []
    for index in sorted(set(indices)):
        if result and result[-1][1] == index:
            result[-1][1] += 1
        else:
            result.append([index, index + 1])
    return result

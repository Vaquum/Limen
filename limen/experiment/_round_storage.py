from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import tempfile
import zlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import BinaryIO, cast

import limen.experiment._round_signals as _signals

__all__: list[str] = []

class RoundStorageError(ValueError):
    """Recorded round storage failed integrity validation."""


FORMAT, LEGACY_FORMAT = 'gzip-jsonl-v2', 'gzip-jsonl-v1'
_MEMBER_LIMIT, _MARKET_LIMIT = 512 * 1024 * 1024, 128 * 1024 * 1024


def _format(directory: Path) -> object:
    plain, compact = (directory / name for name in ('round_data.jsonl', 'round_data.jsonl.gz'))
    metadata = directory / 'metadata.json'
    value: object = _json(metadata.read_bytes(), metadata) if metadata.exists() else {}
    if not isinstance(value, dict):
        raise RoundStorageError(f'Invalid storage metadata in {metadata}')
    settings = cast(dict[str, object], value)
    storage = settings.get('round_storage')
    if storage not in (None, LEGACY_FORMAT, FORMAT) or (plain.exists() and compact.exists()):
        raise RoundStorageError(f'Conflicting or unknown round storage in {directory}')
    if storage is None and compact.exists():
        raise RoundStorageError(f'Compressed round data lacks format metadata in {directory}')
    if storage is not None and plain.exists():
        raise RoundStorageError(f'Compact metadata conflicts with legacy round data in {directory}')
    if (directory / 'round_market').exists() and storage is None:
        raise RoundStorageError(f'Market directory lacks compressed format metadata in {directory}')
    if (directory / 'round_signals').exists() and storage != FORMAT:
        raise RoundStorageError(f'Signal directory lacks v2 format metadata in {directory}')
    return storage


def round_path(directory: Path, *, fresh: bool = False) -> Path:
    compact = _format(directory) is not None or (fresh and not (directory / 'metadata.json').exists() and not (directory / 'round_data.jsonl').exists())
    return directory / ('round_data.jsonl.gz' if compact else 'round_data.jsonl')


def _members(stream: BinaryIO, path: Path, *, tail: bool = False,
             limit: int = _MEMBER_LIMIT) -> Iterator[tuple[bytes, int]]:
    while header := stream.read(2):
        start = stream.tell() - len(header)
        if header == b'\x1f' and tail:
            return
        if header != b'\x1f\x8b':
            raise RoundStorageError(f'Invalid gzip member header in {path} at byte {start}')
        decoder, data, chunk = zlib.decompressobj(wbits=31), bytearray(), header
        while not decoder.eof:
            if not chunk:
                if tail:
                    return
                raise RoundStorageError(f'Incomplete gzip member in {path} at byte {start}')
            try:
                data.extend(decoder.decompress(chunk, limit - len(data) + 1))
            except zlib.error as exc:
                raise RoundStorageError(f'Corrupt gzip member in {path} at byte {start}: {exc}') from exc
            if len(data) > limit:
                raise RoundStorageError(f'Decompressed size limit exceeded in {path} at byte {start}')
            if decoder.eof:
                _ = stream.seek(-len(decoder.unused_data), 1)
            else:
                chunk = stream.read(65536)
        yield bytes(data), stream.tell()


def _market(value: object, path: Path, count: int | None) -> dict[str, object]:
    if not isinstance(value, dict) or set(cast(dict[str, object], value)) != {'ret'}:
        raise RoundStorageError(f'Invalid market payload in {path}')
    values = cast(dict[str, object], value)['ret']
    if not isinstance(values, list) or (count is not None and len(cast(list[object], values)) != count):
        raise RoundStorageError(f'Market length differs from predictions in {path}')
    if any(v is not None and (type(v) not in (int, float) or not math.isfinite(cast(float, v))) for v in cast(list[object], values)):
        raise RoundStorageError(f'Invalid market return in {path}')
    return cast(dict[str, object], value)


def _row_count(record: Mapping[str, object]) -> int:
    values = record.get('preds')
    if not isinstance(values, list):
        execution = record.get('execution')
        values = cast(Mapping[str, object], execution).get('pos') if isinstance(execution, Mapping) else None
    if not isinstance(values, list):
        raise RoundStorageError('Market recording requires predictions or execution positions for row alignment')
    return len(cast(list[object], values))


def _restore(record: dict[str, object], directory: Path, storage: object) -> dict[str, object]:
    signals = tuple(field for field in ('preds', 'probs') if field + '_ref' in record)
    for field in ('preds', 'probs', 'market'):
        key = field + '_ref'
        if key not in record:
            continue
        signal = field != 'market'
        reference = _signals.reference(record[key], signal=signal)
        if field in record or storage is None or (signal and storage != FORMAT) or reference is None:
            raise RoundStorageError(f'Invalid {field} reference for round {record.get("round_id")} in {directory}')
        digest, length = reference
        path = directory / ('round_signals' if signal else 'round_market') / (digest + '.json.gz')
        if not path.is_file():
            raise FileNotFoundError(f'Missing {field} for round {record.get("round_id")}: {path}')
        with path.open('rb') as stream:
            iterator = _members(stream, path, limit=_MARKET_LIMIT)
            member = next(iterator, None)
            if member is None or next(iterator, None) is not None:
                raise RoundStorageError(f'Expected one {field} member in {path}')
        if hashlib.sha256(member[0]).hexdigest() != digest:
            raise RoundStorageError(f'{field} digest mismatch in {path} for round {record.get("round_id")}')
        value = _json(member[0], path)
        if signal and _signals.count(value) != length:
            raise RoundStorageError(f'Invalid signal payload or count in {path}')
        value = value if signal else _market(value, path, _row_count(record))
        record = {(field if name == key else name): (value if name == key else item) for name, item in record.items()}
    if signals and not _signals.aligned(record, ('preds', 'probs'), eligible=True):
        raise RoundStorageError(f'Signal lengths differ from predictions or execution alignment in {directory}')
    return record


def append_record(path: Path, record: Mapping[str, object]) -> None:
    storage = _format(path.parent)
    if path.name in ('round_data.jsonl', 'round_data.jsonl.gz') and round_path(path.parent) != path:
        raise RoundStorageError(f'Round data path conflicts with metadata: {path}')
    if any(field + '_ref' in record for field in ('market', 'preds', 'probs')):
        raise RoundStorageError(f'Writer requires resolved values in {path}')
    if path.suffix != '.gz':
        with path.open('ab+') as stream:
            separator = b''
            if stream.tell():
                _ = stream.seek(-1, 2)
                if stream.read(1) != b'\n':
                    separator = b'\n'
            _ = stream.write(separator + (json.dumps(record) + '\n').encode('utf-8'))
        return
    wire = dict(record)
    if storage == FORMAT and not _signals.aligned(record, ('preds', 'probs'), eligible=True):
        raise RoundStorageError(f'Signal lengths differ from predictions or execution alignment in {path}')
    blobs: dict[Path, bytes] = {}
    for field in ('market', 'preds', 'probs'):
        value, signal = record.get(field), field != 'market'
        count = _signals.count(value) if signal else None
        if value is None or (signal and (storage != FORMAT or count is None)):
            continue
        if not signal:
            _ = _market(value, path, _row_count(record))
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('utf-8')
        if len(raw) > _MARKET_LIMIT:
            raise RoundStorageError(f'{field} size limit exceeded in {path}')
        digest = hashlib.sha256(raw).hexdigest()
        reference: object = {'sha256': digest, 'count': count} if signal else digest
        wire = {(field + '_ref' if key == field else key): (reference if key == field else item) for key, item in wire.items()}
        blobs[path.parent / ('round_signals' if signal else 'round_market') / (digest + '.json.gz')] = raw
    encoded = (json.dumps(wire) + '\n').encode('utf-8')
    if len(encoded) > _MEMBER_LIMIT:
        raise RoundStorageError(f'Round size limit exceeded in {path}')
    for blob, raw in blobs.items():
        folder = blob.parent
        folder.mkdir(exist_ok=True)
        if not blob.exists():
            with tempfile.NamedTemporaryFile(dir=folder, prefix='payload-', suffix='.tmp', delete=False) as temporary:
                _ = temporary.write(gzip.compress(raw, compresslevel=1, mtime=0))
                temporary_path = Path(temporary.name)
            _ = temporary_path.replace(blob)
    with path.open('ab') as stream:
        _ = stream.write(gzip.compress(encoded, compresslevel=1, mtime=0))


def _frames(stream: BinaryIO, path: Path, *, tail: bool = False) -> Iterator[tuple[bytes, int]]:
    if path.suffix == '.gz':
        yield from _members(stream, path, tail=tail)
    else:
        while line := stream.readline(_MEMBER_LIMIT + 1):
            if len(line) > _MEMBER_LIMIT:
                raise RoundStorageError(f'Round size limit exceeded in {path}')
            if line.strip():
                yield line, stream.tell()


@contextmanager
def open_text(path: Path, *, limit: int | None = None, tail: bool = False) -> Iterator[Iterator[str]]:
    storage = _format(path.parent)
    if path.name in ('round_data.jsonl', 'round_data.jsonl.gz') and round_path(path.parent) != path:
        raise RoundStorageError(f'Round data path conflicts with metadata: {path}')
    with path.open('rb') as stream:
        def lines() -> Iterator[str]:
            previous = -1
            for raw, end in islice(_frames(stream, path, tail=tail), limit):
                decoded = _json(raw, path)
                if path.suffix != '.gz':
                    if isinstance(decoded, dict):
                        _ = _restore(cast(dict[str, object], decoded), path.parent, None)
                    yield raw.decode('utf-8')
                else:
                    if not isinstance(decoded, dict):
                        raise RoundStorageError(f'Round must be an object in {path} ending at byte {end}')
                    record = cast(dict[str, object], decoded)
                    index = record.get('_round_index')
                    if type(index) is not int or index <= previous:
                        raise RoundStorageError(f'Invalid round order in {path} ending at byte {end}')
                    previous = index
                    yield json.dumps(_restore(record, path.parent, storage)) + '\n'
        yield lines()


def csv_prefix(path: Path, count: int) -> list[tuple[int, str]]:
    result: list[tuple[int, str]] = []
    seen = 0
    with (path.parent / 'results.csv').open(newline='') as stream:
        for index, row in enumerate(islice(csv.DictReader(stream), count)):
            seen += 1
            if not row.get('strict_mode_error'):
                result.append((index, row['id']))
    if seen != count:
        raise RoundStorageError(f'Incomplete successful CSV prefix for {path}')
    return result


def validate_prefix(path: Path, expected: Sequence[tuple[int, str]]) -> None:
    if not path.exists():
        if not expected:
            return
        raise RoundStorageError(f'round_data.jsonl not found for committed prefix: {path}')
    with open_text(path, limit=len(expected)) as stream:
        for occurrence in expected:
            line = next(stream, None)
            record = cast(dict[str, object], json.loads(line)) if line is not None else {}
            if (record.get('_round_index'), record.get('round_id')) != occurrence:
                raise RoundStorageError(f'Round evidence differs from successful terminal CSV occurrences in {path}')


def truncate_rounds(path: Path, start_round: int) -> None:
    storage = _format(path.parent)
    end = 0
    with path.open('rb') as stream:
        count = len(csv_prefix(path, start_round)) if (path.parent / 'results.csv').exists() else None
        limit = count if path.suffix != '.gz' else None
        for position, (raw, offset) in enumerate(islice(_frames(stream, path, tail=True), limit)):
            record = cast(dict[str, object], json.loads(raw))
            committed = position < count if count is not None else cast(int, record.get('_round_index', 0)) < start_round
            if committed:
                end = offset
            elif path.suffix != '.gz':
                break
            _ = _restore(record, path.parent, storage)
    with path.open('r+b') as stream:
        _ = stream.truncate(end)


def text(path: Path) -> str:
    with open_text(path) as lines:
        return "".join(lines)


def propagate(error: BaseException) -> None:
    if isinstance(error, RoundStorageError):
        raise error


def _json(raw: str | bytes, path: Path) -> object:
    try:
        return cast(object, json.loads(raw))
    except (ValueError, UnicodeDecodeError) as exc:
        raise RoundStorageError(f'Malformed JSON in {path}') from exc

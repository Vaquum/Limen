from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
import tempfile
import zlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import BinaryIO, cast

__all__: list[str] = []

class RoundStorageError(ValueError):
    """Recorded round storage failed integrity validation."""


FORMAT = 'gzip-jsonl-v1'
_MEMBER_LIMIT = 512 * 1024 * 1024
_MARKET_LIMIT = 128 * 1024 * 1024


def round_path(directory: Path, *, fresh: bool = False) -> Path:
    plain, compact = (directory / name for name in ('round_data.jsonl', 'round_data.jsonl.gz'))
    metadata = directory / 'metadata.json'
    value: object = _json(metadata.read_text(), metadata) if metadata.exists() else {}
    if not isinstance(value, dict):
        raise RoundStorageError(f'Invalid storage metadata in {metadata}')
    settings = cast(dict[str, object], value)
    storage = settings.get('round_storage')
    if storage not in (None, FORMAT) or (plain.exists() and compact.exists()):
        raise RoundStorageError(f'Conflicting or unknown round storage in {directory}')
    if storage is None and compact.exists():
        raise RoundStorageError(f'Compressed round data lacks format metadata in {directory}')
    if storage == FORMAT and plain.exists():
        raise RoundStorageError(f'Compact metadata conflicts with legacy round data in {directory}')
    if (directory / 'round_market').exists() and not compact.exists():
        raise RoundStorageError(f'Market directory lacks compressed round data in {directory}')
    return compact if storage == FORMAT or (fresh and not metadata.exists() and not plain.exists()) else plain


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


def _restore(record: dict[str, object], directory: Path) -> dict[str, object]:
    if 'market_ref' not in record:
        return record
    digest = record['market_ref']
    if 'market' in record or not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None:
        raise RoundStorageError(f'Invalid market reference for round {record.get("round_id")} in {directory}')
    path = directory / 'round_market' / (digest + '.json.gz')
    with path.open('rb') as stream:
        iterator = _members(stream, path, limit=_MARKET_LIMIT)
        member = next(iterator, None)
        if member is None or next(iterator, None) is not None:
            raise RoundStorageError(f'Expected one market member in {path}')
    if hashlib.sha256(member[0]).hexdigest() != digest:
        raise RoundStorageError(f'Market digest mismatch in {path} for round {record.get("round_id")}')
    predictions = record.get('preds')
    count = len(cast(list[object], predictions)) if isinstance(predictions, list) else None
    market = _market(_json(member[0], path), path, count)
    return {('market' if key == 'market_ref' else key): (market if key == 'market_ref' else value)
            for key, value in record.items()}


def append_record(path: Path, record: Mapping[str, object]) -> None:
    if path.suffix != '.gz':
        with path.open('a') as stream:
            _ = stream.write(json.dumps(record) + '\n')
        return
    wire = dict(record)
    raw: bytes | None = None
    if (market := wire.get('market')) is not None:
        _ = _market(market, path, len(cast(list[object], wire['preds'])))
        raw = json.dumps(market, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('utf-8')
        if len(raw) > _MARKET_LIMIT:
            raise RoundStorageError(f'Market size limit exceeded in {path}')
        digest = hashlib.sha256(raw).hexdigest()
        wire = {('market_ref' if key == 'market' else key): (digest if key == 'market' else value)
                for key, value in wire.items()}
    encoded = (json.dumps(wire) + '\n').encode('utf-8')
    if len(encoded) > _MEMBER_LIMIT:
        raise RoundStorageError(f'Round size limit exceeded in {path}')
    if raw is not None:
        folder = path.parent / 'round_market'
        folder.mkdir(exist_ok=True)
        blob = folder / (hashlib.sha256(raw).hexdigest() + '.json.gz')
        if not blob.exists():
            with tempfile.NamedTemporaryFile(dir=folder, prefix='market-', suffix='.tmp', delete=False) as temporary:
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
    if path.name in ('round_data.jsonl', 'round_data.jsonl.gz') and round_path(path.parent) != path:
        raise RoundStorageError(f'Round data path conflicts with metadata: {path}')
    with path.open('rb') as stream:
        def lines() -> Iterator[str]:
            previous = -1
            for raw, end in islice(_frames(stream, path, tail=tail), limit):
                if path.suffix != '.gz':
                    try:
                        _ = json.loads(raw)
                    except (ValueError, UnicodeDecodeError) as exc:
                        raise RoundStorageError(f'Malformed JSON in {path} ending at byte {end}') from exc
                    yield raw.decode('utf-8')
                else:
                    try:
                        decoded = cast(object, json.loads(raw))
                    except (ValueError, UnicodeDecodeError) as exc:
                        raise RoundStorageError(f'Invalid round JSON in {path} ending at byte {end}') from exc
                    if not isinstance(decoded, dict):
                        raise RoundStorageError(f'Round must be an object in {path} ending at byte {end}')
                    record = cast(dict[str, object], decoded)
                    index = record.get('_round_index')
                    if type(index) is not int or index <= previous:
                        raise RoundStorageError(f'Invalid round order in {path} ending at byte {end}')
                    previous = index
                    yield json.dumps(_restore(record, path.parent)) + '\n'
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
    end = 0
    with path.open('rb') as stream:
        count = len(csv_prefix(path, start_round)) if (path.parent / 'results.csv').exists() else None
        for raw, offset in islice(_frames(stream, path, tail=True), count):
            record = cast(dict[str, object], json.loads(raw))
            if cast(int, record.get('_round_index', 0)) >= start_round:
                break
            _ = _restore(record, path.parent)
            end = offset
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

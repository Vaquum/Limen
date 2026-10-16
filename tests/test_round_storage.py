import gzip
import hashlib
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from limen.cohort import Cohort
from limen.experiment import _round_storage as storage
from limen.experiment.acceptance_report import acceptance_report
from limen.inference import Trainer
from tests.test_record_execution import _assert_aligned_market, _config, _loop, _run, _stop_after_first
from tests import test_walk_forward_uel as forward

logger = logging.getLogger(__name__)


@pytest.fixture(scope='module')
def recorded_rounds(tmp_path_factory):
    root = Path(__file__).parent / 'fixtures'
    bars = pl.read_parquet(root / 'spot_1h_20240101_20241231.parquet').head(1200)
    captured = {}
    for kind in ('rule', 'binary', 'ridge', 'folds'):
        directory = tmp_path_factory.mktemp('storage-' + kind)
        loop = forward._loop(forward._config(), forward._bars(), directory) if kind == 'folds' else _loop(_config(kind), bars, directory)
        if kind == 'folds':
            forward._run(loop)
        else:
            _run(loop, record_execution=True, record_model_outputs=True)
        with storage.open_text(storage.round_path(directory)) as stream:
            captured[kind] = {'records': [json.loads(line) for line in stream], 'directory': directory, 'loop': loop}
    return captured


def _write(directory, records, *, compact=True):
    directory.mkdir()
    if compact:
        (directory / 'metadata.json').write_text(json.dumps({'round_storage': storage.FORMAT}))
    path = storage.round_path(directory)
    for record in records:
        storage.append_record(path, record)
    return path


def _writer_record(loop, path, record):
    if 'folds' in record:
        alignment = {'folds': record['folds']}
    else:
        alignment = dict(record['alignment'])
        alignment['missing_datetimes'] = [datetime.fromisoformat(value) for value in alignment['missing_datetimes']]
        for field in ('first_test_datetime', 'last_test_datetime'):
            alignment[field] = datetime.fromisoformat(alignment[field])
        for field in ('execution', 'market', 'trade_contract', 'trade_contract_digest', 'trade_ledger', 'learning_binding'):
            if field in record:
                alignment[field] = record[field]
        alignment['model_outputs'] = {key: record[key] for key in ('probs', 'optimal_threshold', 'threshold_rule') if key in record}
    loop._append_round_data(path, record['round_id'], record['round_params'], record.get('preds', []), alignment, round_index=record['_round_index'])


def test_compressed_equals_uncompressed(tmp_path, recorded_rounds):
    for kind, captured in recorded_rounds.items():
        legacy, compact = tmp_path / (kind + '-legacy'), tmp_path / (kind + '-compact')
        original = _write(legacy, [], compact=False)
        compressed = _write(compact, [])
        for record in captured['records']:
            _writer_record(captured['loop'], original, record)
            _writer_record(captured['loop'], compressed, record)
        expected = ''.join(json.dumps(record) + '\n' for record in captured['records'])
        assert original.read_text() == storage.text(compressed) == expected
        assert list(json.loads(expected.splitlines()[0])) == list(json.loads(storage.text(compressed).splitlines()[0]))


def test_new_runs_write_compressed_by_default(recorded_rounds):
    for kind, captured in recorded_rounds.items():
        directory = captured['directory']
        assert storage.round_path(directory).name == 'round_data.jsonl.gz'
        assert not (directory / 'round_data.jsonl').exists()
        assert json.loads((directory / 'metadata.json').read_text())['round_storage'] == storage.FORMAT
        if kind == 'folds':
            assert not (directory / 'round_market').exists()


def test_market_deduplication_respects_alignment_and_values(tmp_path, recorded_rounds):
    records = recorded_rounds['rule']['records']
    path = _write(tmp_path / 'dedup', records[:1])
    blobs = list((path.parent / 'round_market').iterdir())
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in blobs}
    real_open, real_replace = Path.open, Path.replace

    def observe_open(p, mode='r', *args, **kwargs):
        assert p not in before or not any(flag in mode for flag in 'wax+')
        return real_open(p, mode, *args, **kwargs)

    def observe_replace(p, destination):
        assert Path(destination) not in before
        return real_replace(p, destination)

    with patch.object(Path, 'open', observe_open), patch.object(Path, 'replace', observe_replace):
        storage.append_record(path, records[1])
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in blobs} == before
    expected = {hashlib.sha256(json.dumps(record['market'], sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest() for record in records}
    assert {p.name.removesuffix('.json.gz') for p in (path.parent / 'round_market').iterdir()} == expected
    assert storage.text(path) == ''.join(json.dumps(record) + '\n' for record in records)


def test_legacy_readers_and_resume(tmp_path, recorded_rounds):
    captured = recorded_rounds['rule']
    directory = tmp_path / 'legacy'
    shutil.copytree(captured['directory'], directory)
    storage.round_path(directory).unlink()
    shutil.rmtree(directory / 'round_market')
    metadata = json.loads((directory / 'metadata.json').read_text())
    metadata.pop('round_storage')
    (directory / 'metadata.json').write_text(json.dumps(metadata))
    path = storage.round_path(directory)
    for record in captured['records']:
        storage.append_record(path, record)
    before, inode = path.read_bytes(), path.stat().st_ino
    with path.open('ab') as stream:
        stream.write((json.dumps(captured['records'][-1]) + '\n').encode()[:100])
    real_open, real_replace = Path.open, Path.replace

    def observe_open(p, mode='r', *args, **kwargs):
        assert p != path or not any(flag in mode for flag in 'wa')
        return real_open(p, mode, *args, **kwargs)

    def observe_replace(p, destination):
        assert Path(destination) != path
        return real_replace(p, destination)

    resumed = _loop(_config('rule'), captured['loop'].data, directory)
    with patch.object(Path, 'open', observe_open), patch.object(Path, 'replace', observe_replace):
        _run(resumed, record_execution=True, record_model_outputs=True, resume=True)
    assert path.read_bytes() == before and path.stat().st_ino == inode
    assert not (directory / 'round_data.jsonl.gz').exists()


@pytest.mark.parametrize('cut', ('header', 'body', 'trailer'))
def test_compact_live_read_and_resume(tmp_path, recorded_rounds, cut):
    records = recorded_rounds['rule']['records']
    path = _write(tmp_path / cut, records[:1])
    prefix, inode = path.read_bytes(), path.stat().st_ino
    member = gzip.compress((json.dumps(records[1]) + '\n').encode(), compresslevel=1, mtime=0)
    length = {'header': 1, 'body': len(member) // 2, 'trailer': len(member) - 4}[cut]
    with path.open('ab') as stream:
        stream.write(member[:length])
    with pytest.raises(storage.RoundStorageError):
        storage.text(path)
    with storage.open_text(path, tail=True) as lines:
        assert list(lines) == [json.dumps(records[0]) + '\n']
    expected = [(records[0]['_round_index'], records[0]['round_id'])]
    storage.validate_prefix(path, expected)
    real_open = Path.open

    def observe_open(p, mode='r', *args, **kwargs):
        assert p != path or not any(flag in mode for flag in 'wa')
        return real_open(p, mode, *args, **kwargs)

    with patch.object(Path, 'open', observe_open):
        storage.truncate_rounds(path, records[1]['_round_index'])
    assert path.read_bytes() == prefix and path.stat().st_ino == inode


@pytest.mark.parametrize('damage', ('zero', 'crc', 'missing_blob', 'blob_digest', 'reference', 'both', 'unknown', 'layout', 'duplicate'))
def test_committed_corruption_fails(tmp_path, recorded_rounds, damage):
    records = recorded_rounds['rule']['records']
    directory = tmp_path / damage
    shutil.copytree(recorded_rounds['rule']['directory'], directory)
    path = storage.round_path(directory)
    if damage in ('zero', 'crc'):
        first = gzip.compress((json.dumps(records[0]) + '\n').encode(), compresslevel=1, mtime=0)
        raw = bytearray(first)
        if damage == 'zero':
            raw[:] = b'\0' * len(raw)
        else:
            raw[-8] ^= 1
        path.write_bytes(raw + path.read_bytes())
    elif damage in ('missing_blob', 'blob_digest'):
        blob = next((path.parent / 'round_market').iterdir())
        if damage == 'missing_blob':
            blob.unlink()
        else:
            other = recorded_rounds['folds']['records'][0]['folds'][0]['market']
            blob.write_bytes(gzip.compress(json.dumps(other).encode(), compresslevel=1, mtime=0))
    elif damage in ('reference', 'both', 'duplicate'):
        record = dict(records[0])
        if damage == 'reference':
            record.pop('market')
            record['market_ref'] = '../../outside'
        elif damage == 'both':
            record['market_ref'] = '0' * 64
        path.write_bytes(gzip.compress((json.dumps(record) + '\n').encode(), compresslevel=1, mtime=0) * (2 if damage == 'duplicate' else 1))
    elif damage == 'unknown':
        (path.parent / 'metadata.json').write_text(json.dumps({'round_storage': 'future'}))
    else:
        (path.parent / 'round_data.jsonl').write_text(json.dumps(records[0]) + '\n')
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in path.parent.rglob('*') if p.is_file()}
    loop = _loop(_config('rule'), recorded_rounds['rule']['loop'].data, directory)
    with pytest.raises((storage.RoundStorageError, FileNotFoundError)):
        _run(loop, record_execution=True, record_model_outputs=True, resume=True)
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before} == before


def test_consumers_match_legacy(tmp_path, recorded_rounds):
    for kind in ('rule', 'binary', 'ridge'):
        captured = recorded_rounds[kind]
        legacy_path = _write(tmp_path / kind, captured['records'], compact=False)
        compact_path = storage.round_path(captured['directory'])
        trainer = object.__new__(Trainer)
        trainer._experiment_dir = legacy_path.parent
        legacy = trainer._load_round_data()
        trainer._experiment_dir = compact_path.parent
        assert json.dumps(trainer._load_round_data()) == json.dumps(legacy)
        assert json.dumps(Cohort._load_round_entries(compact_path)) == json.dumps(Cohort._load_round_entries(legacy_path))
        for record in json.loads('[' + ','.join(storage.text(compact_path).splitlines()) + ']'):
            _assert_aligned_market(record, captured['loop'].data)
    # Walk-forward fixture construction already ran acceptance on compact records.
    assert (recorded_rounds['folds']['directory'] / 'acceptance_report.json').exists()
    rejected = tmp_path / 'corrupt-acceptance'
    shutil.copytree(recorded_rounds['folds']['directory'], rejected)
    path = storage.round_path(rejected)
    path.write_bytes(path.read_bytes()[:-4])
    reports = {p: p.read_bytes() for p in rejected.glob('acceptance_report.*')}
    with pytest.raises(storage.RoundStorageError):
        acceptance_report(rejected)
    assert {p: p.read_bytes() for p in reports} == reports


def test_recorded_workload_storage_reduction(tmp_path, recorded_rounds):
    total_legacy = total_compact = 0
    started, cpu = time.perf_counter(), time.process_time()
    for kind, captured in recorded_rounds.items():
        if kind == 'folds':
            continue
        legacy = _write(tmp_path / (kind + '-legacy'), captured['records'], compact=False)
        compact = _write(tmp_path / (kind + '-compact'), captured['records'])
        assert legacy.read_text() == storage.text(compact)
        raw = legacy.stat().st_size
        stored = sum(p.stat().st_size for p in compact.parent.rglob('*') if p.is_file())
        total_legacy += raw
        total_compact += stored
        logger.info('%s: rounds=%d legacy=%d compact_all_files=%d ratio=%.4f', kind, len(captured['records']), raw, stored, stored / raw)
    assert total_compact < total_legacy
    logger.info('wall_seconds=%.6f cpu_seconds=%.6f total_ratio=%.4f', time.perf_counter() - started, time.process_time() - cpu, total_compact / total_legacy)


@pytest.mark.parametrize('limit', ('_MARKET_LIMIT', '_MEMBER_LIMIT'))
def test_writer_rejects_unreadable_sizes_before_writing(tmp_path, recorded_rounds, monkeypatch, limit):
    directory = tmp_path / limit
    path = _write(directory, [])
    before = {p: p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    monkeypatch.setattr(storage, limit, 1)
    with pytest.raises(storage.RoundStorageError, match='size limit'):
        storage.append_record(path, recorded_rounds['rule']['records'][0])
    assert {p: p.read_bytes() for p in directory.rglob('*') if p.is_file()} == before


def test_orphaned_first_market_blob_allows_empty_prefix(tmp_path, recorded_rounds):
    path = _write(tmp_path / 'orphan', recorded_rounds['rule']['records'][:1])
    path.unlink()
    assert (path.parent / 'round_market').is_dir()
    assert storage.round_path(path.parent) == path
    storage.validate_prefix(path, [])
    record = recorded_rounds['rule']['records'][0]
    with pytest.raises(storage.RoundStorageError, match='not found'):
        storage.validate_prefix(path, [(record['_round_index'], record['round_id'])])


def test_resume_rejects_crc_damage_beyond_checkpoint(tmp_path, recorded_rounds):
    records = recorded_rounds['rule']['records']
    path = _write(tmp_path / 'bad-uncommitted-crc', records[:1])
    member = bytearray(gzip.compress((json.dumps(records[1]) + '\n').encode(), mtime=0))
    member[-8] ^= 1
    with path.open('ab') as stream:
        stream.write(member)
    before = path.read_bytes()
    storage.validate_prefix(path, [(records[0]['_round_index'], records[0]['round_id'])])
    with pytest.raises(storage.RoundStorageError, match='Corrupt gzip'):
        storage.truncate_rounds(path, records[1]['_round_index'])
    assert path.read_bytes() == before


def test_legacy_resume_appends_after_record_without_newline(tmp_path, recorded_rounds):
    bars = recorded_rounds['rule']['loop'].data
    first = _loop(_config('rule'), bars, tmp_path)
    _stop_after_first(first)
    _run(first, record_execution=True, record_model_outputs=True)
    path = storage.round_path(tmp_path)
    record = storage.text(path).rstrip('\n')
    path.unlink()
    shutil.rmtree(tmp_path / 'round_market')
    metadata = json.loads((tmp_path / 'metadata.json').read_text())
    metadata.pop('round_storage')
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    path = storage.round_path(tmp_path)
    path.write_text(record)
    before, inode = path.read_bytes(), path.stat().st_ino
    resumed = _loop(_config('rule'), bars, tmp_path)
    _run(resumed, record_execution=True, record_model_outputs=True, resume=True)
    assert path.stat().st_ino == inode and path.read_bytes().startswith(before + b'\n')
    assert storage.text(path) == ''.join(json.dumps(record) + '\n' for record in recorded_rounds['rule']['records'])

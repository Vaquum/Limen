import copy
import gzip
import hashlib
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from limen.cohort import Cohort
from limen.experiment import _round_storage as storage
from limen.inference import Trainer
from tests import test_factorized_preparation as factorized
from tests.test_factorize_equivalence import _numeric_equal
from tests.test_record_execution import _config, _loop, _run
from tests.test_round_storage import recorded_rounds as recorded_rounds

logger = logging.getLogger(__name__)


def _records(directory):
    return [json.loads(line) for line in storage.text(storage.round_path(directory)).splitlines()]


def _raw_records(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def _raw_write(path, records):
    path.write_bytes(b''.join(gzip.compress((json.dumps(record) + '\n').encode(),
                                          compresslevel=1, mtime=0) for record in records))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode()


def _write(directory, records):
    directory.mkdir()
    (directory / 'metadata.json').write_text(json.dumps({'round_storage': storage.FORMAT}))
    path = storage.round_path(directory)
    for record in records:
        storage.append_record(path, record)
    return path


def _v1(directory, records, *, plain=False):
    """Freeze the previous format independently of the new writer."""
    directory.mkdir(exist_ok=True)
    metadata = directory / 'metadata.json'
    values = json.loads(metadata.read_text()) if metadata.exists() else {}
    if plain:
        values.pop('round_storage', None)
    else:
        values['round_storage'] = 'gzip-jsonl-v1'
    metadata.write_text(json.dumps(values))
    path = directory / ('round_data.jsonl' if plain else 'round_data.jsonl.gz')
    members = []
    for record in records:
        wire = dict(record)
        if not plain and record.get('market') is not None:
            raw = _canonical(record['market'])
            digest = hashlib.sha256(raw).hexdigest()
            folder = directory / 'round_market'
            folder.mkdir(exist_ok=True)
            (folder / (digest + '.json.gz')).write_bytes(gzip.compress(raw, compresslevel=1, mtime=0))
            wire = {('market_ref' if key == 'market' else key):
                    (digest if key == 'market' else value) for key, value in wire.items()}
        raw = (json.dumps(wire) + '\n').encode()
        members.append(raw if plain else gzip.compress(raw, compresslevel=1, mtime=0))
    path.write_bytes(b''.join(members))
    return path


def _sizes(directory):
    files = [path for path in directory.rglob('*') if path.is_file()]
    return (sum(path.stat().st_size for path in files),
            sum(path.stat().st_blocks * 512 for path in files), len(files))


def _snapshot(directory):
    return {path.relative_to(directory): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob('*') if path.is_file()}


@pytest.fixture(scope='module')
def economic_sweep(tmp_path_factory):
    config = factorized._config()
    config['sfd']['manifest']['split_dates'].update(
        train_start='2024-01-01', train_end='2024-01-20',
        val_start='2024-01-20', val_end='2024-01-25',
        test_start='2024-01-25', test_end='2024-02-15')
    config['sfd']['params'].update(lookback=[2, 4], fee=[4., 9., 14., 19.], n_estimators=[16, 32])
    config['uel'].update(factorize=True, n_permutations=16)
    bars = pl.read_parquet(factorized.FIXTURE, n_rows=1200).with_columns(pl.lit(3600).alias('base_interval'))
    directory = tmp_path_factory.mktemp('signal-economic')
    started = time.perf_counter()
    loop = factorized._loop(config, directory, bars=bars)
    factorized._run(loop, enabled=True, count=16)
    return {'config': config, 'bars': bars, 'directory': directory,
            'records': _records(directory), 'seconds': time.perf_counter() - started}


def test_recorded_storage_savings(tmp_path, economic_sweep):
    records = economic_sweep['records']
    distinct = list({(_canonical(record['preds']), _canonical(record['probs'])):
                     record for record in records}.values())
    assert len(records) == 16 and len(distinct) == 4
    assert {len(record['preds']) for record in records} == {503}
    for label, selected in (('single', records[:1]), ('distinct', distinct), ('repeated', records)):
        before = time.perf_counter()
        old = _v1(tmp_path / (label + '-v1'), selected)
        old_seconds = time.perf_counter() - before
        before = time.perf_counter()
        new = _write(tmp_path / (label + '-v2'), selected)
        new_seconds = time.perf_counter() - before
        assert storage.text(new) == storage.text(old)
        baseline, compact = _sizes(old.parent), _sizes(new.parent)
        logger.info('signals %s: rounds=%d rows=%d v1_bytes=%d v2_all_bytes=%d '
                    'delta_bytes=%d v1_allocated=%d v2_allocated=%d v1_files=%d v2_files=%d '
                    'v1_write_seconds=%.6f v2_write_seconds=%.6f run_seconds=%.6f',
                    label, len(selected), len(selected[0]['preds']), baseline[0], compact[0],
                    compact[0] - baseline[0], baseline[1], compact[1], baseline[2], compact[2],
                    old_seconds, new_seconds, economic_sweep['seconds'])
        if label == 'repeated':
            assert compact[0] < baseline[0]
            assert len(list((new.parent / 'round_signals').iterdir())) == 8


def test_logical_evidence(tmp_path, recorded_rounds, economic_sweep):
    captures = list(recorded_rounds.values()) + [economic_sweep]
    for index, captured in enumerate(captures):
        directory, records = captured['directory'], captured['records']
        baseline = _v1(tmp_path / str(index), records)
        path = storage.round_path(directory)
        assert json.loads((directory / 'metadata.json').read_text())['round_storage'] == 'gzip-jsonl-v2'
        assert storage.text(path) == storage.text(baseline)
        trainer = object.__new__(Trainer)
        trainer._experiment_dir = baseline.parent
        old = trainer._load_round_data()
        trainer._experiment_dir = directory
        _numeric_equal(trainer._load_round_data(), old)
        _numeric_equal(Cohort._load_round_entries(path), Cohort._load_round_entries(baseline))
        wire = _raw_records(path)
        for logical, encoded in zip(records, wire, strict=True):
            assert logical['_round_index'] == encoded['_round_index']
            assert logical['round_params'] == encoded['round_params']
            if 'folds' in logical:
                assert encoded['folds'] == logical['folds']
                assert not (directory / 'round_signals').exists()
            else:
                for field in ('preds', 'probs'):
                    if isinstance(logical.get(field), list):
                        assert field not in encoded
                        assert encoded[field + '_ref'] == {
                            'sha256': hashlib.sha256(_canonical(logical[field])).hexdigest(),
                            'count': len(logical[field]),
                        }


def test_trainer_and_cohort_use_shared_signals(economic_sweep):
    directory, records, bars = (economic_sweep[key] for key in ('directory', 'records', 'bars'))
    record = records[0]
    sensor, = Trainer(directory, data=bars).train([record['round_id']])
    observed = {value.datetime: value for value in sensor.predict_all(bars)}
    alignment = record['alignment']
    first, last = [datetime.fromisoformat(alignment[key]) for key in ('first_test_datetime', 'last_test_datetime')]
    missing = {datetime.fromisoformat(value) for value in alignment['missing_datetimes']}
    dates = [value for value in bars['datetime'] if first <= value <= last and value not in missing]
    assert len(dates) == len(record['preds'])
    assert all(observed[value].reason is None for value in dates)
    np.testing.assert_array_equal([observed[value].prediction for value in dates], record['preds'])
    np.testing.assert_allclose([observed[value].probability for value in dates], record['probs'], rtol=1e-12, atol=1e-12)
    cohort = Cohort(experiment_log_path=str(directory), permutation_ids=[record['round_id']])
    assert cohort.permutation_ids == [record['round_id']]


@pytest.mark.parametrize('plain', (False, True))
def test_compatibility(tmp_path, recorded_rounds, plain):
    captured = recorded_rounds['binary']
    records = captured['records']
    directory = tmp_path / 'legacy'
    shutil.copytree(captured['directory'], directory)
    storage.round_path(directory).unlink()
    for folder in ('round_market', 'round_signals'):
        shutil.rmtree(directory / folder)
    path = _v1(directory, records[:1], plain=plain)
    prefix = path.read_bytes()
    storage.append_record(path, records[1])
    assert path.read_bytes().startswith(prefix)
    expected = ''.join(json.dumps(record) + '\n' for record in records)
    assert storage.text(path) == expected
    assert not (directory / 'round_signals').exists()
    before = _snapshot(directory)
    resumed = _loop(_config('binary'), captured['loop'].data, directory)
    _run(resumed, resume=True, record_execution=True, record_model_outputs=True)
    assert storage.text(path) == expected
    assert path.read_bytes() == before[path.relative_to(directory)][0]
    assert not (directory / 'round_signals').exists()


def test_recovery(tmp_path, economic_sweep, monkeypatch):
    config = economic_sweep['config']
    directory = tmp_path / 'interrupted'
    first = factorized._loop(config, directory, bars=economic_sweep['bars'])
    checkpoint = first._checkpoint

    def stop(*args, **kwargs):
        checkpoint(*args, **kwargs)
        if args[4] == 1:
            first._shutdown_requested = True

    monkeypatch.setattr(first, '_checkpoint', stop)
    factorized._run(first, enabled=True, count=16)
    path = storage.round_path(directory)
    prefix = path.read_bytes()
    assert 0 < len(_records(directory)) < 16
    before = _snapshot(directory / 'round_signals')
    resumed = factorized._loop(config, directory, bars=economic_sweep['bars'])
    factorized._run(resumed, enabled=True, count=16, resume=True)
    assert path.read_bytes().startswith(prefix)
    for name, saved in before.items():
        blob = directory / 'round_signals' / name
        assert (blob.read_bytes(), blob.stat().st_mtime_ns) == saved
    _numeric_equal(_records(directory), economic_sweep['records'])
    expected = pl.read_csv(economic_sweep['directory'] / 'results.csv', infer_schema_length=None).drop('execution_time')
    actual = pl.read_csv(directory / 'results.csv', infer_schema_length=None).drop('execution_time')
    assert actual.equals(expected)
    assert actual['_round_index'].to_list() == list(range(16))


@pytest.mark.parametrize('damage', ('missing', 'gzip', 'digest', 'payload_type', 'nested', 'nonfinite',
                                  'count', 'boolean_count', 'reference_type', 'conflict', 'cross_length', 'unknown', 'legacy'))
def test_committed_reference_rejection_is_read_only(tmp_path, economic_sweep, damage):
    directory = tmp_path / damage
    shutil.copytree(economic_sweep['directory'], directory)
    path = storage.round_path(directory)
    records = _raw_records(path)
    reference = records[0]['preds_ref']
    blob = directory / 'round_signals' / (reference['sha256'] + '.json.gz')
    if damage == 'missing':
        blob.unlink()
    elif damage == 'gzip':
        blob.write_bytes(b'broken')
    elif damage == 'digest':
        blob.write_bytes(gzip.compress(gzip.decompress(blob.read_bytes()) + b' ', mtime=0))
    elif damage in ('payload_type', 'nested', 'nonfinite'):
        value = {'payload_type': {}, 'nested': [[0]], 'nonfinite': [float('nan')]}[damage]
        raw = json.dumps(value).encode()
        digest = hashlib.sha256(raw).hexdigest()
        (blob.parent / (digest + '.json.gz')).write_bytes(gzip.compress(raw, mtime=0))
        reference.update(sha256=digest, count=1)
    elif damage in ('count', 'boolean_count'):
        reference['count'] = True if damage == 'boolean_count' else reference['count'] + 1
    elif damage == 'reference_type':
        reference['sha256'] = []
    elif damage == 'conflict':
        records[0]['preds'] = economic_sweep['records'][0]['preds']
    elif damage == 'cross_length':
        raw = _canonical([])
        digest = hashlib.sha256(raw).hexdigest()
        (blob.parent / (digest + '.json.gz')).write_bytes(gzip.compress(raw, mtime=0))
        records[0]['probs_ref'] = {'sha256': digest, 'count': 0}
    else:
        metadata = json.loads((directory / 'metadata.json').read_text())
        metadata['round_storage'] = 'future' if damage == 'unknown' else 'gzip-jsonl-v1'
        (directory / 'metadata.json').write_text(json.dumps(metadata))
    _raw_write(path, records)
    before = _snapshot(directory)
    resumed = factorized._loop(economic_sweep['config'], directory, bars=economic_sweep['bars'])
    with pytest.raises((storage.RoundStorageError, FileNotFoundError)):
        factorized._run(resumed, enabled=True, count=16, resume=True)
    assert _snapshot(directory) == before


def test_exact_json_types_omissions_and_blob_lifetime(tmp_path, economic_sweep):
    record = copy.deepcopy(economic_sweep['records'][0])
    record.pop('market')
    record.pop('execution')
    values = [True, False, 1, 1.0, 0, 0.0, -0.0, None, -1.0, 2.0]
    record['preds'] = record['probs'] = values
    path = _write(tmp_path / 'typed', [record])
    assert storage.text(path) == json.dumps(record) + '\n'
    wire = _raw_records(path)[0]
    assert wire['preds_ref'] == wire['probs_ref']
    assert len(list((path.parent / 'round_signals').iterdir())) == 1
    before = _snapshot(path.parent / 'round_signals')
    storage.append_record(path, record | {'_round_index': record['_round_index'] + 1})
    assert _snapshot(path.parent / 'round_signals') == before
    for field in ('preds', 'probs'):
        for value in (None, [[1]], ['unavailable'], [float('nan')], [float('inf')]):
            probe = {key: val for key, val in record.items() if key not in ('preds', 'probs')}
            probe[field] = value
            selected = _write(tmp_path / (field + '-' + str(len(list(tmp_path.iterdir())))), [probe])
            assert storage.text(selected) == json.dumps(probe) + '\n'
            assert field in _raw_records(selected)[0] and field + '_ref' not in _raw_records(selected)[0]
    omitted = {key: value for key, value in record.items() if key not in ('preds', 'probs')}
    selected = _write(tmp_path / 'omitted', [omitted])
    assert storage.text(selected) == json.dumps(omitted) + '\n'
    assert not (selected.parent / 'round_signals').exists()


@pytest.mark.parametrize('plain', (False, True))
def test_unversioned_signal_references_are_rejected(tmp_path, economic_sweep, plain):
    record = copy.deepcopy(economic_sweep['records'][0])
    path = _v1(tmp_path / 'old-reference', [record], plain=plain)
    record['preds_ref'] = {'sha256': '0' * 64, 'count': len(record.pop('preds'))}
    raw = (json.dumps(record) + '\n').encode()
    path.write_bytes(raw if plain else gzip.compress(raw, mtime=0))
    before = _snapshot(path.parent)
    with pytest.raises(storage.RoundStorageError):
        storage.text(path)
    assert _snapshot(path.parent) == before


def test_signal_limits_and_stale_directory(tmp_path, economic_sweep, monkeypatch):
    record = {key: value for key, value in economic_sweep['records'][0].items()
              if key not in ('market', 'execution')}
    path = _write(tmp_path / 'limit', [])
    before = _snapshot(path.parent)
    monkeypatch.setattr(storage, '_MARKET_LIMIT', 1)
    with pytest.raises(storage.RoundStorageError, match='size limit'):
        storage.append_record(path, record)
    assert _snapshot(path.parent) == before
    path = _v1(tmp_path / 'stale', [])
    (path.parent / 'round_signals').mkdir()
    with pytest.raises(storage.RoundStorageError):
        storage.round_path(path.parent)

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
import yaml

from limen.experiment import _round_storage as storage
from limen.experiment._factorize import FactorizedRounds
from limen.experiment.errors import StrictModeError
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search.grid_strategy import GridStrategy
from limen.sfd.reference_architecture.lightgbm_binary import LightGBMBinary
from tests.test_factorize_equivalence import _numeric_equal
from tests.test_factorized_preparation import _config, _instrument, _loop, _run, _science

ROOT = Path(__file__).resolve().parents[1]


def _shuffled():
    cfg = _config()
    cfg['sfd']['params']['lookback'] = [2, 4]
    cfg['uel'].update(factorize=True, n_permutations=8)
    cfg['uel']['search_strategy'] = {'type': 'grid', 'shuffle': True, 'seed': 42}
    return cfg


def _snapshot(path):
    return {str(p.relative_to(path)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in path.rglob('*') if p.is_file()}


def _stop_after(monkeypatch, loop, index):
    original = loop._checkpoint

    def stop(*args, **kwargs):
        original(*args, **kwargs)
        if args[4] == index:
            loop._shutdown_requested = True

    monkeypatch.setattr(loop, '_checkpoint', stop)


def test_recorded_traversal_contract(tmp_path, monkeypatch):
    captured = []
    for enabled in (False, True):
        cfg = _shuffled()
        loop = _loop(cfg, tmp_path / str(enabled))
        with monkeypatch.context() as hooks:
            counts = _instrument(hooks)
            _run(loop, enabled=enabled, count=8)
        assert counts['model'] == (4 if enabled else 8)
        assert counts['target'] == counts['scaler'] == (2 if enabled else 8)
        rows, records = _science(tmp_path / str(enabled))
        assert [r['_round_index'] for r in rows] == list(range(8))
        assert [r['round_id'] for r in records] == [r['_id'] for r in rows]
        assert len({r['_id'] for r in rows}) == 8
        assert all(record['probs'] and record['execution']['pos'] for record in records)
        assert json.loads((tmp_path / str(enabled) / 'metadata.json').read_text())['grid_shuffle_version'] == 2
        captured.append([rows, records])
    _numeric_equal(*captured)

    split = _loop(_shuffled(), tmp_path / 'split')
    _stop_after(monkeypatch, split, 2)
    _run(split, enabled=True, count=8)
    resumed = _loop(_shuffled(), tmp_path / 'split')
    with monkeypatch.context() as hooks:
        counts = _instrument(hooks)
        _run(resumed, enabled=True, count=8, resume=True)
    assert counts['target'] > 0 and counts['model'] > 0  # cold invocation-local caches
    _numeric_equal(captured[1], _science(tmp_path / 'split'))


@pytest.mark.parametrize('change', ['seed', 'version', 'removed_version', 'metadata', 'missing_legacy_declarations', 'domain_order', 'domain_value', 'saved_domain'])
def test_resume_rejects_before_mutation(tmp_path, monkeypatch, change):
    cfg = _shuffled()
    loop = _loop(cfg, tmp_path)
    _stop_after(monkeypatch, loop, 1)
    _run(loop, enabled=True, count=8)
    checkpoint_path = tmp_path / 'checkpoint.json'
    checkpoint = json.loads(checkpoint_path.read_text())
    strategy = checkpoint['msq_state']['strategy_state']
    changed = copy.deepcopy(cfg)
    if change == 'seed':
        changed['uel']['search_strategy']['seed'] = 43
    elif change == 'version':
        strategy['permutation_version'] = 1
        checkpoint_path.write_text(json.dumps(checkpoint))
    elif change == 'removed_version':
        del strategy['permutation_version']
        checkpoint_path.write_text(json.dumps(checkpoint))
    elif change == 'metadata':
        metadata_path = tmp_path / 'metadata.json'
        metadata = json.loads(metadata_path.read_text())
        del metadata['grid_shuffle_version']
        metadata_path.write_text(json.dumps(metadata))
    elif change == 'missing_legacy_declarations':
        for key in ('permutation_version', 'shuffle', 'seed', 'domain_digest'):
            strategy.pop(key)
        checkpoint_path.write_text(json.dumps(checkpoint))
        metadata_path = tmp_path / 'metadata.json'
        metadata = json.loads(metadata_path.read_text())
        del metadata['grid_shuffle_version']
        metadata_path.write_text(json.dumps(metadata))
    elif change == 'domain_order':
        changed['sfd']['params']['lookback'].reverse()
    elif change == 'domain_value':
        changed['sfd']['params']['lookback'][1] = 5
    else:
        checkpoint['domain_state']['lookback'].reverse()
        checkpoint_path.write_text(json.dumps(checkpoint))
    before = _snapshot(tmp_path)
    resumed = _loop(changed, tmp_path)
    with pytest.raises(ValueError, match='seed|version|domain|[Cc]ontent hash'):
        _run(resumed, enabled=True, count=8, resume=True)
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize('failed', [(0,), (3,), (7,), (0, 1, 2), tuple(range(8))])
def test_terminal_failures_and_cold_resume(tmp_path, monkeypatch, failed):
    cfg = _shuffled()
    cfg['sfd']['params']['tag'] = ['001']
    ordered = list(GridStrategy(ParamDomain(cfg['sfd']['params']), shuffle=True, seed=42))
    rejected = [{k: v for k, v in ordered[index].items() if not k.startswith('_')} for index in failed]
    original = FactorizedRounds.evaluate

    def evaluate(self, params):
        if params in rejected:
            raise StrictModeError()  # empty exception must still be a terminal failure
        return original(self, params)

    monkeypatch.setattr(FactorizedRounds, 'evaluate', evaluate)
    full = _loop(cfg, tmp_path / 'full')
    _run(full, enabled=True, count=8)
    split = _loop(cfg, tmp_path / 'split')
    _stop_after(monkeypatch, split, 2)
    _run(split, enabled=True, count=8)
    resumed = _loop(cfg, tmp_path / 'split')
    _run(resumed, enabled=True, count=8, resume=True)
    for path in (tmp_path / 'full', tmp_path / 'split'):
        rows = pl.read_csv(path / 'results.csv', infer_schema_length=None)
        import csv
        with (path / 'results.csv').open(newline='') as stream:
            assert all(row['tag'] == '001' for row in csv.DictReader(stream))
        assert rows['_round_index'].to_list() == list(range(8))
        assert rows['strict_mode_error'].is_not_null().sum() == len(failed)
        assert rows['strict_mode_error'].drop_nulls().to_list() == ['StrictModeError'] * len(failed)
        round_path = storage.round_path(path)
        records = [json.loads(line) for line in storage.text(round_path).splitlines()] if round_path.exists() else []
        assert [r['_round_index'] for r in records] == [i for i in range(8) if i not in failed]
        assert [r['round_id'] for r in records] == [rows['_id'][i] for i in range(8) if i not in failed]
    off = pl.read_csv(tmp_path / 'full/results.csv', infer_schema_length=None).drop('execution_time')
    on = pl.read_csv(tmp_path / 'split/results.csv', infer_schema_length=None).drop('execution_time')
    assert off.equals(on)
    full_records = storage.round_path(tmp_path / 'full')
    split_records = storage.round_path(tmp_path / 'split')
    if full_records.exists():
        _numeric_equal(json.loads('[' + ','.join(storage.text(full_records).splitlines()) + ']'),
                       json.loads('[' + ','.join(storage.text(split_records).splitlines()) + ']'))


def test_interrupted_cache_miss_replays_same_candidate(tmp_path, monkeypatch):
    cfg = _shuffled()
    full = _loop(cfg, tmp_path / 'full')
    _run(full, enabled=True, count=8)
    split = _loop(cfg, tmp_path / 'split')
    original = FactorizedRounds.evaluate
    attempts = []
    interrupting = [False]
    original_train = LightGBMBinary.train

    def train(*args, **kwargs):
        if interrupting[0]:
            split._shutdown_requested = True
            raise KeyboardInterrupt
        return original_train(*args, **kwargs)

    def interrupt(self, params):
        attempts.append(dict(params))
        if len(attempts) == 4:
            self.cache.clear()
            interrupting[0] = True
        return original(self, params)

    with monkeypatch.context() as hooks:
        hooks.setattr(FactorizedRounds, 'evaluate', interrupt)
        hooks.setattr(LightGBMBinary, 'train', train)
        _run(split, enabled=True, count=8)
    checkpoint = json.loads((tmp_path / 'split/checkpoint.json').read_text())
    assert checkpoint['metadata']['experiment_round'] == 2
    resumed = _loop(cfg, tmp_path / 'split')
    seen = []

    def capture(self, params):
        seen.append(dict(params))
        return original(self, params)

    with monkeypatch.context() as hooks:
        hooks.setattr(FactorizedRounds, 'evaluate', capture)
        _run(resumed, enabled=True, count=8, resume=True)
    assert seen[0] == attempts[-1]
    _numeric_equal(_science(tmp_path / 'full'), _science(tmp_path / 'split'))


def test_native_typed_domain_is_preserved_at_restore(tmp_path, monkeypatch):
    cfg = _shuffled()
    cfg['sfd']['manifest']['scaler']['params'] = {'quantile_range': '{scale_range}'}
    cfg['sfd']['params']['scale_range'] = [(0.25, 0.75)]
    split = _loop(cfg, tmp_path)
    _stop_after(monkeypatch, split, 1)
    _run(split, enabled=True, count=8)
    resumed = _loop(cfg, tmp_path)
    original = resumed._restore_checkpoint_state

    def restored(*args, **kwargs):
        index = original(*args, **kwargs)
        assert isinstance(resumed._search_strategy.domain.params['scale_range'][0], tuple)
        resumed._shutdown_requested = True
        return index

    monkeypatch.setattr(resumed, '_restore_checkpoint_state', restored)
    _run(resumed, enabled=True, count=8, resume=True)
    assert resumed.experiment_log.height == 2


def test_fresh_process_cli_example_and_resume(tmp_path):
    config = yaml.safe_load((ROOT / 'docs/examples/factorized_shuffle.yaml').read_text())
    config['sfd']['manifest']['data_source']['params']['file_path_or_url'] = str(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet')
    source = tmp_path / 'manifest.yaml'
    source.write_text(yaml.safe_dump(config))
    env = {**os.environ, 'PYTHONPATH': str(ROOT), 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'POLARS_MAX_THREADS': '1'}
    stop_script = '''from limen.experiment import UniversalExperimentLoop
original = UniversalExperimentLoop._checkpoint
def checkpoint(self, *args, **kwargs):
    original(self, *args, **kwargs)
    if args[4] == 1:
        self._shutdown_requested = True
UniversalExperimentLoop._checkpoint = checkpoint
from limen.cli.main import cli
cli()
'''
    for command in ([sys.executable, '-c', stop_script, 'run', str(source), '--no-progress-bar'],
                    [sys.executable, '-m', 'limen', 'run', '--resume', str(tmp_path / 'results/dev/factorized_shuffle'), '--no-progress-bar']):
        result = subprocess.run(command, cwd=tmp_path,
                                env=env, text=True, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
    path = tmp_path / 'results/dev/factorized_shuffle'
    rows, records = _science(path)
    assert len(rows) == len(records) == 8
    assert [r['_round_index'] for r in rows] == list(range(8))
    assert json.loads((path / 'metadata.json').read_text())['grid_shuffle_version'] == 2

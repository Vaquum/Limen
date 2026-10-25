import copy
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
from uuid import uuid4

import polars as pl
import pytest
from ruamel.yaml import YAML

from limen.cli.commands.profile import run_profile
from limen.cli.commands.run import run_experiment
from limen.cli.git_utils import run_git
from limen.experiment import UniversalExperimentLoop
from limen.experiment import _holdout_history as history
from limen.experiment._holdout_source import admit_frame
from limen.experiment.holdout import release_holdout, reserve_holdout, withdraw_holdout
from limen.inference import Trainer
from limen.log.log import Log
from limen.yaml import CompiledSFD
from limen.yaml.compiler import build_search_strategy
from limen.yaml.profiler import profile
from tests.test_trainer import _E2E_LOGREG_YAML

_FIXTURE = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'
_QUARTER = Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet'


def _project(path):
    path.mkdir()
    (path / 'limen.toml').write_text('[project]\nname="holdout-test"\n')
    run_git(['init', '--quiet'], cwd=path).check_returncode()
    return path


def _configuration(path):
    config = YAML(typ='safe').load(_E2E_LOGREG_YAML)
    manifest = config['sfd']['manifest']
    manifest['data_source'] = {'method': 'limen.data.HistoricalData.get_any_file', 'params': {'file_path_or_url': str(_FIXTURE.resolve())}}
    manifest['split_dates'] = dict(zip(('train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end'),
        ('2024-01-01', '2024-01-03', '2024-01-03', '2024-01-04', '2024-01-04', '2024-01-05'), strict=True))
    config['uel']['n_permutations'] = 1
    config['sfd']['params']['C'] = [1.0]
    (path / 'research.yaml').write_text(json.dumps(config))
    return config


def _lease_worker(root, pipe):
    history._BUSY_TIMEOUT_SECONDS = 0.2
    pipe.send('ready')
    while (command := pipe.recv()) != 'stop':
        try:
            if command == 'reserve':
                result = reserve_holdout(root, '2024-01-05', '2024-01-06', interval_seconds=3600)
            else:
                binding = history.bind(root)
                with history.admission(Path(root)):
                    result = history.start_research(binding, 'replay')
                    history.finish_research(binding, result, {'worker': True})
            pipe.send(('ok', result))
        except (sqlite3.OperationalError, ValueError) as exc:
            pipe.send((type(exc).__name__, str(exc)))


def test_holdout_reservation_and_research_exclusion(tmp_path, monkeypatch):
    root = _project(tmp_path / 'project')
    config = _configuration(root)
    first = reserve_holdout(root, '2024-01-05', '2024-01-06', interval_seconds=3600)
    monkeypatch.setattr(history.time, 'time_ns', lambda: 0)
    binding = history.bind(root)
    compiled = CompiledSFD(config, project_root=root)
    loop = UniversalExperimentLoop(sfd=compiled, project_root=root)
    assert loop.data.height == 96
    assert loop.data['datetime'].dt.epoch('ns').max() < history.utc_day('2024-01-05')
    raw = pl.read_parquet(_FIXTURE).head(168)
    native = SimpleNamespace(__name__='holdout_native', params=compiled.params, manifest=compiled.manifest)
    assert UniversalExperimentLoop(sfd=native, data=raw, project_root=root).data.equals(loop.data)
    seen = []
    manifest = compiled.manifest()
    manifest.set_pre_split_data_selector(lambda frame: seen.append(frame.height) or frame.slice(1))
    prepared = manifest.prepare_data(raw, {key: values[0] for key, values in compiled.params().items()})
    assert seen == [96]
    assert prepared['holdout']['selected']['rows'] == 95
    assert prepared['holdout']['source']['admitted']['rows'] == 96
    assert prepared['holdout']['source']['kind'] == 'caller_frame'
    with pytest.raises(ValueError, match='used holdout'):
        withdraw_holdout(root, first)
    with pytest.raises(ValueError, match='intersects'):
        reserve_holdout(root, '2024-01-05', '2024-01-07', interval_seconds=3600)
    release_holdout(root, first)
    assert history.bind(root).cutoff_ns is None
    with pytest.raises(ValueError, match='revision changed'):
        history.check_binding(binding)
    with pytest.raises(ValueError, match='released'):
        reserve_holdout(root, '2024-01-05', '2024-01-06', interval_seconds=3600)
    unused = reserve_holdout(root, '2024-01-08', '2024-01-09', interval_seconds=3600)
    withdraw_holdout(root, unused)
    replacement = reserve_holdout(root, '2024-01-08', '2024-01-09', interval_seconds=3600)
    assert replacement != unused
    for interval in (True, 3600.0, 172800, 7, 0):
        with pytest.raises(ValueError):
            reserve_holdout(root, '2024-01-10', '2024-01-11', interval_seconds=interval)
    for start in ('2024-1-01', '2024-02-30', '2024-01-01T00:00:00'):
        with pytest.raises(ValueError):
            reserve_holdout(root, start, '2024-01-11', interval_seconds=3600)
    assert run_git(['check-ignore', 'manifests/holdout.sqlite3'], cwd=root).returncode == 0
    assert run_git(['check-ignore', 'manifests/holdout-project.json'], cwd=root).returncode == 1
    assert json.loads((root / 'manifests/holdout-project.json').read_text())['project_id'] == history.bind(root).project_id
    gap = raw.head(96).filter(pl.col('datetime') != raw['datetime'][48])
    admitted = admit_frame(gap, interval=3600, lower_ns=history.utc_day('2024-01-01'), upper_ns=history.utc_day('2024-01-05'))
    assert admitted.height == 95
    with pytest.raises(ValueError, match='complete native support'):
        admit_frame(gap, interval=3600, lower_ns=history.utc_day('2024-01-01'), upper_ns=history.utc_day('2024-01-05'), complete=True)
    with pytest.raises(ValueError, match='native interval/grid'):
        admit_frame(pl.read_parquet(_QUARTER).head(96), interval=3600, lower_ns=history.utc_day('2025-01-01'), upper_ns=history.utc_day('2025-01-02'))
    with pytest.raises(ValueError, match='increasing'):
        admit_frame(raw.reverse(), interval=3600, lower_ns=None, upper_ns=history.utc_day('2024-01-05'))


def test_holdout_research_provenance(tmp_path):
    root = _project(tmp_path / 'project')
    config = _configuration(root)
    reserve_holdout(root, '2024-01-05', '2024-01-06', interval_seconds=3600)
    directory = root / 'results'
    compiled = CompiledSFD(config, project_root=root)
    loop = UniversalExperimentLoop(sfd=compiled, project_root=root, experiment_dir=directory,
                                   search_strategy=build_search_strategy(config), yaml_reference=config)
    loop.run('guarded', n_permutations=1, progress_bar=False, post_processing=True)
    metadata = json.loads((directory / 'metadata.json').read_text())
    from limen.experiment import _round_storage
    with _round_storage.open_text(_round_storage.round_path(directory)) as stream:
        rounds = [json.loads(line) for line in stream if line.strip()]
    assert len(rounds) == 1
    provenance = rounds[0]['holdout']
    assert provenance['project_id'] == metadata['holdout']['project_id']
    assert provenance['round_id'] == rounds[0]['round_id']
    assert history.research_record(history.bind(root), provenance['research_record_id'])['round_id'] == rounds[0]['round_id']
    assert provenance['source']['materialized']['rows'] > provenance['source']['admitted']['rows'] == 96
    assert provenance['source']['kind'] == 'local_file'
    assert provenance['input_schema'] and provenance['model_schema']
    assert all(item['name'] not in ('datetime', 'quantile_flag', '__backtest_source_row__') for item in provenance['model_schema'])
    assert provenance['splits']['test']['last_close_ns'] == history.utc_day('2024-01-05')
    sensors = Trainer(directory, project_root=root).train([rounds[0]['round_id']])
    fitted = sensors[0]._fitted_params['_holdout']
    assert fitted['role'] == 'trainer'
    assert fitted['experiment_record_id'] == provenance['research_record_id']
    assert fitted['input_schema'] == provenance['input_schema']
    assert fitted['model_schema'] == provenance['model_schema']
    replay = Log(uel_object=loop)
    assert replay._get_test_data_with_all_cols(0).height > 0
    retained = replay.data
    for changed in (retained.slice(1), pl.read_parquet(_FIXTURE).head(168)):
        replay.data = changed
        with pytest.raises(ValueError, match='replay admitted source changed'):
            replay._get_test_data_with_all_cols(0)
    replay.data = retained
    manifest = compiled.manifest()
    manifest.set_pre_split_data_selector(lambda frame: frame.slice(1))
    with pytest.raises(ValueError, match='replay selection/split geometry changed'):
        replay._get_test_data_with_all_cols(0)
    manifest.pre_split_data_selector = None
    with history.connection(root / 'manifests/holdout.sqlite3') as db:
        assert {'experiment', 'trainer', 'replay'} <= {row[0] for row in db.execute('SELECT DISTINCT role FROM research')}
    # The same real source supports an ordinary rule-based experiment.
    rule = copy.deepcopy(config)
    body = rule['sfd']['manifest']
    body.update(type='rule_based', indicators=[], reference_architecture='limen.sfd.reference_architecture.rule_based.rule_based',
                strategy={'conditions': [{'id': 'up', 'type': 'relative', 'column': 'close', 'operator': '>', 'other_column': 'open'}], 'entry': 'up'})
    for key in ('target', 'scaler'):
        body.pop(key)
    rule['sfd']['params'] = {'sharpe_std_threshold': [0.5]}
    rule_compiled = CompiledSFD(rule, project_root=root)
    native = UniversalExperimentLoop(sfd=rule_compiled, project_root=root)
    native.run(str(root / 'rule'), n_permutations=1, progress_bar=False, prep_each_round=True)
    rule_csv = root / 'rule.csv'
    retained_csv = rule_csv.read_bytes()
    nested = subprocess.run(
        [sys.executable, '-m', 'tests.run',
         'tests/test_runtime_tracking.py::test_execute_test_suite_writes_profile_and_stops_on_first_failure', '-q'],
        cwd=Path(__file__).resolve().parents[1],
        env=dict(os.environ, LIMEN_RUNTIME_PROFILE_PATH=str(root / 'nested-runtime.json')),
        capture_output=True, text=True, check=False,
    )
    assert nested.returncode == 0, nested.stdout + nested.stderr
    assert rule_csv.read_bytes() == retained_csv
    rule_data = rule_compiled.manifest().prepare_data(native.data, {'sharpe_std_threshold': 0.5})
    assert rule_data['holdout']['schema_reason'] == 'rule-based-no-fitted-model'
    assert rule_data['holdout']['input_schema'] is None
    release_holdout(root, next(iter([row[0] for row in _reservations(root)])))
    with pytest.raises(ValueError, match='revision changed'):
        replay._get_test_data_with_all_cols(0)


def _reservations(root):
    with history.connection(root / 'manifests/holdout.sqlite3') as db:
        return db.execute("SELECT id FROM reservations WHERE state='active'").fetchall()


def test_holdout_revision_admission(tmp_path, monkeypatch, capsys):
    root = _project(tmp_path / 'project')
    config = _configuration(root)
    old = UniversalExperimentLoop(sfd=CompiledSFD(config, project_root=root), project_root=root)
    assert not (root / 'manifests/holdout.sqlite3').exists()
    custom = UniversalExperimentLoop(data=old.data, sfd=SimpleNamespace(__name__='custom', params=lambda: {'choice': [1]}, prep=lambda *_: {}, model=lambda *_: {}), project_root=root)
    for invalid in (None, lambda: None):
        with pytest.raises((TypeError, ValueError)):
            UniversalExperimentLoop(data=old.data, sfd=SimpleNamespace(manifest=invalid, params=lambda: {}), project_root=root)
    empty = history.bind(root)
    with pytest.raises(ValueError, match='active reservation'):
        history.start_research(empty, 'experiment')
    assert not (root / 'manifests/holdout.sqlite3').exists()
    for function in (history.finish_research, history.research_record):
        args = (empty, str(uuid4()), {}) if function is history.finish_research else (empty, str(uuid4()))
        with pytest.raises(sqlite3.OperationalError):
            function(*args)
    assert not (root / 'manifests/holdout.sqlite3').exists()
    ordinary = tmp_path / 'ordinary'
    ordinary.mkdir()
    (ordinary / 'limen.toml').write_text('[project]\nname="ordinary"\n')
    assert history.bind(ordinary) is None
    context = multiprocessing.get_context('fork' if 'fork' in multiprocessing.get_all_start_methods() else 'spawn')
    parent, child = context.Pipe()
    process = context.Process(target=_lease_worker, args=(str(root), child))
    process.start()
    try:
        assert parent.poll(10) and parent.recv() == 'ready'
        with history.admission(root):
            parent.send('reserve')
            assert parent.poll(5)
            assert parent.recv()[0] == 'OperationalError'
        parent.send('reserve')
        assert parent.poll(5)
        assert parent.recv()[0] == 'ok'
        binding = history.bind(root)
        with history.admission(root):
            record = history.start_research(binding, 'experiment')
            parent.send('research')
            assert parent.poll(5)
            assert parent.recv()[0] == 'ok'
            history.finish_research(binding, record, {'complete': True})
        with pytest.raises(ValueError, match='already finalized'):
            history.finish_research(binding, record, {})
        with pytest.raises(ValueError, match='unknown'):
            history.finish_research(binding, str(uuid4()), {})
        for stale in (old, custom):
            with pytest.raises(ValueError, match='revision changed'):
                stale.run('stale', n_permutations=1, progress_bar=False)
    finally:
        parent.send('stop')
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
    assert process.exitcode == 0
    new = UniversalExperimentLoop(sfd=CompiledSFD(config, project_root=root), project_root=root)
    for kwargs in ({'prep': lambda *_: {}}, {'model': lambda *_: {}}):
        with pytest.raises(ValueError, match='refuses'):
            new.run('unsupported', n_permutations=1, progress_bar=False, **kwargs)
    new.data = pl.read_parquet(_FIXTURE).head(168)
    new.run(str(root / 'mutated-frame'), n_permutations=1, progress_bar=False, prep_each_round=True)
    assert new.data.height == 96
    new.data.extend(pl.read_parquet(_FIXTURE).slice(96, 24))
    new.run(str(root / 'extended-frame'), n_permutations=1, progress_bar=False, prep_each_round=True, post_processing=True)
    source = new._alignment[0]['holdout']['source']
    assert source['kind'] == 'caller_frame'
    assert source['materialized']['rows'] == 120
    assert source['admitted']['rows'] == new.data.height == 96
    # A file-location hint binds even when CLI is launched outside the project.
    monkeypatch.chdir(tmp_path)
    config['uel']['workers'] = 2
    config['sfd']['params']['n_jobs'] = [1]
    assert run_profile(root / 'research.yaml') is True
    measured = profile(CompiledSFD(config, project_root=root))
    assert measured.sample_permutations_completed > 0 and not measured.errors
    (root / 'research.yaml').write_text(json.dumps(config))
    assert run_experiment(root / 'research.yaml', results_base=root, progress_bar=False) is True
    assert 'Experiment complete' in capsys.readouterr().out
    other = _project(tmp_path / 'other')
    with pytest.raises(ValueError, match='root hints disagree'):
        UniversalExperimentLoop(sfd=CompiledSFD(_configuration(other), project_root=other), project_root=root)
    sidecar = root / 'manifests/holdout.sqlite3-wal'
    sidecar.write_text('')
    run_git(['add', '--force', 'manifests/holdout.sqlite3-wal'], cwd=root).check_returncode()
    with pytest.raises(ValueError, match='must not be tracked'):
        history.bind(root)
    run_git(['rm', '--cached', '--', 'manifests/holdout.sqlite3-wal'], cwd=root).check_returncode()
    sidecar.unlink()
    with history.connection(root / 'manifests/holdout-lease.sqlite3') as db:
        db.execute('PRAGMA journal_mode=WAL')
    with pytest.raises(ValueError, match='rollback-journal'):
        history.bind(root)
    with history.connection(root / 'manifests/holdout-lease.sqlite3') as db:
        db.execute('PRAGMA journal_mode=DELETE')
    for filename, label in (('holdout.sqlite3', 'history'), ('holdout-lease.sqlite3', 'lease')):
        with history.connection(root / 'manifests' / filename) as db:
            db.execute(f'PRAGMA user_version={3 if label == "history" else 2}')
        with pytest.raises(ValueError, match=f'Unsupported holdout {label}'):
            history.bind(root)
        with history.connection(root / 'manifests' / filename) as db:
            db.execute('PRAGMA user_version=1')
    marker = root / 'manifests/holdout-project.json'
    assert marker.exists()
    (root / 'manifests/holdout.sqlite3').unlink()
    with pytest.raises(ValueError, match='history is missing'):
        history.bind(root)

    retry = CompiledSFD(config, project_root=root)
    for _ in range(2):
        with pytest.raises(ValueError, match='history is missing'):
            retry.manifest()
        assert retry._manifest_cache is None

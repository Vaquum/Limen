import io
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner
from polars.testing import assert_frame_equal
from ruamel.yaml import YAML

from limen.cli.main import cli
from limen.cli.commands.run import run_experiment
from limen.cli.commands.resume import run_resume
from limen.cohort import Cohort
from limen.experiment import UniversalExperimentLoop
from limen.experiment.errors import StrictModeError
from limen.experiment.manifest_core import MLManifest
from limen.inference import Trainer
from limen.yaml import CompiledSFD, build_search_strategy, validate
from limen.yaml.compiler import build_pruning_strategies
from limen.yaml.store import canonical_manifest_id
from tests.test_experiment_objective import objective_config
from tests.test_record_model_outputs import _config as output_config
from tests.test_record_model_outputs import recorded_source

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet'
_TRANSIENT = {'execution_time'}


def parallel_config(*, strategy='grid', objective=False, walk_forward=False):
    config = objective_config() if objective else output_config()
    config['metadata']['name'] = 'recorded_parallel'
    manifest = config['sfd']['manifest']
    manifest['data_source'] = {
        'method': 'limen.data.HistoricalData.get_any_file',
        'params': {'file_path_or_url': str(FIXTURE)},
    }
    manifest['split_dates']['test_predict_guard'] = False
    config['sfd']['params'].update(
        C=[0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0],
        n_jobs=[1], random_state=[42], max_iter=[300],
    )
    config['uel'].update(
        n_permutations=6, prep_each_round=True,
        search_strategy={'type': strategy, 'seed': 43},
        checkpoint_interval=3, feedback_interval=2,
        record_execution=True, record_model_outputs=True,
    )
    if objective:
        manifest['calibration']['threshold_function']['params']['threshold_step'] = 0.1
        config['uel']['pruning_strategies'] = [{
            'type': 'budget',
            'params': {'max_permutations': 5, 'trim_strategy': 'worst_first',
                       'check_after_pct': 0.0},
        }]
    else:
        config['uel']['pruning_strategies'] = [{
            'type': 'budget',
            'params': {'max_permutations': 5, 'trim_strategy': 'worst_first',
                       'metric': 'accuracy', 'check_after_pct': 0.0},
        }]
    if walk_forward:
        manifest.pop('split_dates')
        manifest['split_walk_forward'] = {
            'n_folds': 2, 'test_bars': 48, 'purge_bars': 4,
            'embargo_bars': 3, 'anchored': True,
        }
        config['uel']['n_permutations'] = 4
        config['uel'].pop('pruning_strategies')
    return config


def make_loop(config, directory, *, callback=None):
    return UniversalExperimentLoop(
        sfd=CompiledSFD(config), data=recorded_source(),
        experiment_dir=directory, yaml_reference=config,
        search_strategy=build_search_strategy(config),
        pruning_strategies=build_pruning_strategies(config),
        feedback_interval=config['uel']['feedback_interval'],
        checkpoint_interval=config['uel']['checkpoint_interval'],
        intra_callback=callback,
    )


def run_loop(config, directory, *, workers=None, resume=False, callback=None):
    loop = make_loop(config, directory, callback=callback)
    loop.run(
        config['metadata']['name'],
        n_permutations=config['uel']['n_permutations'],
        prep_each_round=True, progress_bar=False,
        record_execution=True, record_model_outputs=True,
        workers=workers, resume=resume,
    )
    return loop


def records(directory, name='round_data.jsonl'):
    path = directory / name
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _assert_same(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_same(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_same(left, right)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-10, abs=1e-12, nan_ok=True)
    else:
        assert actual == expected


def assert_artifacts(directory):
    frame = pl.read_csv(directory / 'results.csv')
    indices = frame['_round_index'].to_list()
    assert indices == list(range(frame.height))
    assert len(set(indices)) == frame.height
    successful = frame.filter(pl.col('strict_mode_error').is_null())
    saved = records(directory)
    assert [(row['_round_index'], row['round_id']) for row in saved] == list(
        zip(successful['_round_index'], successful['id'], strict=True)
    )
    for row in saved:
        assert row['round_params']['n_jobs'] == 1
        if 'folds' in row:
            assert len(row['folds']) == 2
        else:
            assert len(row['preds']) == len(row['probs'])
            expected = np.asarray(row['probs']) >= row['optimal_threshold'] if row['threshold_rule'] == '>=' else np.asarray(row['probs']) > row['optimal_threshold']
            np.testing.assert_array_equal(expected, row['preds'])
            assert row['execution'] is not None
            assert row['market'] is not None
    return frame


def assert_runs_equal(actual, expected):
    left, right = assert_artifacts(actual), assert_artifacts(expected)
    assert set(left.columns) == set(right.columns)
    columns = [name for name in right.columns if name not in _TRANSIENT]
    assert_frame_equal(left.select(columns), right.select(columns), check_exact=False,
                       rel_tol=1e-10, abs_tol=1e-12)
    _assert_same(records(actual), records(expected))
    left_audit, right_audit = records(actual, 'audit.jsonl'), records(expected, 'audit.jsonl')
    for row in [*left_audit, *right_audit]:
        row.pop('timestamp', None)
    _assert_same(left_audit, right_audit)
    assert [row['round'] for row in left_audit] == [row['round'] for row in right_audit]
    assert all(row['msq_state_after']['yielded_count'] == row['round'] + 1 for row in left_audit)


def _controlled_round(function, params, marker_directory, mode):
    directory = Path(marker_directory)
    index = params['_round_index']
    (directory / f'{index}.start').write_text(str(time.monotonic()))
    if mode == 'reorder' and index == 0:
        time.sleep(1.0)
    if mode in ('stop_zero', 'stop_outstanding') and (index == 0 or mode == 'stop_zero'):
        time.sleep(15)
    if mode == 'stop_final' and index == 3:
        time.sleep(15)
    if mode == 'dead' and index == 1:
        os._exit(17)
    original = MLManifest.run_model

    def controlled(manifest, data, round_params):
        if mode == 'all_strict' or (mode == 'strict' and index in (0, 2)):
            raise StrictModeError(f'recorded control failure {index}')
        if mode == 'fatal' and index == 1:
            raise RuntimeError('recorded fatal worker control')
        return original(manifest, data, round_params)

    with patch.object(MLManifest, 'run_model', controlled):
        result = function(params)
    (directory / f'{index}.done').write_text(str(time.monotonic()))
    return result


class _ControlledPool(ProcessPoolExecutor):
    marker_directory = None
    mode = None

    def submit(self, function, params):
        return super().submit(_controlled_round, function, params,
                              str(self.marker_directory), self.mode)


def controlled_pool(directory, mode):
    directory.mkdir(parents=True, exist_ok=True)
    _ControlledPool.marker_directory = directory
    _ControlledPool.mode = mode
    return patch('limen.experiment._parallel_rounds.ProcessPoolExecutor', _ControlledPool)


@pytest.fixture(scope='module')
def parallel_campaigns(tmp_path_factory):
    root = tmp_path_factory.mktemp('parallel_recorded')
    campaigns = {}
    for name, options in (
        ('grid', {'objective': True}),
        ('random', {'strategy': 'random'}),
        ('folds', {'walk_forward': True}),
    ):
        config = parallel_config(**options)
        prefixes = []

        def boundary(log, msq):
            assert msq.yielded_count == log.height
            prefixes.append(log['_round_index'].to_list())

        serial = root / f'{name}_serial'
        run_loop(config, serial, workers=1, callback=boundary)
        parallel = root / f'{name}_parallel'
        markers = root / f'{name}_markers'
        serial_prefixes = prefixes.copy()
        prefixes.clear()
        with controlled_pool(markers, 'reorder'):
            run_loop(config, parallel, workers=4, callback=boundary)
        assert prefixes == serial_prefixes
        campaigns[name] = (config, serial, parallel, markers)
    return campaigns


@pytest.mark.parametrize('name', ('grid', 'random', 'folds'))
def test_recorded_parallel_round_parity_and_order(parallel_campaigns, name):
    config, serial, parallel, markers = parallel_campaigns[name]
    assert_runs_equal(parallel, serial)
    assert float((markers / '1.done').read_text()) < float((markers / '0.done').read_text())
    assert all(int(path.stem) <= 2 for path in markers.glob('*.start')
               if float(path.read_text()) < float((markers / '0.done').read_text()))
    metadata = json.loads((parallel / 'metadata.json').read_text())
    assert metadata['workers'] == 4
    assert metadata['worker_native_threads'] == 1
    assert metadata['yaml_reference'] == config
    assert metadata['manifest_id'] == canonical_manifest_id(config)
    assert metadata['manifest_id'] == json.loads((serial / 'metadata.json').read_text())['manifest_id']
    if name == 'grid':
        assert assert_artifacts(parallel)['val_backtest_total_return'].null_count() == 0
        assert records(parallel)[0]['optimal_threshold'] == 2.0
        assert records(parallel)[0]['threshold_rule'] == '>='
    if name == 'folds':
        assert_frame_equal(pl.read_parquet(parallel / 'trial_returns.parquet'),
                           pl.read_parquet(serial / 'trial_returns.parquet'))


def test_parallel_artifacts_reconstruct_existing_consumers(parallel_campaigns):
    _, serial, parallel, _ = parallel_campaigns['grid']
    bars = recorded_source()
    ids = assert_artifacts(parallel)['id'].to_list()[:2]
    members = Trainer(parallel, data=bars).train(ids)
    serial_members = Trainer(serial, data=bars).train(ids)
    for actual, expected in zip(members, serial_members, strict=True):
        assert actual.predict_all(bars) == expected.predict_all(bars)
    cohort = Cohort(experiment_log_path=str(parallel), permutation_ids=ids)
    cohort.set_members(members)
    raw = bars.tail(48)
    assert len(cohort.predict_all(raw)) == len(raw)
    assert all(member.round_params['n_jobs'] == 1 for member in members)


@pytest.mark.parametrize('workers', (True, False, 0, -1, 1.5, '4', None))
def test_parallel_workers_yaml_rejects_invalid_values(workers):
    config = parallel_config()
    config['uel']['workers'] = workers
    assert not validate(config).valid


@pytest.mark.parametrize('workers', (True, 0, -1, 1.5, '4'))
def test_parallel_workers_native_rejects_before_artifacts(tmp_path, workers):
    loop = make_loop(parallel_config(), tmp_path / 'invalid')
    with pytest.raises((ValueError, TypeError), match='workers'):
        loop.run('invalid_workers', workers=workers, prep_each_round=True, progress_bar=False)
    assert not (tmp_path / 'invalid').exists()


@pytest.mark.parametrize('case', ('n_jobs', 'null_jobs', 'negative_jobs', 'factorize', 'post_processing', 'standard', 'custom', 'architecture', 'context', 'prepare_once'))
def test_parallel_unsupported_admission_is_non_destructive(tmp_path, case):
    config = parallel_config()
    if case in ('n_jobs', 'null_jobs', 'negative_jobs'):
        config['sfd']['params']['n_jobs'] = [{'n_jobs': 2, 'null_jobs': None, 'negative_jobs': -1}[case]]
    if case == 'architecture':
        config['sfd']['manifest']['reference_architecture'] = 'limen.sfd.reference_architecture.random_binary'
    path = tmp_path / 'admission'
    loop = make_loop(config, path)
    if case == 'standard':
        loop._search_strategy = None
    if case == 'custom':
        loop.model = lambda data, round_params: {}
    options = {'factorize': True} if case == 'factorize' else {'post_processing': True} if case == 'post_processing' else {'context_params': {'n_jobs': 2}} if case == 'context' else {}
    with pytest.raises((ValueError, TypeError), match=r'parallel|workers|n_jobs|MSQ|compiled|factorize|post_processing'):
        loop.run('invalid_parallel', workers=4, prep_each_round=case != 'prepare_once', progress_bar=False, **options)
    assert not path.exists()


def test_parallel_cli_rejects_bad_workers_without_output(tmp_path):
    for value in ('0', '-1', 'not-an-integer'):
        result = CliRunner().invoke(cli, ['run', '--workers', value, str(tmp_path / 'unused.yaml')])
        assert result.exit_code != 0
    assert list(tmp_path.iterdir()) == []


def test_parallel_yaml_cli_precedence_and_resume_provenance(tmp_path, parallel_campaigns):
    config = parallel_config()
    config['uel']['workers'] = 4
    config['uel']['output_path'] = str(tmp_path / 'cli')
    assert validate(config).valid
    yaml_path = tmp_path / 'campaign.yaml'
    _write_yaml(config, yaml_path)
    with patch('limen.cli.commands.run.UniversalExperimentLoop') as loop_factory:
        assert run_experiment(yaml_path, progress_bar=False, workers=2)
        assert loop_factory.return_value.run.call_args.kwargs['workers'] == 2
        assert loop_factory.return_value.check_parallel_admission.call_args.args == (2, True)
    config['uel']['output_path'] = str(tmp_path / 'yaml')
    _write_yaml(config, yaml_path)
    with patch('limen.cli.commands.run.UniversalExperimentLoop') as loop_factory:
        assert run_experiment(yaml_path, progress_bar=False)
        assert loop_factory.return_value.run.call_args.kwargs['workers'] == 4
    without_workers = dict(config)
    without_workers['uel'] = {key: value for key, value in config['uel'].items() if key != 'workers'}
    assert canonical_manifest_id(config) != canonical_manifest_id(without_workers)
    source = parallel_campaigns['grid'][2]
    with patch('limen.cli.commands.resume.UniversalExperimentLoop') as loop_factory:
        assert run_resume(source, progress_bar=False)
        assert loop_factory.return_value.run.call_args.kwargs['workers'] == 4


def _write_yaml(config, path):
    stream = io.StringIO()
    YAML().dump(config, stream)
    path.write_text(stream.getvalue())


def _wait_for(path, process, *, timeout=40):
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert process.poll() is None, f'process exited {process.returncode} before {path.name}'
        assert time.monotonic() < deadline, f'timed out waiting for {path.name}'
        time.sleep(0.025)


def _subprocess_campaign(config_path, output, marker_directory, mode):
    config = YAML(typ='safe').load(Path(config_path).read_text())
    with controlled_pool(Path(marker_directory), mode):
        run_loop(config, Path(output), workers=4)


@pytest.mark.parametrize('phase', ('zero', 'outstanding', 'final'))
def test_parallel_sigterm_restores_matching_prefix(tmp_path, parallel_campaigns, phase):
    config = parallel_config(walk_forward=True)
    config_path = tmp_path / 'campaign.yaml'
    _write_yaml(config, config_path)
    directory, markers = tmp_path / 'stopped', tmp_path / 'markers'
    command = [sys.executable, '-m', 'tests.test_parallel_rounds', str(config_path),
               str(directory), str(markers), f'stop_{phase}']
    output = tmp_path / 'process.log'
    with output.open('w') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        try:
            _wait_for(markers / ('3.start' if phase == 'final' else '0.start'), process)
            if phase == 'outstanding':
                _wait_for(markers / '1.done', process)
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=20) == 0, output.read_text()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    checkpoint = json.loads((directory / 'checkpoint.json').read_text())
    prefix = checkpoint['metadata']['experiment_round'] + 1
    assert prefix == (3 if phase == 'final' else 0)
    assert assert_artifacts(directory).height == prefix
    run_loop(config, directory, resume=True)
    assert_runs_equal(directory, parallel_campaigns['folds'][2])


@pytest.mark.parametrize('mode', ('strict', 'all_strict', 'fatal', 'dead'))
def test_parallel_failure_prefix_and_recovery(tmp_path, mode):
    config = parallel_config()
    config['uel'].pop('pruning_strategies')
    config['uel']['n_permutations'] = 4
    directory = tmp_path / 'failure'
    with controlled_pool(tmp_path / 'markers', mode):
        if mode in ('fatal', 'dead'):
            with pytest.raises(RuntimeError, match=r'worker|process|fatal|terminated'):
                run_loop(config, directory, workers=4)
        else:
            run_loop(config, directory, workers=4)
    frame = assert_artifacts(directory)
    checkpoint = json.loads((directory / 'checkpoint.json').read_text())
    assert checkpoint['metadata']['experiment_round'] + 1 == frame.height
    if mode == 'strict':
        assert frame['strict_mode_error'].is_not_null().to_list() == [True, False, True, False]
        assert [row['_round_index'] for row in records(directory)] == [1, 3]
    elif mode == 'all_strict':
        assert frame.height == 4 and frame['strict_mode_error'].null_count() == 0
        assert records(directory) == []
    else:
        run_loop(config, directory, resume=True)
        assert assert_artifacts(directory).height == 4


def test_parallel_resume_rejects_changed_policy_and_broken_evidence(parallel_campaigns, tmp_path):
    config, _, source, _ = parallel_campaigns['grid']
    for name in ('policy', 'missing', 'corrupt', 'duplicate'):
        directory = tmp_path / name
        shutil.copytree(source, directory)
        if name == 'missing':
            (directory / 'round_data.jsonl').unlink()
        elif name == 'corrupt':
            saved = records(directory)
            saved[0]['round_id'] = 'unrecorded-identity'
            (directory / 'round_data.jsonl').write_text('\n'.join(json.dumps(row) for row in saved) + '\n')
        elif name == 'duplicate':
            frame = pl.read_csv(directory / 'results.csv')
            frame.with_columns(pl.when(pl.col('_round_index') == 1).then(0).otherwise(pl.col('_round_index')).alias('_round_index')).write_csv(directory / 'results.csv')
        before = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
        with pytest.raises(ValueError, match=r'worker|policy|round|evidence|record|prefix|occurrence|identity'):
            run_loop(config, directory, workers=2 if name == 'policy' else None, resume=True)
        assert {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()} == before


def test_parallel_serial_omitted_matches_explicit_one(tmp_path):
    config = parallel_config()
    config['uel']['n_permutations'] = 1
    config['uel'].pop('pruning_strategies')
    implicit, explicit = tmp_path / 'implicit', tmp_path / 'explicit'
    run_loop(config, implicit)
    run_loop(config, explicit, workers=1)
    assert_runs_equal(implicit, explicit)
    metadata = json.loads((implicit / 'metadata.json').read_text())
    assert 'workers' not in metadata and 'worker_native_threads' not in metadata


if __name__ == '__main__':
    _subprocess_campaign(*sys.argv[1:])

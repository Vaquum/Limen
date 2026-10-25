import gzip
from limen.experiment import _round_storage as _storage
import csv
import faulthandler
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from ruamel.yaml import YAML
from threadpoolctl import threadpool_info, threadpool_limits

from limen.yaml import CompiledSFD, build_search_strategy, validate
from limen.backtest.execution_events import with_predictions
from limen.backtest.trade_contract import json_value
from limen.backtest.trade_execution import trade_execution
from limen.experiment import UniversalExperimentLoop
from limen.experiment.errors import StrictModeError
from limen.experiment._parallel_worker import freeze_sources, startup_limits
from limen.experiment.manifest_core import MLManifest
from limen.yaml.compiler import build_pruning_strategies
from limen.yaml.store import canonical_manifest_id

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet'
_TRANSIENT = {'execution_time'}


def recorded_source(**kwargs):
    from tests.test_record_model_outputs import recorded_source as _recorded_source

    data = _recorded_source(**kwargs).head(900)
    interval = (data['datetime'][1] - data['datetime'][0]).total_seconds()
    return data.with_columns(pl.lit(interval).alias('base_interval'))


def parallel_config(*, strategy='grid', objective=False, walk_forward=False):
    from tests.test_experiment_objective import objective_config
    from tests.test_record_model_outputs import _config as output_config

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
        manifest['calibration']['threshold_function']['params'].update(
            threshold_min=0.5, threshold_max=0.5, threshold_step=0.1,
        )
        config['uel']['pruning_strategies'] = [{
            'type': 'budget',
            'params': {'max_permutations': 5, 'trim_strategy': 'worst_first',
                       'check_after_pct': 0.0},
        }]
    else:
        config['uel']['pruning_strategies'] = [{
            'type': 'budget',
            'params': {'max_permutations': 5, 'trim_strategy': 'worst_first',
                       'metric': 'auc', 'maximize': True, 'check_after_pct': 0.0},
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


def run_loop(config, directory, *, workers=None, resume=False, callback=None, context_params=None):
    loop = make_loop(config, directory, callback=callback)
    with threadpool_limits(limits=1):
        loop.run(
            config['metadata']['name'],
            n_permutations=config['uel']['n_permutations'],
            prep_each_round=True, progress_bar=False,
            record_execution=True, record_model_outputs=True,
            workers=workers, resume=resume, context_params=context_params,
            recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True),
        )
    return loop


def records(directory, name='round_data.jsonl'):
    path = _storage.round_path(directory) if name == 'round_data.jsonl' else directory / name
    text = _storage.text(path) if name == 'round_data.jsonl' and path.exists() else path.read_text() if path.exists() else ''
    return [json.loads(line) for line in text.splitlines()]


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
            if 'trade_ledger' in row:
                assert row['trade_ledger']['states']
            else:
                assert row['execution'] is not None
                assert row['market'] is not None
    return frame


def assert_runs_equal(actual, expected):
    left, right = assert_artifacts(actual), assert_artifacts(expected)
    assert set(left.columns) == set(right.columns)
    assert_frame_equal(left.select('id', '_round_index'),
                       right.select('id', '_round_index'))
    left_audit, right_audit = records(actual, 'audit.jsonl'), records(expected, 'audit.jsonl')
    for row in [*left_audit, *right_audit]:
        row.pop('timestamp', None)
    _assert_same(left_audit, right_audit)
    assert [row['round'] for row in left_audit] == [row['round'] for row in right_audit]
    assert all(row['msq_state_after']['yielded_count'] == row['round'] + 1 for row in left_audit)
    columns = [name for name in right.columns if name not in _TRANSIENT]
    assert_frame_equal(left.select(columns), right.select(columns), check_exact=False,
                       rel_tol=1e-10, abs_tol=1e-12)
    _assert_same(records(actual), records(expected))


def _controlled_round(function, params, marker_directory, mode):
    directory = Path(marker_directory)
    index = params['_round_index']
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
    if os.name == 'posix':
        assert os.getpgrp() != os.getpgid(os.getppid())
    (directory / f'{index}.start').write_text(str(time.monotonic()))
    if mode == 'reorder' and index == 0:
        deadline = time.monotonic() + 40
        while not (directory / '1.done').exists():
            assert time.monotonic() < deadline, 'timed out waiting for round 1 completion'
            time.sleep(0.025)
    if (mode in ('stop_zero', 'stop_outstanding') and (index == 0 or mode == 'stop_zero')) or (mode == 'dead' and index == 0):
        time.sleep(15)
    if mode == 'stop_final' and index == 3:
        time.sleep(15)
    if mode == 'dead' and index == 1:
        (directory / f'{index}.failed').touch()
        os._exit(17)
    method = 'prepare_data' if mode == 'strict' else 'run_model'
    original = getattr(MLManifest, method)

    def controlled(manifest, data, round_params):
        if mode == 'strict' and (index < 3 or index == 4):
            if index == 0:
                raise StrictModeError()
            raise StrictModeError(f'recorded control failure {index}')
        if mode == 'fatal' and index == 1:
            (directory / f'{index}.failed').touch()
            raise RuntimeError('recorded fatal worker control')
        return original(manifest, data, round_params)

    with patch.object(MLManifest, method, controlled):
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


def _serial_round_proof(config_path, destination):
    configs = YAML(typ='safe').load(Path(config_path).read_text())
    root = Path(destination)
    for name, config in configs.items():
        prefixes = []
        pools = []

        def boundary(log, msq, prefixes=prefixes, pools=pools, name=name):
            assert msq.yielded_count == log.height
            if name == 'random' and log.height == 3:
                scores = log.group_by('C').agg(pl.col('auc').mean())['auc']
                assert scores.eq(scores.min()).sum() == 1
            assert pl.thread_pool_size() == 1
            actual = threadpool_info()
            assert actual and all(pool['num_threads'] == 1 for pool in actual)
            pools.extend(actual)
            prefixes.append(log['_round_index'].to_list())

        run_loop(config, root / f'{name}_serial', workers=1, callback=boundary)
        assert prefixes and pools
        (root / f'{name}_serial.json').write_text(json.dumps({
            'prefixes': prefixes, 'native_pools': pools, 'polars_threads': pl.thread_pool_size(),
        }))


@pytest.fixture(scope='module')
def parallel_campaigns(tmp_path_factory):
    root = tmp_path_factory.mktemp('parallel_recorded')
    campaigns = {}
    configs = {name: parallel_config(**options) for name, options in (
        ('grid', {'objective': True}),
        ('random', {'strategy': 'random'}),
        ('folds', {'walk_forward': True}),
    )}
    serial_config = root / 'serial_campaigns.yaml'
    _write_yaml(configs, serial_config)
    with startup_limits():
        subprocess.run([
            sys.executable, '-c',
            'from tests.test_parallel_rounds import _serial_round_proof\nimport sys\n_serial_round_proof(*sys.argv[1:])',
            str(serial_config), str(root),
        ], cwd=ROOT, check=True, timeout=40)
    for name, config in configs.items():
        prefixes = []

        def boundary(log, msq, prefixes=prefixes, name=name):
            assert msq.yielded_count == log.height
            if name == 'random' and log.height == 3:
                scores = log.group_by('C').agg(pl.col('auc').mean())['auc']
                assert scores.eq(scores.min()).sum() == 1
            prefixes.append(log['_round_index'].to_list())

        serial = root / f'{name}_serial'
        parallel = root / f'{name}_parallel'
        markers = root / f'{name}_markers'
        serial_proof = json.loads((root / f'{name}_serial.json').read_text())
        assert serial_proof['polars_threads'] == 1
        assert all(pool['num_threads'] == 1 for pool in serial_proof['native_pools'])
        with controlled_pool(markers, 'reorder'), patch.object(_ControlledPool, '__init__', autospec=True, side_effect=ProcessPoolExecutor.__init__) as executor:
            run_loop(config, parallel, workers=4, callback=boundary)
        assert executor.call_args.kwargs['max_workers'] == 3
        assert prefixes == serial_proof['prefixes']
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
        record = records(parallel)[0]
        assert record['optimal_threshold'] == 2.0
        assert record['threshold_rule'] == '>='
        manifest = CompiledSFD(config).manifest()
        prepared = manifest.prepare_data(recorded_source(), record['round_params'])
        context = prepared['_trade_context']
        ledger = trade_execution(with_predictions(context.partitions[2], record['preds']), context.policy)
        _assert_same(record['trade_ledger'], json_value({key: getattr(ledger, key).to_dicts() for key in ('states', 'intents', 'fills', 'episodes', 'funding')}))
        saved_metrics = pl.read_csv(parallel / 'results.csv').row(0, named=True)
        for key, value in ledger.metrics.items():
            assert saved_metrics[f'backtest_{key}'] == pytest.approx(value)
    if name == 'folds':
        assert_frame_equal(pl.read_parquet(parallel / 'trial_returns.parquet'),
                           pl.read_parquet(serial / 'trial_returns.parquet'))


def test_parallel_artifacts_reconstruct_existing_consumers(parallel_campaigns):
    from limen.cohort import Cohort
    from limen.inference import Trainer

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


@pytest.mark.parametrize('case', ('n_jobs', 'null_jobs', 'negative_jobs', 'factorize', 'post_processing', 'standard', 'custom', 'context', 'prepare_once', 'params_override', 'prep_override', 'model_override'))
def test_parallel_unsupported_admission_is_non_destructive(tmp_path, case):
    config = parallel_config()
    if case in ('n_jobs', 'null_jobs', 'negative_jobs'):
        config['sfd']['params']['n_jobs'] = [{'n_jobs': 2, 'null_jobs': None, 'negative_jobs': -1}[case]]
    path = tmp_path / 'admission'
    loop = make_loop(config, path)
    if case == 'standard':
        loop._search_strategy = None
    if case == 'custom':
        loop.model = lambda data, round_params: {}
    options = {'factorize': True} if case == 'factorize' else {'post_processing': True} if case == 'post_processing' else {'context_params': {'n_jobs': 2}} if case == 'context' else {}
    if case.endswith('_override'):
        options = {case.removesuffix('_override'): lambda: {}}
    with pytest.raises((ValueError, TypeError), match=r'(?i)parallel|workers|n_jobs|MSQ|compiled|factorize|post_processing|prep_each_round|preparation'):
        loop.run('invalid_parallel', workers=4, prep_each_round=case != 'prepare_once', progress_bar=False, **options)
    assert not path.exists()
    assert loop._parallel_sources is None
    if case.endswith('_override'):
        loop.data = recorded_source().slice(12, 900)

        def observe_fresh_sources(*args, **kwargs):
            assert_frame_equal(loop._parallel_sources.learning, loop.data)
            raise RuntimeError('recorded fresh-source control boundary')

        with patch.object(loop, '_run', side_effect=observe_fresh_sources), pytest.raises(RuntimeError, match='recorded fresh-source control boundary'):
            loop.run('fresh_admission', workers=4, prep_each_round=True, progress_bar=False)
        assert not path.exists()
        assert loop._parallel_sources is None


def test_parallel_cli_rejects_bad_workers_without_output(tmp_path):
    from click.testing import CliRunner

    from limen.cli.main import cli

    for value in ('0', '-1', 'not-an-integer'):
        result = CliRunner().invoke(cli, ['run', '--workers', value, str(tmp_path / 'unused.yaml')])
        assert result.exit_code != 0
    assert list(tmp_path.iterdir()) == []


def test_parallel_yaml_cli_precedence_and_resume_provenance(tmp_path, parallel_campaigns):
    from limen.cli.commands.run import run_experiment
    from limen.cli.commands.resume import run_resume

    config = parallel_config()
    config['uel']['workers'] = 4
    config['uel']['output_path'] = 'cli'
    assert validate(config).valid
    yaml_path = tmp_path / 'campaign.yaml'
    _write_yaml(config, yaml_path)
    with patch('limen.cli.commands.run.UniversalExperimentLoop') as loop_factory:
        loop_factory.return_value.interrupted = False
        assert run_experiment(yaml_path, results_base=tmp_path, progress_bar=False, workers=2)
        assert loop_factory.return_value.run.call_args.kwargs['workers'] == 2
        assert loop_factory.return_value.check_parallel_admission.call_args.args == (2, True)
    config['uel']['output_path'] = 'yaml'
    _write_yaml(config, yaml_path)
    with patch('limen.cli.commands.run.UniversalExperimentLoop') as loop_factory:
        loop_factory.return_value.interrupted = False
        assert run_experiment(yaml_path, results_base=tmp_path, progress_bar=False)
        assert loop_factory.return_value.run.call_args.kwargs['workers'] == 4
    without_workers = dict(config)
    without_workers['uel'] = {key: value for key, value in config['uel'].items() if key != 'workers'}
    assert canonical_manifest_id(config) != canonical_manifest_id(without_workers)
    source = parallel_campaigns['grid'][2]
    with patch('limen.cli.commands.resume.UniversalExperimentLoop') as loop_factory:
        loop_factory.return_value.interrupted = False
        assert run_resume(source, progress_bar=False)
        assert loop_factory.return_value.run.call_args.kwargs['workers'] == 4


def _write_yaml(config, path):
    stream = io.StringIO()
    YAML().dump(config, stream)
    path.write_text(stream.getvalue())


def _wait_for(path, process, output, *, timeout=40):
    deadline = time.monotonic() + timeout
    while not path.exists():
        returncode = process.poll()
        if path.exists():
            return
        if returncode is not None:
            pytest.fail(f'process exited {returncode} before {path.name}:\n{output.read_text()}')
        if time.monotonic() >= deadline:
            pytest.fail(f'timed out waiting for {path.name}:\n{output.read_text()}')
        time.sleep(0.025)


def _subprocess_campaign(config_path, output, marker_directory, mode):
    from limen.cli.main import cli

    assert CompiledSFD.__name__ == 'CompiledSFD'
    assert cli.name == 'cli'
    config = YAML(typ='safe').load(Path(config_path).read_text())
    checkpoint = UniversalExperimentLoop._checkpoint

    def await_initial_stop(loop, *args, **kwargs):
        checkpoint(loop, *args, **kwargs)
        deadline = time.monotonic() + 40
        while not loop._shutdown_requested:
            assert time.monotonic() < deadline, 'timed out waiting for initial-checkpoint SIGTERM'
            time.sleep(0.025)

    stop_boundary = patch.object(UniversalExperimentLoop, '_checkpoint', await_initial_stop) if mode == 'stop_zero' else nullcontext()
    faulthandler.dump_traceback_later(10)
    try:
        with controlled_pool(Path(marker_directory), mode), stop_boundary:
            run_loop(config, Path(output), workers=2)
    finally:
        faulthandler.cancel_dump_traceback_later()


@pytest.mark.parametrize('phase', ('zero', 'final', 'group'))
def test_parallel_sigterm_restores_matching_prefix(tmp_path, parallel_campaigns, phase):
    campaign = 'grid' if phase == 'zero' else 'folds'
    config = parallel_campaigns[campaign][0]
    config_path = tmp_path / 'campaign.yaml'
    _write_yaml(config, config_path)
    directory, markers = tmp_path / 'stopped', tmp_path / 'markers'
    mode = 'outstanding' if phase == 'group' else phase
    command = [sys.executable, '-m', 'tests.test_parallel_rounds', str(config_path),
               str(directory), str(markers), f'stop_{mode}']
    output = tmp_path / 'process.log'
    with output.open('w') as stream:
        with startup_limits():
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        try:
            first = directory / 'checkpoint.json' if phase == 'zero' else markers / ('3.start' if phase == 'final' else '0.start')
            _wait_for(first, process, output=output)
            if phase in ('outstanding', 'group'):
                _wait_for(markers / '1.done', process, output=output)
            if phase == 'group':
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.send_signal(signal.SIGTERM)
            try:
                returncode = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pytest.fail(f'SIGTERM shutdown exceeded 20 seconds:\n{output.read_text()}')
            assert returncode == 0, output.read_text()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    checkpoint = json.loads((directory / 'checkpoint.json').read_text())
    prefix = checkpoint['metadata']['experiment_round'] + 1
    assert prefix == (3 if phase == 'final' else 0)
    assert assert_artifacts(directory).height == prefix
    with patch('limen.experiment._parallel_rounds.ProcessPoolExecutor', wraps=ProcessPoolExecutor) as executor:
        resumed = run_loop(config, directory, resume=True)
    if phase == 'final':
        assert executor.call_args.kwargs['max_workers'] == 1
        assert json.loads((directory / 'metadata.json').read_text())['workers'] == 2
    if phase == 'zero':
        assert resumed.experiment_log.schema['val_backtest_total_return'] == pl.Float64
        assert resumed.experiment_log['val_backtest_total_return'].is_finite().all()
    assert_runs_equal(directory, parallel_campaigns[campaign][1])


@pytest.mark.parametrize('mode', ('strict', 'fatal', 'dead'))
def test_parallel_failure_prefix_and_recovery(tmp_path, mode):
    config = parallel_config(objective=mode == 'strict')
    config['uel'].pop('pruning_strategies')
    config['uel']['n_permutations'] = 6 if mode == 'strict' else 4
    if mode == 'strict':
        config['sfd']['params'].update(A_tag=['', None], context_value=['001'])
    context_params: dict[str, object] | None = {'context_tag': '002', 'context_value': 2} if mode == 'strict' else None
    directory = tmp_path / 'failure'
    if mode in ('fatal', 'dead'):
        config_path = tmp_path / 'campaign.yaml'
        _write_yaml(config, config_path)
        output = tmp_path / 'process.log'
        command = [sys.executable, '-m', 'tests.test_parallel_rounds', str(config_path),
                   str(directory), str(tmp_path / 'markers'), mode]
        with output.open('w') as stream:
            with startup_limits():
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            try:
                _wait_for(tmp_path / 'markers' / '1.failed', process, output=output)
                try:
                    returncode = process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    pytest.fail(f'{mode} worker shutdown exceeded 20 seconds:\n{output.read_text()}')
                failure_output = output.read_text()
                assert returncode != 0, failure_output
                assert 'Traceback (most recent call last)' in failure_output, failure_output
                assert 'Parallel worker' in failure_output, failure_output
                expected = 'recorded fatal worker control' if mode == 'fatal' else 'terminated abruptly'
                assert expected in failure_output, failure_output
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
    else:
        with controlled_pool(tmp_path / 'markers', mode):
            def stop_strict_prefix(log, msq):
                assert log.schema['val_backtest_total_return'] == pl.Float64
                if log.height == 3:
                    loop._shutdown_requested = True

            loop = make_loop(config, directory, callback=stop_strict_prefix)
            loop.run(config['metadata']['name'], n_permutations=6,
                     prep_each_round=True, progress_bar=False, workers=2,
                     record_execution=True, record_model_outputs=True, context_params=context_params,
                     recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True))
    frame = assert_artifacts(directory)
    checkpoint = json.loads((directory / 'checkpoint.json').read_text())
    assert checkpoint['metadata']['experiment_round'] + 1 == frame.height
    if mode in ('fatal', 'dead'):
        assert frame.height in (0, 1)
        if mode == 'fatal':
            assert frame.height == 1
        strategy = build_search_strategy(config)
        identities = [next(strategy)['_id'] for _ in range(2)]
        assert f'round {frame.height} ({identities[frame.height]})' in failure_output, failure_output
    if mode == 'strict':
        assert frame.height == 3 and frame['strict_mode_error'].null_count() == 0
        assert frame['strict_mode_error'][0] == 'StrictModeError'
        assert records(directory) == []
        sampled = build_search_strategy(config)
        assert checkpoint['failed_round_params'] == [
            {'_round_index': index, 'round_id': frame['id'][index],
             'round_params': {key: value for key, value in (next(sampled) | (context_params or {})).items() if key != '_id'}}
            for index in range(3)
        ]
        assert [entry['round_params']['A_tag'] for entry in checkpoint['failed_round_params']] == ['', None, '']
        for name in ('missing_failures', 'missing_failure', 'changed_failure_id', 'changed_failure_params', 'missing_failure_params', 'missing_one_context', 'missing_one_context_without_argument', 'missing_all_context', 'missing_all_context_without_argument'):
            rejected = tmp_path / name
            shutil.copytree(directory, rejected)
            saved = json.loads((rejected / 'checkpoint.json').read_text())
            if name == 'missing_failures':
                saved.pop('failed_round_params')
            elif name == 'missing_failure':
                saved['failed_round_params'].pop(0)
            elif name == 'changed_failure_id':
                saved['failed_round_params'][0]['round_id'] = 'unrecorded-identity'
            elif name == 'changed_failure_params':
                saved['failed_round_params'][0]['round_params']['C'] *= 2
            elif name in ('missing_one_context', 'missing_one_context_without_argument'):
                saved['failed_round_params'][0]['round_params'].pop('context_tag')
            elif name in ('missing_all_context', 'missing_all_context_without_argument'):
                for entry in saved['failed_round_params']:
                    entry['round_params'].pop('context_tag')
            else:
                saved['failed_round_params'][0]['round_params'].pop('C')
            (rejected / 'checkpoint.json').write_text(json.dumps(saved))
            before = {path.name: path.read_bytes() for path in rejected.iterdir() if path.is_file()}
            with pytest.raises(ValueError, match=r'failed|parameter|occurrence|record'):
                run_loop(config, rejected, resume=True, context_params=None if name.endswith('_without_argument') else context_params)
            assert {path.name: path.read_bytes() for path in rejected.iterdir() if path.is_file()} == before
        rejected = tmp_path / 'omitted_context'
        shutil.copytree(directory, rejected)
        before = {path.name: path.read_bytes() for path in rejected.iterdir() if path.is_file()}
        with pytest.raises(ValueError, match='context parameter declaration'):
            run_loop(config, rejected, resume=True)
        assert {path.name: path.read_bytes() for path in rejected.iterdir() if path.is_file()} == before
        feedback_heights = []

        def numeric_feedback(log, msq):
            assert log.schema['val_backtest_total_return'] == pl.Float64
            assert log['val_backtest_total_return'].drop_nulls().is_finite().all()
            assert log.schema['context_tag'] == pl.String
            assert log['context_tag'].to_list() == ['002'] * log.height
            assert log.schema['context_value'] == pl.Int64
            assert log['context_value'].to_list() == [2] * log.height
            assert log.schema['A_tag'] == pl.String
            assert log['A_tag'].to_list() == ['', None, '', None, ''][:log.height]
            feedback_heights.append(log.height)

        _storage.round_path(directory).unlink()
        with controlled_pool(tmp_path / 'resume_markers', mode):
            resumed = run_loop(config, directory, resume=True, callback=numeric_feedback, context_params=context_params)
        assert feedback_heights == [5]
        completed = assert_artifacts(directory)
        assert completed.height == 6
        assert completed['strict_mode_error'].is_not_null().to_list() == [True, True, True, False, True, False]
        scores = resumed.experiment_log['val_backtest_total_return']
        assert scores.dtype == pl.Float64 and scores.null_count() == 4
        assert len(scores.drop_nulls()) == 2
        assert scores.drop_nulls().is_finite().all()
        assert resumed.experiment_log['context_tag'].to_list() == ['002'] * 6
        assert resumed.experiment_log['context_value'].to_list() == [2] * 6
        assert resumed.experiment_log['A_tag'].to_list() == ['', None, '', None, '', None]
        with (directory / 'results.csv').open(newline='') as stream:
            assert [row['context_tag'] for row in csv.DictReader(stream)] == ['002'] * 6
        assert [row['_round_index'] for row in records(directory)] == [3, 5]
        checkpoint = json.loads((directory / 'checkpoint.json').read_text())
        assert checkpoint['metadata']['experiment_round'] == 5
        assert [entry['_round_index'] for entry in checkpoint['failed_round_params']] == [0, 1, 2, 4]
        assert [entry['round_params']['A_tag'] for entry in checkpoint['failed_round_params']] == ['', None, '', '']
    else:
        run_loop(config, directory, resume=True)
        assert assert_artifacts(directory).height == 4


def test_parallel_freezes_declared_execution_source_independently():
    manifest = CompiledSFD(parallel_config(objective=True)).manifest()
    source = manifest.data_source_config
    manifest.backtest_config = replace(manifest.backtest_config, execution_data_source=source)
    learning = recorded_source()
    declared = pl.read_parquet(FIXTURE)
    assert declared.height > learning.height
    with patch('limen.experiment._parallel_worker.DataSourceResolver.resolve', return_value=declared) as resolve:
        frozen = freeze_sources(manifest, learning)
    resolve.assert_called_once_with(source)
    assert_frame_equal(frozen.learning, learning)
    assert_frame_equal(frozen.execution, declared)


def test_parallel_resume_rejects_changed_policy_and_broken_evidence(parallel_campaigns, tmp_path):
    config, _, source, _ = parallel_campaigns['grid']
    for name in ('policy', 'missing', 'corrupt', 'duplicate', 'missing_params', 'empty_params', 'changed_params', 'changed_csv_params', 'missing_ledger', 'null_ledger', 'null_state'):
        directory = tmp_path / name
        shutil.copytree(source, directory)
        if name == 'missing':
            (_storage.round_path(directory)).unlink()
        elif name == 'corrupt':
            saved = records(directory)
            saved[0]['round_id'] = 'unrecorded-identity'
            (_storage.round_path(directory)).write_bytes(b''.join(gzip.compress((json.dumps(row) + '\n').encode(), compresslevel=1, mtime=0) for row in saved))
        elif name == 'duplicate':
            frame = pl.read_csv(directory / 'results.csv')
            frame.with_columns(pl.when(pl.col('_round_index') == 1).then(0).otherwise(pl.col('_round_index')).alias('_round_index')).write_csv(directory / 'results.csv')
        elif name == 'changed_csv_params':
            frame = pl.read_csv(directory / 'results.csv')
            frame.with_columns(pl.when(pl.col('_round_index') == 0).then(pl.col('C') * 2).otherwise(pl.col('C')).alias('C')).write_csv(directory / 'results.csv')
        elif name in ('missing_params', 'empty_params', 'changed_params', 'missing_ledger', 'null_ledger', 'null_state'):
            saved = records(directory)
            if name == 'missing_params':
                saved[0].pop('round_params')
            elif name == 'empty_params':
                saved[0]['round_params'] = {}
            elif name == 'changed_params':
                saved[0]['round_params']['C'] *= 2
            elif name == 'missing_ledger':
                saved[0].pop('trade_ledger')
            elif name == 'null_state':
                saved[0]['trade_ledger']['states'] = [None]
            else:
                saved[0]['trade_ledger'] = None
            (_storage.round_path(directory)).write_bytes(b''.join(gzip.compress((json.dumps(row) + '\n').encode(), compresslevel=1, mtime=0) for row in saved))
        if name != 'missing':
            assert records(directory)
        before = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
        with pytest.raises(ValueError, match=r'worker|policy|round|evidence|record|prefix|occurrence|identity|ledger'):
            run_loop(config, directory, workers=2 if name == 'policy' else None, resume=True)
        assert {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()} == before


def test_parallel_resume_rejects_null_snapshot_execution(parallel_campaigns, tmp_path):
    config, _, source, _ = parallel_campaigns['random']
    for field in ('execution', 'market'):
        directory = tmp_path / field
        shutil.copytree(source, directory)
        saved = records(directory)
        assert saved[0][field] is not None
        saved[0][field] = None
        (_storage.round_path(directory)).write_bytes(b''.join(gzip.compress((json.dumps(row) + '\n').encode(), compresslevel=1, mtime=0) for row in saved))
        before = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
        with pytest.raises(ValueError, match=r'execution|market|record|evidence'):
            run_loop(config, directory, resume=True)
        assert {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()} == before


def test_parallel_periodic_checkpoint_preserves_callback_filters(tmp_path):
    config = parallel_config()
    config['uel'].pop('pruning_strategies')
    config['sfd']['params']['tag'] = ['001']
    config['sfd']['params']['empty_tag'] = ['']
    config['sfd']['params']['numeric_tag'] = ['001']
    context_params = {'context_tag': '002', 'empty_context': '', 'none_context': None, 'numeric_tag': 2}
    assert validate(config).valid

    def recorded_parameter(params):
        return params['C'] == 3.0

    def remove_recorded_parameter(log, msq):
        if log.height == 3:
            msq.remove_custom(recorded_parameter)

    serial, parallel = tmp_path / 'serial', tmp_path / 'parallel'
    run_loop(config, serial, workers=1, callback=remove_recorded_parameter, context_params=context_params)
    run_loop(config, parallel, workers=2, callback=remove_recorded_parameter, context_params=context_params)
    left, right = assert_artifacts(parallel), assert_artifacts(serial)
    columns = [name for name in right.columns if name not in _TRANSIENT]
    assert_frame_equal(left.select(columns), right.select(columns), check_exact=False,
                       rel_tol=1e-10, abs_tol=1e-12)
    _assert_same(records(parallel), records(serial))
    assert pl.read_csv(parallel / 'results.csv')['C'].to_list() == [0.1, 0.3, 0.5, 1.0, 5.0, 10.0]
    assert json.loads((parallel / 'metadata.json').read_text())['context_parameter_keys'] == sorted(context_params)
    assert 'context_parameter_keys' not in json.loads((serial / 'metadata.json').read_text())
    for name in ('context_tag', 'empty_context', 'none_context', 'missing_context_keys', 'malformed_context_keys'):
        rejected = tmp_path / name
        shutil.copytree(parallel, rejected)
        if name in ('missing_context_keys', 'malformed_context_keys'):
            metadata = json.loads((rejected / 'metadata.json').read_text())
            if name == 'missing_context_keys':
                metadata.pop('context_parameter_keys')
            else:
                metadata['context_parameter_keys'] = [None]
            (rejected / 'metadata.json').write_text(json.dumps(metadata))
        else:
            saved = records(rejected)
            for entry in saved:
                entry['round_params'].pop(name)
            (_storage.round_path(rejected)).write_bytes(b''.join(gzip.compress((json.dumps(entry) + '\n').encode(), compresslevel=1, mtime=0) for entry in saved))
        before = {path.name: path.read_bytes() for path in rejected.iterdir() if path.is_file()}
        with pytest.raises(ValueError, match=r'context|parameter|record'):
            run_loop(config, rejected, resume=True)
        assert {path.name: path.read_bytes() for path in rejected.iterdir() if path.is_file()} == before
    for attempt in range(2):
        warning = pytest.warns(UserWarning, match='Checkpoint had non-restorable filters') if attempt == 0 else nullcontext()
        with warning, patch.object(ProcessPoolExecutor, '_launch_processes', side_effect=AssertionError('Completed resume launched workers')):
            resumed = run_loop(config, parallel, resume=True, context_params=context_params if attempt == 0 else None)
        with (parallel / 'results.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        for name, value in {'tag': '001', 'context_tag': '002', 'empty_tag': '', 'empty_context': ''}.items():
            assert resumed.experiment_log.schema[name] == pl.String
            assert resumed.experiment_log[name].to_list() == [value] * 6
            assert [row[name] for row in rows] == [value] * 6
            assert [row['round_params'][name] for row in records(parallel)] == [value] * 6
        assert resumed.experiment_log['none_context'].to_list() == [None] * 6
        assert resumed.experiment_log.schema['numeric_tag'] == pl.Int64
        assert resumed.experiment_log['numeric_tag'].to_list() == [2] * 6
        assert [row['round_params']['numeric_tag'] for row in records(parallel)] == [2] * 6
        assert [row['round_params']['none_context'] for row in records(parallel)] == [None] * 6
        assert resumed.experiment_log['strict_mode_error'].null_count() == 6
        assert_runs_equal(parallel, serial)


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

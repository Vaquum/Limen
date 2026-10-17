import copy
import gzip
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner
from polars.testing import assert_frame_equal
from ruamel.yaml import YAML
from threadpoolctl import threadpool_limits

from limen.cli.main import cli
from limen.cohort import Cohort
from limen.experiment import UniversalExperimentLoop
from limen.experiment import _round_storage as storage
from limen.experiment._recording import parse_recording
from limen.experiment.acceptance_report import acceptance_report
from limen.experiment.errors import StrictModeError
from limen.inference import Trainer
from limen.yaml import CompiledSFD, build_search_strategy, validate
from limen.yaml.compiler import build_pruning_strategies
from limen.yaml.store import canonical_manifest_id
from tests.test_experiment_objective import objective_config
from tests.test_factorized_preparation import _config as factorized_config
from tests.test_record_model_outputs import _config as binary_config
from tests.test_round_storage import _writer_record

FIXTURE = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'
FULL = dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True)
LEAN = dict.fromkeys(FULL, False)
_DEFAULT_REFERENCE = object()


def _config(kind='serial'):
    if kind == 'factorized':
        config = factorized_config()
    elif kind == 'walk_forward':
        config = binary_config()
        manifest = config['sfd']['manifest']
        manifest.pop('split_dates')
        manifest['split_walk_forward'] = {
            'n_folds': 2, 'test_bars': 48, 'purge_bars': 4,
            'embargo_bars': 3, 'anchored': True,
        }
        manifest['acceptance'] = {'min_deflated_sharpe_probability': 0.95}
    else:
        config = objective_config()
        config['sfd']['manifest']['split_dates'] = {
            'train_start': '2024-01-01', 'train_end': '2024-01-04',
            'val_start': '2024-01-05', 'val_end': '2024-01-06',
            'test_start': '2024-01-07', 'test_end': '2024-01-08',
        }
        config['sfd']['manifest']['calibration']['threshold_function']['params'].update(
            threshold_min=0.5, threshold_max=0.5, threshold_step=0.1,
        )
        config['sfd']['params']['C'] = [0.1, 1.0, 10.0, 30.0]
        config['uel']['n_permutations'] = 4
        config['uel']['pruning_strategies'] = [{
            'type': 'budget', 'params': {
                'max_permutations': 3, 'trim_strategy': 'worst_first',
                'check_after_pct': 0.0,
            },
        }]
    manifest = config['sfd']['manifest']
    manifest['data_source'] = {
        'method': 'limen.data.HistoricalData.get_any_file',
        'params': {'file_path_or_url': str(FIXTURE)},
    }
    if 'split_dates' in manifest:
        manifest['split_dates']['test_predict_guard'] = False
    if kind != 'factorized':
        config['sfd']['params'].update(n_jobs=[1], random_state=[42])
    config['uel'].update(prep_each_round=True, feedback_interval=1, checkpoint_interval=1)
    return config


def _bars(kind='serial'):
    count = 384 if kind == 'walk_forward' else 192 if kind == 'factorized' else 240
    return pl.read_parquet(FIXTURE, n_rows=count).with_columns(pl.lit(3600).alias('base_interval'))


def _loop(config, path, *, bars=None, search=True, reference=_DEFAULT_REFERENCE):
    return UniversalExperimentLoop(
        sfd=CompiledSFD(config), data=_bars() if bars is None else bars,
        experiment_dir=path, yaml_reference=config if reference is _DEFAULT_REFERENCE else reference,
        search_strategy=build_search_strategy(config) if search else None,
        pruning_strategies=build_pruning_strategies(config),
        checkpoint_interval=1, feedback_interval=1,
    )


def _run(loop, *, kind='serial', **kwargs):
    config = loop._compiled_yaml if loop._yaml_reference is None else loop._yaml_reference
    options = {
        'n_permutations': config['uel']['n_permutations'], 'prep_each_round': True,
        'progress_bar': False, 'record_execution': True, 'record_model_outputs': True,
        'factorize': kind == 'factorized', 'workers': 2 if kind == 'parallel' else 1,
        'post_processing': kind != 'parallel',
    }
    options.update(kwargs)
    with threadpool_limits(limits=1):
        loop.run('recording_policy', **options)


def _records(path):
    return [json.loads(line) for line in storage.text(storage.round_path(path)).splitlines()]


def _snapshot(path):
    return {str(item.relative_to(path)): (item.read_bytes(), item.stat().st_mtime_ns)
            for item in path.rglob('*') if item.is_file()}


def _project(record, policy):
    expected = copy.deepcopy(record)
    if 'folds' in expected:
        if not policy['fold_details']:
            del expected['folds']
        else:
            expected['folds'] = [_project(fold, policy) for fold in expected['folds']]
    if not policy['predictions']:
        expected.pop('preds', None)
    if not policy['trade_ledger']:
        expected.pop('trade_ledger', None)
    if not policy['alignment'] and 'alignment' in expected:
        expected['alignment'].pop('missing_datetimes', None)
        expected['alignment'].pop('test_datetimes', None)
    return expected


@pytest.fixture(scope='module')
def campaigns(tmp_path_factory):
    result = {}
    for kind in ('serial', 'factorized', 'parallel', 'walk_forward'):
        config, bars = _config(kind), _bars(kind)
        pairs = []
        for label, options in (('full', {}), ('lean', {'recording': LEAN})):
            path = tmp_path_factory.mktemp(f'recording-{kind}-{label}')
            loop = _loop(config, path, bars=bars)
            _run(loop, kind=kind, **options)
            pairs.append({'loop': loop, 'path': path, 'records': _records(path)})
        result[kind] = {'config': config, 'bars': bars, 'full': pairs[0], 'lean': pairs[1]}
    return result


@pytest.mark.parametrize('case', ('yaml', 'native', 'precedence', 'admission', 'pbo_admission', 'compiled_sources'))
def test_policy_validation(case, tmp_path, monkeypatch):
    config = _config('walk_forward')
    if case in ('yaml', 'native'):
        invalid = [False, True, [], 'predictions', {'unknown': True},
                   {'predictions': None}, {'alignment': 0}, {'trade_ledger': 1},
                   {'fold_details': 'false'}, {'predictions': '{record_predictions}'}]
        if case == 'yaml':
            invalid.append(None)
        for number, policy in enumerate(invalid):
            candidate = copy.deepcopy(config)
            path = tmp_path / str(number)
            if case == 'yaml':
                candidate['uel']['recording'] = policy
                report = validate(candidate)
                assert not report.valid
                assert any('recording' in error.path for error in report.errors)
            else:
                loop = _loop(candidate, path)
                with pytest.raises((TypeError, ValueError), match='recording'):
                    _run(loop, recording=policy)
            assert not path.exists()
        for policy in ({}, FULL, {'predictions': False}):
            candidate = copy.deepcopy(config)
            candidate['uel']['recording'] = policy
            assert validate(candidate).valid
    elif case == 'admission':
        for search, directory in ((False, tmp_path / 'no-search'), (True, None)):
            loop = _loop(config, directory, search=search)
            with pytest.raises((TypeError, ValueError), match='recording'):
                _run(loop, recording=FULL)
            if directory is not None:
                assert not directory.exists()
    elif case == 'pbo_admission':
        config['sfd']['manifest']['acceptance'] = {'max_pbo': 0.9}
        for key in ('alignment', 'fold_details'):
            candidate = copy.deepcopy(config)
            candidate['uel']['recording'] = {key: False}
            report = validate(candidate)
            assert not report.valid
            assert any('PBO' in error.message or 'max_pbo' in error.message for error in report.errors)
            path = tmp_path / (key + '-native')
            loop = _loop(config, path, bars=_bars('walk_forward'))
            with pytest.raises(ValueError, match=r'PBO|max_pbo'):
                _run(loop, kind='walk_forward', recording={key: False})
            assert not path.exists()
            candidate['uel']['output_path'] = str(tmp_path / (key + '-cli'))
            yaml_path = tmp_path / (key + '.yaml')
            YAML().dump(candidate, yaml_path)
            result = CliRunner().invoke(cli, ['run', '--no-progress-bar', str(yaml_path)])
            assert result.exit_code != 0
            assert 'PBO' in result.output or 'max_pbo' in result.output
            assert not Path(candidate['uel']['output_path']).exists()
        for key in ('predictions', 'trade_ledger'):
            candidate = copy.deepcopy(config)
            candidate['uel']['recording'] = {key: False}
            assert validate(candidate).valid
            path = tmp_path / (key + '-allowed')
            loop = _loop(candidate, path, bars=_bars('walk_forward'))
            _run(loop, kind='walk_forward')
            report = json.loads((path / 'acceptance_report.json').read_text())
            assert isinstance(report['pbo'], float)
            assert isinstance(report['verdicts']['max_pbo'], bool)
    elif case == 'compiled_sources':
        class _PolicyResolvedError(RuntimeError):
            pass

        def resolved_before_dispatch(*args, **kwargs):
            raise _PolicyResolvedError('recording admission completed')

        monkeypatch.setattr('limen.experiment._parallel_rounds.begin_run', resolved_before_dispatch)
        config['uel']['recording'] = {'predictions': False}
        unchanged = copy.deepcopy(config)
        no_policy = copy.deepcopy(config)
        del no_policy['uel']['recording']
        assert canonical_manifest_id(config) != canonical_manifest_id(no_policy)
        bars = _bars('walk_forward')
        for workers in (1, 2):
            for label, declaration, expected in (
                ('yaml', None, {**FULL, 'predictions': False}),
                ('partial', {'trade_ledger': False}, {**FULL, 'trade_ledger': False}),
                ('empty', {}, FULL),
            ):
                path = tmp_path / f'{workers}-{label}'
                loop = _loop(config, path, bars=bars, reference=None)
                options = {} if declaration is None else {'recording': declaration}
                with pytest.raises(_PolicyResolvedError, match='admission completed'):
                    _run(loop, workers=workers, post_processing=False, **options)
                assert loop._recording == expected
                assert not path.exists()
            for label, declaration in (('omitted', None), ('native', FULL)):
                conflicting = copy.deepcopy(config)
                conflicting['uel']['recording'] = {'alignment': False}
                path = tmp_path / f'{workers}-conflict-{label}'
                loop = _loop(config, path, bars=bars, reference=conflicting)
                options = {} if declaration is None else {'recording': declaration}
                with pytest.raises(ValueError, match='recording'):
                    _run(loop, workers=workers, post_processing=False, **options)
                assert not path.exists()
        assert config == unchanged
    else:
        config['uel']['recording'] = {'predictions': False}
        for label, options in (('yaml', {}), ('native', {'recording': FULL})):
            path = tmp_path / label
            loop = _loop(config, path, bars=_bars('walk_forward'),
                         reference=None if label == 'native' else config)
            _run(loop, kind='walk_forward', n_permutations=2, **options)
            records = _records(path)
            assert all(('preds' in fold) == (label == 'native')
                       for record in records for fold in record['folds'])
            metadata = json.loads((path / 'metadata.json').read_text())
            assert metadata['recording'] == (FULL if label == 'native' else {**FULL, 'predictions': False})
            assert metadata['yaml_reference']['uel']['recording'] == {'predictions': False}
            assert metadata['manifest_id'] == canonical_manifest_id(config)
            if label == 'native':
                resumed = _loop(config, path, bars=_bars('walk_forward'), reference=None)
                _run(resumed, kind='walk_forward', n_permutations=2, resume=True)
                assert _records(path) == records
                assert resumed._recording == FULL


@pytest.mark.parametrize('kind', ('serial', 'factorized', 'parallel', 'walk_forward'))
def test_payload_projection(kind, campaigns, tmp_path):
    captured = campaigns[kind]
    full, lean = captured['full'], captured['lean']
    assert lean['records'] == [_project(record, LEAN) for record in full['records']]
    assert all(set(record) >= {'round_id', '_round_index', 'round_params'} for record in lean['records'])
    if kind == 'serial':
        assert any(record['trade_ledger']['states'] for record in full['records'])
        for original, projected in zip(full['records'], lean['records'], strict=True):
            for key in ('trade_contract', 'trade_contract_digest', 'learning_binding'):
                assert projected.get(key) == original.get(key)
        raw_full = sum(len(json.dumps(record)) for record in full['records'])
        raw_lean = sum(len(json.dumps(record)) for record in lean['records'])
        assert raw_lean < raw_full
        assert storage.round_path(lean['path']).stat().st_size < storage.round_path(full['path']).stat().st_size
    if kind != 'parallel':
        for left, right in zip(full['loop'].preds, lean['loop'].preds, strict=True):
            np.testing.assert_array_equal(left, right)
        assert len(full['loop']._alignment) == len(lean['loop']._alignment)
        for left, right in zip(full['loop']._alignment, lean['loop']._alignment, strict=True):
            assert set(left) == set(right)
            if kind != 'walk_forward':
                assert right['missing_datetimes'] == left['missing_datetimes']
            if kind == 'serial':
                assert right['trade_ledger'] == left['trade_ledger']
            elif kind == 'walk_forward':
                assert all('preds' in fold and 'net_returns' in fold for fold in right['folds'])
    assert 'recording' not in json.loads((full['path'] / 'metadata.json').read_text())
    assert json.loads((lean['path'] / 'metadata.json').read_text())['recording'] == LEAN
    expected_identity = canonical_manifest_id(captured['config'])
    for saved in (full, lean):
        metadata = json.loads((saved['path'] / 'metadata.json').read_text())
        assert metadata['manifest_id'] == expected_identity
        assert metadata['yaml_reference'] == captured['config']
    original = copy.deepcopy(full['records'])
    loop = full['loop']
    previous = loop._recording
    try:
        for number, values in enumerate(itertools.product((False, True), repeat=len(FULL))):
            policy = dict(zip(FULL, values, strict=True))
            directory = tmp_path / str(number)
            directory.mkdir()
            (directory / 'metadata.json').write_text(json.dumps({'round_storage': storage.FORMAT}))
            path = storage.round_path(directory)
            loop._recording = parse_recording(policy)
            for record in full['records']:
                source = copy.deepcopy(record)
                if 'learning_binding' in source:
                    source['learning_binding'] = source['learning_binding']['model']
                before_write = copy.deepcopy(source)
                _writer_record(loop, path, source)
                assert source == before_write
            assert _records(directory) == [_project(record, policy) for record in original]
    finally:
        loop._recording = previous
    assert full['records'] == original
    if kind == 'factorized':
        damaged = _project(original[0], {**FULL, 'predictions': False})
        damaged['market']['ret'].pop()
        path = tmp_path / 'damaged.jsonl.gz'
        with pytest.raises(storage.RoundStorageError, match='length'):
            storage.append_record(path, damaged)
        assert not path.exists()
        for damage in ('missing_execution', 'execution_type', 'missing_pos', 'pos_type', 'pos_length'):
            valid = _project(original[0], {**FULL, 'predictions': False})
            directory = tmp_path / damage
            directory.mkdir()
            (directory / 'metadata.json').write_text(json.dumps({'round_storage': storage.FORMAT}))
            path = storage.round_path(directory)
            storage.append_record(path, valid)
            wire = json.loads(gzip.decompress(path.read_bytes()))
            for record in (valid, wire):
                if damage == 'missing_execution':
                    record.pop('execution')
                elif damage == 'execution_type':
                    record['execution'] = None
                elif damage == 'missing_pos':
                    record['execution'].pop('pos')
                elif damage == 'pos_type':
                    record['execution']['pos'] = None
                else:
                    record['execution']['pos'].pop()
            output = directory / 'rejected.jsonl.gz'
            before = _snapshot(directory)
            with pytest.raises(storage.RoundStorageError, match=r'Market|alignment'):
                storage.append_record(output, valid)
            assert _snapshot(directory) == before
            path.write_bytes(gzip.compress((json.dumps(wire) + '\n').encode('utf-8'), mtime=0))
            with pytest.raises(storage.RoundStorageError, match=r'Market|alignment'):
                _records(directory)


@pytest.mark.parametrize('kind', ('serial', 'factorized', 'parallel', 'walk_forward'))
def test_scientific_equivalence(kind, campaigns):
    captured = campaigns[kind]
    full, lean = captured['full'], captured['lean']
    left, right = (item['loop'].experiment_log.drop('execution_time') for item in (full, lean))
    assert_frame_equal(left, right, check_exact=False, rel_tol=1e-10, abs_tol=1e-12)
    stored = [pl.read_csv(item['path'] / 'results.csv').drop('execution_time') for item in (full, lean)]
    assert_frame_equal(*stored, check_exact=False, rel_tol=1e-10, abs_tol=1e-12)
    for item in (full, lean):
        rows = item['loop'].experiment_log
        assert rows['id'].to_list() == [record['round_id'] for record in item['records']]
        assert rows['strict_mode_error'].null_count() == rows.height
    audits = []
    for item in (full, lean):
        path = item['path'] / 'audit.jsonl'
        entries = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        for entry in entries:
            del entry['timestamp']
        audits.append(entries)
    assert audits[0] == audits[1]
    if kind in ('serial', 'parallel'):
        assert 'val_backtest_total_return' in left.columns
        assert any(entry['interventions'] for entry in audits[0])
    if kind == 'walk_forward':
        assert_frame_equal(full['loop'].fold_results, lean['loop'].fold_results)
        assert_frame_equal(*[pl.read_parquet(item['path'] / 'trial_returns.parquet') for item in (full, lean)])
        reports = [json.loads((item['path'] / 'acceptance_report.json').read_text()) for item in (full, lean)]
        expected, projected = reports
        assert isinstance(expected['pbo'], float)
        assert projected['pbo'] is None
        assert projected['errors']['pbo']
        for key in expected:
            if key not in ('pbo', 'errors'):
                assert projected[key] == expected[key]
        assert {key: value for key, value in projected['errors'].items() if key != 'pbo'} == expected['errors']
        assert projected['thresholds'] == {'min_deflated_sharpe_probability': 0.95}
        assert 'pbo unavailable' in (lean['path'] / 'acceptance_report.md').read_text()


@pytest.mark.parametrize('case', ('trainer', 'resume_rejection', 'default_resume', 'cli', 'strict_failure', 'pbo_report'))
def test_consumer_contracts(case, campaigns, tmp_path):
    captured = campaigns['serial']
    if case == 'trainer':
        predictions = []
        chosen = []
        for label in ('full', 'lean'):
            path = captured[label]['path']
            row = captured[label]['records'][0]
            sensor = Trainer(path, data=captured['bars']).train([row['round_id']])[0]
            predictions.append([item.prediction for item in sensor.predict_all(captured['bars'])])
            all_members = Cohort(experiment_log_path=str(path), selector='all')
            assert set(all_members.permutation_ids) == set(captured[label]['loop'].experiment_log['id'])
            cohort = Cohort(experiment_log_path=str(path), selector='top_n',
                            selector_params={'column': 'val_backtest_total_return', 'n': 1})
            chosen.append(cohort.permutation_ids)
        assert predictions[0] == predictions[1]
        assert chosen[0] == chosen[1]
    elif case == 'resume_rejection':
        policies = [None, FULL, LEAN, *({key: False} for key in FULL)]
        for kind, captured in campaigns.items():
            for saved in ('full', 'lean'):
                for index, policy in enumerate(policies):
                    if saved == 'full' and policy in (None, FULL):
                        continue
                    path = tmp_path / f'{kind}-{saved}-{index}'
                    shutil.copytree(captured[saved]['path'], path)
                    before = _snapshot(path)
                    loop = _loop(captured['config'], path, bars=captured['bars'])
                    options = {} if policy is None else {'recording': policy}
                    with pytest.raises(ValueError, match='recording'):
                        _run(loop, kind=kind, resume=True, **options)
                    assert _snapshot(path) == before
            for label, options in (('omitted', {}), ('override', {'recording': FULL})):
                path = tmp_path / f'{kind}-changed-yaml-{label}'
                shutil.copytree(captured['full']['path'], path)
                candidate = copy.deepcopy(captured['config'])
                candidate['uel']['recording'] = {'alignment': False}
                loop = _loop(candidate, path, bars=captured['bars'])
                before = _snapshot(path)
                with pytest.raises(ValueError, match='recording'):
                    _run(loop, kind=kind, resume=True, **options)
                assert _snapshot(path) == before
    elif case == 'default_resume':
        path = tmp_path / 'default'
        shutil.copytree(captured['full']['path'], path)
        records = _records(path)
        persisted = pl.read_csv(path / 'results.csv', schema_overrides={'_convergence_warning': pl.Boolean})
        loop = _loop(captured['config'], path, bars=captured['bars'])
        _run(loop, resume=True)
        assert _records(path) == records
        assert_frame_equal(loop.experiment_log, persisted)
    elif case == 'cli':
        config = _config('walk_forward')
        config['sfd']['manifest']['data_source']['params']['row_count_limit'] = 384
        config['uel'].update(recording=LEAN, output_path=str(tmp_path / 'cli-run'),
                             output_format='parquet', record_execution=True, record_model_outputs=True)
        yaml_path = tmp_path / 'recording.yaml'
        YAML().dump(config, yaml_path)
        runner = CliRunner()
        with threadpool_limits(limits=1):
            result = runner.invoke(cli, ['run', '--no-progress-bar', str(yaml_path)])
        assert result.exit_code == 0, result.output
        path = tmp_path / 'cli-run'
        assert_frame_equal(pl.read_csv(path / 'results.csv'), pl.read_parquet(path / 'results.parquet'),
                           check_dtypes=False)
        assert all('folds' not in record for record in _records(path))
        before = _snapshot(path)
        rejected = runner.invoke(cli, ['run', '--no-progress-bar', '--resume', str(path)])
        assert rejected.exit_code != 0
        assert 'recording' in rejected.output
        assert _snapshot(path) == before
    elif case == 'pbo_report':
        path = tmp_path / 'selected-pbo'
        shutil.copytree(campaigns['walk_forward']['lean']['path'], path)
        before = _snapshot(path)
        with pytest.raises(ValueError, match=r'PBO|max_pbo'):
            acceptance_report(path, acceptance={'max_pbo': 0.9})
        assert _snapshot(path) == before
    else:
        config = _config()
        logs = []
        for label, declaration in (('full', None), ('lean', LEAN)):
            policy = None if declaration is None else dict(declaration)
            loop = _loop(config, tmp_path / label)
            original = loop.model

            def failed_first(data, round_params, _original=original, _policy=policy):
                if round_params['C'] == 0.1:
                    if _policy is not None:
                        _policy['predictions'] = True
                    raise StrictModeError('recorded controlled failure')
                return _original(data, round_params)

            loop.model = failed_first
            options = {} if policy is None else {'recording': policy}
            _run(loop, n_permutations=2, **options)
            logs.append(loop.experiment_log.drop('execution_time'))
            records = _records(tmp_path / label)
            assert len(records) == 1 and records[0]['_round_index'] == 1
            assert ('preds' in records[0]) == (label == 'full')
            assert loop.experiment_log['strict_mode_error'].null_count() == 1
        assert_frame_equal(*logs)

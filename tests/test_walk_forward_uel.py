import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest

from limen.backtest.backtest_snapshot import BACKTEST_SNAPSHOT_COLUMNS
from limen.backtest.execution_events import with_predictions
from limen.backtest.long_flat_strategy import long_flat_strategy
from limen.backtest.trade_execution import trade_execution
from limen.calibration import sklearn_probability_calibrator
from limen.data.utils import split_walk_forward
from limen.experiment import Manifest, RuleBasedManifest, UniversalExperimentLoop
from limen.experiment.errors import StrictModeError
from limen.experiment.reducer import BudgetReducer
from limen.sfd.reference_architecture import LogRegBinary
from limen.yaml import CompiledSFD, build_search_strategy, parse, validate
from tests.test_experiment_objective import objective_config

_FIXTURES = Path(__file__).parent / 'fixtures'
_GEOMETRY = {'n_folds': 2, 'test_bars': 48, 'purge_bars': 4,
             'embargo_bars': 3, 'anchored': True}


_MANIFEST_YAML = '''
schema_version: "1.0"

metadata:
  name: recorded_walk_forward
  limen_version: "5.21.0"
  mode: development
  description: Two recorded hourly folds with independent execution

sfd:
  manifest:
    type: rule_based
    data_source:
      method: limen.data.HistoricalData.get_spot_klines
      params:
        kline_size: 3600
    split_walk_forward:
      n_folds: 2
      test_bars: 48
      purge_bars: 4
      embargo_bars: 3
      anchored: true
    indicators:
      - func: limen.indicators.window_return
        params:
          period: 1
    strategy:
      conditions:
        - id: entry
          name: positive_return
          type: threshold
          column: ret_1
          operator: ">"
          value: "{entry_return}"
      entry: entry
    backtest:
      fee_bps: 7.0
      slip_bps: 3.0
      notional_rate: 0.5
    reference_architecture: limen.sfd.reference_architecture.rule_based
  params:
    entry_return: [0.0, 0.001]

uel:
  n_permutations: 2
  search_strategy:
    type: grid
  checkpoint_interval: 1
  prep_each_round: true
  output_format: csv
'''


def _config():
    config, errors = parse(_MANIFEST_YAML)
    assert not errors
    return config


def _bars():
    return pl.read_parquet(_FIXTURES / 'spot_1h_20240101_20241231.parquet').head(384)


def _loop(config, bars, path, *, search=True, callback=None, ratios=None, reducers=()):
    loop = UniversalExperimentLoop(
        sfd=CompiledSFD(config), data=bars, experiment_dir=path,
        search_strategy=build_search_strategy(config) if search else None,
        yaml_reference=config, checkpoint_interval=1, feedback_interval=1,
        intra_callback=callback, pruning_strategies=list(reducers),
    )
    if ratios is not None:
        loop.manifest.set_split_config(*ratios)
    return loop


def _run(loop, *, resume=False, n_permutations=2):
    loop.run('walk_forward', n_permutations=n_permutations, prep_each_round=True,
             random_search=False, progress_bar=False, post_processing=True,
             record_execution=loop._search_strategy is not None, resume=resume)


def _fold_manifest(manifest, fold):
    clone = copy.deepcopy(manifest)
    clone._walk_forward_fold = fold
    return clone


def _records(path):
    return [json.loads(line) for line in (path / 'round_data.jsonl').read_text().splitlines()]


@pytest.fixture(scope='module')
def fold_runs(tmp_path_factory):
    bars = _bars()
    results = []
    for anchored in (True, False):
        for search in (True, False):
            config = _config()
            config['sfd']['manifest']['split_walk_forward']['anchored'] = anchored
            path = tmp_path_factory.mktemp(f'folds_{anchored}_{search}')
            feedback = []
            loop = _loop(config, bars, path, search=search,
                         callback=lambda log, msq, feedback=feedback: feedback.append(log.clone()))
            _run(loop)
            results.append((loop, config, bars, path, search, feedback))
    return results


def test_manifest_block_validation():
    config = _config()
    assert validate(config).valid
    compiled = CompiledSFD(config).manifest()
    native = Manifest().set_split_walk_forward(**_GEOMETRY)
    assert compiled.split_config == (8, 1, 2)
    assert compiled.split_walk_forward == native.split_walk_forward
    assert compiled.split_walk_forward.as_dict() == _GEOMETRY
    conflicting = copy.deepcopy(config)
    conflicting['sfd']['manifest']['split_dates'] = {
        'train_start': '2024-01-01', 'train_end': '2024-01-03',
        'val_start': '2024-01-03', 'val_end': '2024-01-04',
        'test_start': '2024-01-04', 'test_end': '2024-01-05',
    }
    assert not validate(conflicting).valid
    with pytest.raises(ValueError, match='split_walk_forward'):
        CompiledSFD(conflicting).manifest()


@pytest.mark.parametrize(('field', 'value'), [
    ('n_folds', 1), ('n_folds', True), ('test_bars', 0),
    ('purge_bars', -1), ('embargo_bars', -1), ('anchored', 1),
    ('n_folds', '{folds}'), ('test_bars', 2.0),
])
def test_geometry_requires_literal_values(field, value):
    geometry = {**_GEOMETRY, field: value}
    config = _config()
    config['sfd']['manifest']['split_walk_forward'] = geometry
    assert not validate(config).valid
    with pytest.raises(ValueError, match='split_walk_forward'):
        Manifest().set_split_walk_forward(**geometry)
    with pytest.raises(ValueError, match='split_walk_forward'):
        CompiledSFD(config).manifest()


@pytest.mark.parametrize('declaration', [None, {}, {'n_folds': 2}, {**_GEOMETRY, 'extra': True}])
def test_geometry_requires_complete_known_fields(declaration):
    config = _config()
    config['sfd']['manifest']['split_walk_forward'] = declaration
    assert not validate(config).valid
    with pytest.raises(ValueError, match='split_walk_forward'):
        CompiledSFD(config).manifest()


def test_fold_loop_ledger_rows(fold_runs):
    for loop, config, bars, _, _, feedback in fold_runs:
        assert loop.experiment_log.height == 2
        assert loop.fold_results.height == 4
        assert loop.fold_results['fold'].to_list() == [0, 1, 0, 1]
        assert loop.manifest._walk_forward_fold is None
        assert loop.manifest.split_walk_forward.as_dict() == config['sfd']['manifest']['split_walk_forward']
        ledger = loop.experiment_backtest_results
        assert len(ledger) == 4 and set(ledger['fold']) == {0, 1}
        assert set(BACKTEST_SNAPSHOT_COLUMNS).issubset(ledger.columns)
        for trial in loop.experiment_log.iter_rows(named=True):
            rows = loop.fold_results.filter(pl.col('id') == trial['id'])
            assert rows.height == 2
            finite = rows['pnl_per_bar_bps_test'].drop_nulls().to_numpy()
            finite = finite[np.isfinite(finite)]
            assert trial['pnl_per_bar_bps_test'] == pytest.approx(finite.mean())
            for fold in range(2):
                manifest = _fold_manifest(loop.manifest, fold)
                prepared = manifest.prepare_data(bars, {'entry_return': trial['entry_return']})
                pool, raw_test = split_walk_forward(bars, **config['sfd']['manifest']['split_walk_forward'])[fold]
                validation_start = pool.height * 8 // 9
                raw_train = pool.head(validation_start - 4)
                raw_validation = pool.slice(validation_start)
                assert prepared['train']['close'].equals(raw_train['close'].slice(1))
                assert prepared['val']['close'].equals(raw_validation['close'].slice(1))
                alignment = prepared['_alignment']
                assert alignment['first_test_datetime'] == raw_test['datetime'][1]
                assert alignment['last_test_datetime'] == raw_test['datetime'][-1]
                assert prepared['test'].height == raw_test.height - 1
        for observed in feedback:
            assert observed.height <= 2
            assert 'fold' not in observed.columns


def test_snapshot_schema_frozen():
    assert BACKTEST_SNAPSHOT_COLUMNS == [
        'edge_bps_p5', 'edge_bps_p50', 'edge_bps_p95',
        'pnl_bps_p5', 'pnl_bps_p50', 'pnl_bps_p95',
        'cost_bps_p5', 'cost_bps_p50', 'cost_bps_p95',
        'drawdown_bps_p5', 'drawdown_bps_p50', 'drawdown_bps_p95',
        'wins_per_bar', 'pnl_per_bar_bps', 'avg_win_bps', 'avg_loss_bps',
        'cvar_95_pnl_bps', 'trades_per_bar', 'inventory_per_bar', 'cost_per_bar_bps',
    ]
    assert 'fold' not in BACKTEST_SNAPSHOT_COLUMNS


def test_trial_returns_artifact(fold_runs):
    for loop, _, bars, path, search, _ in fold_runs:
        artifact = pq.ParquetFile(path / 'trial_returns.parquet')
        assert artifact.metadata.num_row_groups == loop.experiment_log.height == 2
        tracks = pl.read_parquet(path / 'trial_returns.parquet')
        assert tracks.columns == ['trial', 'bar', 'net_return']
        for trial_index, trial in enumerate(loop.experiment_log.iter_rows(named=True)):
            expected = []
            for fold in range(2):
                manifest = _fold_manifest(loop.manifest, fold)
                prepared = manifest.prepare_data(bars, {'entry_return': trial['entry_return']})
                result = manifest.run_model(prepared, {'entry_return': trial['entry_return']})
                prices = prepared['test']
                execution = long_flat_strategy(
                    result['_preds'], prices['open'].to_numpy(), prices['close'].to_numpy(),
                    (prices['close'] - prices['open']).to_numpy(), fee_bps=7.0, slip_bps=3.0,
                )
                assert execution.net[0] == 0.0
                expected.extend(execution.net * 0.5)
            group = pl.from_arrow(artifact.read_row_group(trial_index))
            assert group['trial'].n_unique() == 1
            assert str(group['trial'][0]) == str(trial['id'])
            assert group['bar'].to_list() == list(range(len(expected)))
            np.testing.assert_array_equal(group['net_return'].to_numpy(), expected)
            if search:
                record = _records(path)[trial_index]
                assert record['round_id'] == trial['id']
                assert len(record['folds']) == 2
                for fold, nested in enumerate(record['folds']):
                    assert nested['fold'] == fold
                    assert len(nested['preds']) == len(nested['execution']['net'])
                    np.testing.assert_array_equal(nested['net_returns'], nested['execution']['net'])
        assert tracks['net_return'].is_finite().all()


def test_default_path_unchanged(tmp_path):
    config = _config()
    manifest_config = config['sfd']['manifest']
    manifest_config.pop('split_walk_forward')
    manifest_config['split_dates'] = {
        'train_start': '2024-01-01', 'train_end': '2024-01-08',
        'val_start': '2024-01-08', 'val_end': '2024-01-11',
        'test_start': '2024-01-11', 'test_end': '2024-01-16',
    }
    bars = _bars()
    loop = _loop(config, bars, tmp_path)
    _run(loop)
    assert loop.fold_results.is_empty()
    assert not (tmp_path / 'trial_returns.parquet').exists()
    assert not (tmp_path / 'fold_results.csv').exists()
    assert 'split_walk_forward' not in json.loads((tmp_path / 'metadata.json').read_text())
    for trial, record in zip(loop.experiment_log.iter_rows(named=True), _records(tmp_path), strict=True):
        assert 'folds' not in record
        prepared = loop.manifest.prepare_data(bars, {'entry_return': trial['entry_return']})
        direct = loop.manifest.run_model(prepared, {'entry_return': trial['entry_return']})
        np.testing.assert_array_equal(record['preds'], direct['_preds'])
        assert trial['pnl_per_bar_bps_test'] == direct['pnl_per_bar_bps_test']


def test_fold_local_fits_calibration_and_objective(tmp_path, monkeypatch):
    config = objective_config('minimize')
    manifest_config = config['sfd']['manifest']
    manifest_config.pop('split_dates')
    manifest_config['split_walk_forward'] = _GEOMETRY
    config['sfd']['params']['C'] = [0.1]
    bars = _bars()
    loop = _loop(config, bars, tmp_path, ratios=(4, 1, 0))
    trained = []
    calibrated = []
    original = LogRegBinary.train

    def capture_train(model, prepared, **params):
        trained.append((model, prepared))
        return original(model, prepared, **params)

    def capture_calibration(model, x_val, y_val, **params):
        fitted = sklearn_probability_calibrator(model, x_val, y_val, **params)
        calibrated.append((model, x_val, y_val, fitted))
        return fitted

    monkeypatch.setattr(LogRegBinary, 'train', capture_train)
    loop.manifest.prediction_calibration_config.calibration_func = capture_calibration
    loop.run('objective_folds', n_permutations=1, prep_each_round=True,
             progress_bar=False, record_execution=True, record_model_outputs=True)
    assert len(trained) == len(calibrated) == 2
    assert trained[0][0] is not trained[1][0]
    assert calibrated[0][3] is not calibrated[1][3]
    assert trained[0][1]['_scaler'] is not trained[1][1]['_scaler']
    record = _records(tmp_path)[0]
    for fold, ((model, prepared), calibration, nested) in enumerate(zip(trained, calibrated, record['folds'], strict=True)):
        pool, test = split_walk_forward(bars, **_GEOMETRY)[fold]
        boundary = pool.height * 4 // 5
        raw_fit, raw_val = pool.head(boundary - 4), pool.slice(boundary)
        context = prepared['_trade_context']
        for rows, raw in zip(context.model_rows, (raw_fit, raw_val, test), strict=True):
            assert rows['datetime'].is_in(raw['datetime'].implode()).all()
        assert calibration[0] is model.model
        assert calibration[1].equals(prepared['x_val'])
        assert calibration[2].equals(prepared['y_val'])
        probabilities = model.predict({'x_test': prepared['x_test']})['_probs']
        np.testing.assert_array_equal(nested['probs'], probabilities)
        np.testing.assert_array_equal(nested['preds'], probabilities >= nested['optimal_threshold'])
        expected = trade_execution(with_predictions(context.partitions[2], nested['preds']), context.policy)
        assert nested['trade_ledger']['states'] == expected.states.to_dicts()
        row = loop.fold_results.filter(pl.col('fold') == fold).row(0, named=True)
        for key, value in expected.metrics.items():
            assert row[f'backtest_{key}'] == value
        endpoints = context.partitions[2].signals['available_at_ns'].to_list()
        equity = [float(expected.states.filter(pl.col('time_ns') <= endpoint)['equity'][-1])
                  for endpoint in endpoints]
        equity[-1] = float(expected.states['equity'][-1])
        previous = np.asarray([context.partitions[2].initial_equity, *equity[:-1]])
        np.testing.assert_array_equal(nested['net_returns'], np.asarray(equity) / previous - 1)
        selected = model.predict({'x_test': prepared['x_val']})['_preds']
        from limen.experiment._objective import prepare_objective
        objective = prepare_objective(prepared, loop.manifest.architecture_function,
                                      loop.manifest.prediction_calibration_config)
        expected_validation = trade_execution(with_predictions(objective.inputs, selected), context.policy)
        row = loop.fold_results.filter(pl.col('fold') == fold).row(0, named=True)
        assert row['val_backtest_total_return'] == expected_validation.metrics['total_return']
        assert row['val_score'] == row['val_backtest_total_return']
    assert loop.experiment_log['val_backtest_total_return'][0] == pytest.approx(loop.fold_results['val_backtest_total_return'].mean())
    original_selection = loop.fold_results.select('fold', 'optimal_threshold', 'val_score', 'val_backtest_total_return')
    changed = pl.concat([bars.head(336), bars.tail(48).select(
        pl.col('datetime'), pl.exclude('datetime').reverse(),
    )])
    perturbed = _loop(config, changed, tmp_path / 'held_out', ratios=(4, 1, 0))
    perturbed.run('objective_folds', n_permutations=1, prep_each_round=True,
                  progress_bar=False, record_execution=True, record_model_outputs=True)
    assert perturbed.fold_results.select(original_selection.columns).equals(original_selection)
    for before, after in zip(trained[:2], trained[2:], strict=True):
        for name in ('x_train', 'y_train', 'x_val', 'y_val'):
            assert before[1][name].equals(after[1][name])


def test_resume_rejects_changed_geometry_and_retains_tracks(tmp_path, monkeypatch):
    config = _config()
    config['sfd']['params'].update(entry_return=[0.0, 0.001, 0.002], fee=[7.0, 9.0])
    config['sfd']['manifest']['backtest']['fee_bps'] = '{fee}'
    bars = _bars()

    def make_loop(path):
        return _loop(config, bars, path, reducers=[BudgetReducer(
            max_permutations=5, trim_strategy='worst_first',
            metric='pnl_per_bar_bps_test', check_after_pct=0.0,
        )])

    full = make_loop(tmp_path / 'full')
    _run(full, n_permutations=6)
    first = make_loop(tmp_path / 'resume')
    original = RuleBasedManifest.run_model
    calls = 0

    def interrupted(manifest, data, round_params):
        nonlocal calls
        result = original(manifest, data, round_params)
        calls += 1
        if calls == 2:
            first._shutdown_requested = True
        return result

    with monkeypatch.context() as context:
        context.setattr(RuleBasedManifest, 'run_model', interrupted)
        _run(first, n_permutations=6)
    assert first.experiment_log.height == 1
    path = tmp_path / 'resume'
    before = {file.name: file.read_bytes() for file in path.iterdir() if file.is_file()}
    changed = copy.deepcopy(config)
    changed['sfd']['manifest']['split_walk_forward']['embargo_bars'] += 1
    with pytest.raises(ValueError, match='split_walk_forward'):
        _run(_loop(changed, bars, path), resume=True, n_permutations=6)
    assert all((path / name).read_bytes() == value for name, value in before.items())
    resumed = make_loop(path)
    _run(resumed, resume=True, n_permutations=6)
    assert resumed.experiment_log.drop('execution_time').equals(full.experiment_log.drop('execution_time'))
    assert resumed.fold_results.equals(full.fold_results)
    assert 2 < resumed.experiment_log.height < 6
    assert pq.ParquetFile(path / 'trial_returns.parquet').metadata.num_row_groups == resumed.experiment_log.height
    assert pl.read_parquet(path / 'trial_returns.parquet').equals(pl.read_parquet(tmp_path / 'full/trial_returns.parquet'))
    assert _records(path) == _records(tmp_path / 'full')
    assert _records(path)[0] == json.loads(before['round_data.jsonl'])
    audits = []
    for output in (path, tmp_path / 'full'):
        entries = [json.loads(line) for line in (output / 'audit.jsonl').read_text().splitlines()]
        for entry in entries:
            entry.pop('timestamp')
        audits.append(entries)
    assert audits[0] == audits[1]
    assert any(entry['interventions'] for entry in audits[0])


@pytest.mark.parametrize('stop_after_success', (False, True))
def test_resume_accounts_for_failed_trials_before_rewriting_returns(tmp_path, monkeypatch, stop_after_success):
    config = _config()
    config['sfd']['params']['entry_return'] = [0.0, 0.001, 0.002]
    bars = _bars()
    original = RuleBasedManifest.run_model

    def fail_first_trial(manifest, data, round_params):
        if round_params['entry_return'] == 0.0:
            raise StrictModeError('Recorded walk-forward trial failed')
        return original(manifest, data, round_params)

    monkeypatch.setattr(RuleBasedManifest, 'run_model', fail_first_trial)
    full = _loop(config, bars, tmp_path / 'full')
    _run(full, n_permutations=3)
    path = tmp_path / 'resume'
    first = _loop(config, bars, path)
    calls = 0

    def interrupt_trial(manifest, data, round_params):
        nonlocal calls
        if not stop_after_success and round_params['entry_return'] == 0.0:
            first._shutdown_requested = True
        result = fail_first_trial(manifest, data, round_params)
        calls += 1
        if stop_after_success and calls == 2:
            first._shutdown_requested = True
        return result

    with monkeypatch.context() as context:
        context.setattr(RuleBasedManifest, 'run_model', interrupt_trial)
        _run(first, n_permutations=3)
    assert first.experiment_log.height == (2 if stop_after_success else 1)
    assert first.experiment_log['strict_mode_error'][0] == 'Recorded walk-forward trial failed'
    if stop_after_success:
        assert first.fold_results.height == 2
        records = _records(path)
        assert len(records) == 1 and records[0]['_round_index'] == 1
        saved_jsonl = (path / 'round_data.jsonl').read_bytes()
        (path / 'round_data.jsonl').write_bytes(saved_jsonl[:len(saved_jsonl) // 2])
        before = {file.name: file.read_bytes() for file in path.iterdir() if file.is_file()}
        with pytest.raises(json.JSONDecodeError):
            _run(_loop(config, bars, path), resume=True, n_permutations=3)
        assert {file.name: file.read_bytes() for file in path.iterdir() if file.is_file()} == before
        (path / 'round_data.jsonl').write_bytes(saved_jsonl)
        records[0]['folds'].pop()
        (path / 'round_data.jsonl').write_text(json.dumps(records[0]) + '\n')
        before = {file.name: file.read_bytes() for file in path.iterdir() if file.is_file()}
        with pytest.raises(ValueError, match='split_walk_forward'):
            _run(_loop(config, bars, path), resume=True, n_permutations=3)
        assert {file.name: file.read_bytes() for file in path.iterdir() if file.is_file()} == before
        (path / 'round_data.jsonl').write_bytes(saved_jsonl)
        saved_csv = (path / 'results.csv').read_bytes()
        (path / 'results.csv').write_text(saved_csv.decode().splitlines()[0] + '\n')
        (path / 'round_data.jsonl').write_text('')
        before = {file.name: file.read_bytes() for file in path.iterdir() if file.is_file()}
        with pytest.raises(ValueError, match='split_walk_forward'):
            _run(_loop(config, bars, path), resume=True, n_permutations=3)
        assert {file.name: file.read_bytes() for file in path.iterdir() if file.is_file()} == before
        (path / 'results.csv').write_bytes(saved_csv)
        (path / 'round_data.jsonl').write_bytes(saved_jsonl)
    else:
        assert first.fold_results.is_empty()
        assert not (path / 'round_data.jsonl').exists()
        assert not (path / 'trial_returns.parquet').exists()
    next_record = (tmp_path / 'full/round_data.jsonl').read_bytes().splitlines()[int(stop_after_success)]
    with (path / 'round_data.jsonl').open('ab') as stream:
        _ = stream.write(next_record[:len(next_record) // 2])
    resumed = _loop(config, bars, path)
    _run(resumed, resume=True, n_permutations=3)
    assert set(resumed.experiment_log.columns) == set(full.experiment_log.columns)
    assert resumed.experiment_log.select(full.experiment_log.columns).drop('execution_time').equals(full.experiment_log.drop('execution_time'))
    assert resumed.fold_results.equals(full.fold_results)
    assert pl.read_csv(path / 'results.csv').drop('execution_time').equals(pl.read_csv(tmp_path / 'full/results.csv').drop('execution_time'))
    assert _records(path) == _records(tmp_path / 'full')
    assert pq.ParquetFile(path / 'trial_returns.parquet').metadata.num_row_groups == 2
    assert pl.read_parquet(path / 'trial_returns.parquet').equals(pl.read_parquet(tmp_path / 'full/trial_returns.parquet'))


@pytest.mark.parametrize('saved_round_data', (False, True))
def test_resume_clears_returns_beyond_failure_only_checkpoint(tmp_path, monkeypatch, saved_round_data):
    config = _config()
    config['sfd']['params']['entry_return'] = [0.0, 0.001, 0.002]
    bars = _bars()
    first = _loop(config, bars, tmp_path)
    original = RuleBasedManifest.run_model

    def fail_first_trial(manifest, data, round_params):
        if round_params['entry_return'] == 0.0:
            first._shutdown_requested = True
            raise StrictModeError('Recorded walk-forward trial failed')
        return original(manifest, data, round_params)

    monkeypatch.setattr(RuleBasedManifest, 'run_model', fail_first_trial)
    _run(first, n_permutations=3)
    checkpoint = (tmp_path / 'checkpoint.json').read_bytes()
    assert first.experiment_log.height == 1 and first.fold_results.is_empty()
    _run(_loop(config, bars, tmp_path), resume=True, n_permutations=3)
    assert pq.ParquetFile(tmp_path / 'trial_returns.parquet').metadata.num_row_groups == 2
    (tmp_path / 'checkpoint.json').write_bytes(checkpoint)
    if not saved_round_data:
        (tmp_path / 'round_data.jsonl').unlink()

    changed = copy.deepcopy(config)
    changed['sfd']['manifest']['split_walk_forward']['embargo_bars'] += 1
    before = {file.name: file.read_bytes() for file in tmp_path.iterdir() if file.is_file()}
    with pytest.raises(ValueError, match='split_walk_forward'):
        _run(_loop(changed, bars, tmp_path), resume=True, n_permutations=3)
    assert {file.name: file.read_bytes() for file in tmp_path.iterdir() if file.is_file()} == before

    def fail_remaining_trials(manifest, data, round_params):
        raise StrictModeError('Recorded walk-forward trial failed')

    monkeypatch.setattr(RuleBasedManifest, 'run_model', fail_remaining_trials)
    resumed = _loop(config, bars, tmp_path)
    _run(resumed, resume=True, n_permutations=3)
    assert resumed.experiment_log.height == 3 and resumed.fold_results.is_empty()
    assert resumed.experiment_log['strict_mode_error'].null_count() == 0
    assert not (tmp_path / 'trial_returns.parquet').exists()
    assert not (tmp_path / 'trial_returns.parquet.tmp').exists()
    assert not (tmp_path / 'round_data.jsonl').exists() or not _records(tmp_path)
    assert pl.read_csv(tmp_path / 'results.csv')['strict_mode_error'].null_count() == 0


@pytest.mark.parametrize('saved_walk_forward', (False, True))
def test_native_resume_rejects_added_or_removed_walk_forward(tmp_path, saved_walk_forward):
    config = _config()
    bars = _bars()

    def native_loop(enabled):
        compiled = CompiledSFD(config)
        manifest = compiled.manifest()
        if not enabled:
            manifest.split_walk_forward = None
        sfd = SimpleNamespace(__name__=__name__, params=compiled.params, manifest=lambda: manifest)
        return UniversalExperimentLoop(
            sfd=sfd, data=bars, experiment_dir=tmp_path,
            search_strategy=build_search_strategy(config), checkpoint_interval=1,
        )

    saved = native_loop(saved_walk_forward)
    assert saved._yaml_reference is None
    _run(saved, n_permutations=2)
    assert (tmp_path / 'checkpoint.json').exists()
    before = {file.name: file.read_bytes() for file in tmp_path.iterdir() if file.is_file()}
    with pytest.raises(ValueError, match='split_walk_forward'):
        _run(native_loop(not saved_walk_forward), resume=True, n_permutations=2)
    assert {file.name: file.read_bytes() for file in tmp_path.iterdir() if file.is_file()} == before


@pytest.mark.parametrize(('period', 'partition'), ((384, 'fit'), (48, 'test')))
def test_rule_based_rejects_empty_partitions_after_indicators(period, partition):
    manifest = CompiledSFD(_config()).manifest()
    manifest.feature_transforms[0].params['period'] = period
    manifest._walk_forward_fold = 0
    with pytest.raises(ValueError, match=f'split_walk_forward fold has no {partition} rows'):
        manifest.prepare_data(_bars(), {'entry_return': 0.0})


@pytest.mark.parametrize('invalid', ('unsorted', 'duplicate'))
def test_source_selector_runs_before_fold_geometry(tmp_path, invalid):
    bars = _bars()
    source = bars.reverse() if invalid == 'unsorted' else pl.concat([bars, bars.tail(1)])
    loop = _loop(_config(), source, tmp_path)
    selected = []

    def select_recorded_rows(raw):
        selected.append(raw)
        return raw.unique(subset='datetime', maintain_order=True).sort('datetime')

    loop.manifest.set_pre_split_data_selector(select_recorded_rows)
    _run(loop, n_permutations=1)
    assert len(selected) == 1 and selected[0].equals(source)
    assert loop.experiment_log.height == 1 and loop.fold_results.height == 2
    assert pq.ParquetFile(tmp_path / 'trial_returns.parquet').metadata.num_row_groups == 1
    for fold in range(2):
        manifest = _fold_manifest(loop.manifest, fold)
        manifest.pre_split_data_selector = None
        expected = manifest.run_model(manifest.prepare_data(bars, {'entry_return': 0.0}), {'entry_return': 0.0})
        row = loop.fold_results.filter(pl.col('fold') == fold).row(0, named=True)
        assert row['pnl_per_bar_bps_test'] == expected['pnl_per_bar_bps_test']


@pytest.mark.parametrize('invalid', ('short', 'unsorted', 'duplicate'))
def test_fold_geometry_rejects_unusable_recorded_rows(invalid, tmp_path):
    bars = _bars()
    if invalid == 'short':
        bars = bars.head(96)
    elif invalid == 'unsorted':
        bars = bars.reverse()
    else:
        bars = pl.concat([bars, bars.tail(1)])
    with pytest.raises(ValueError, match='split_walk_forward'):
        _run(_loop(_config(), bars, tmp_path))


def test_objective_requires_fold_local_validation(tmp_path):
    config = objective_config()
    manifest = config['sfd']['manifest']
    manifest.pop('split_dates')
    manifest['split_walk_forward'] = _GEOMETRY
    with pytest.raises(ValueError, match='validation'):
        _run(_loop(config, _bars(), tmp_path, ratios=(1, 0, 0)))

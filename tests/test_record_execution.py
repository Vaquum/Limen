import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner
from ruamel.yaml import YAML

from limen.backtest._snapshot_ledger import snapshot_ledger
from limen.backtest.long_flat_strategy import ExecutionResult
from limen.cli.main import cli
from limen.cohort import Cohort
from limen.data import HistoricalData
from limen.experiment import UniversalExperimentLoop
from limen.inference import Trainer
from limen.sfd.reference_architecture import RidgeRegressor
from limen.sfd.reference_architecture import _backtest_evaluation
from limen.yaml import CompiledSFD, build_search_strategy, parse, validate

ROOT = Path(__file__).resolve().parents[1]


def _recorded_spot_klines(**kwargs):
    return pl.read_parquet(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet').head(1200)


@pytest.fixture(scope='module')
def recorded_bars():
    return pl.read_parquet(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet').head(1200)


def _config(kind='ridge', *, notional_rate=0.5, take_profit_bps=None):
    config, errors = parse((ROOT / 'limen/yaml/templates/ridge_regressor.yaml').read_text())
    assert not errors
    manifest = config['sfd']['manifest']
    manifest['split_dates'] = {
        'train_start': '2024-01-01', 'train_end': '2024-01-23',
        'val_start': '2024-01-24', 'val_end': '2024-01-30',
        'test_start': '2024-01-31', 'test_end': '2024-02-06',
    }
    manifest['backtest'] = {
        'fee_bps': 7.0, 'slip_bps': 3.0, 'notional_rate': notional_rate,
        'take_profit_bps': take_profit_bps, 'stop_loss_bps': 50.0 if take_profit_bps is not None else None,
    }
    config['sfd']['params'] = {'alpha': [0.1, 1.0], 'solver': ['svd']}
    if kind == 'binary':
        manifest['target'] = {
            'name': 'quantile_flag', 'class': 'limen.targets.QuantileBinaryTarget',
            'fit_params': {'source_column': 'ret_1', 'quantile': 0.5},
            'transform_params': {'shift': -1},
        }
        manifest['reference_architecture'] = 'limen.sfd.reference_architecture.logreg_binary'
        config['sfd']['params'] = {'C': [0.1, 1.0], 'max_iter': [1000]}
    elif kind == 'rule':
        config['sfd']['manifest'] = {
            'type': 'rule_based', 'data_source': manifest['data_source'],
            'split_dates': manifest['split_dates'], 'backtest': manifest['backtest'],
            'strategy': {
                'conditions': [{'id': 'entry', 'name': 'entry', 'type': 'relative',
                                'column': 'close', 'operator': '>', 'other_column': 'open'}],
                'entry': 'entry',
            },
            'reference_architecture': 'limen.sfd.reference_architecture.rule_based',
        }
        config['sfd']['params'] = {'sharpe_std_threshold': [0.5, 0.7]}
    config['uel'].update(n_permutations=2, search_strategy={'type': 'grid'}, checkpoint_interval=1)
    return config


def _loop(config, recorded_bars, path, *, search=True):
    return UniversalExperimentLoop(
        sfd=CompiledSFD(config), data=recorded_bars,
        search_strategy=build_search_strategy(config) if search else None,
        experiment_dir=path, checkpoint_interval=1, yaml_reference=config,
    )


def _run(loop, **kwargs):
    loop.run('recorded_execution', n_permutations=2, prep_each_round=True, progress_bar=False, **kwargs)


def _records(path):
    return [json.loads(line) for line in (path / 'round_data.jsonl').read_text().splitlines()]


def _stop_after_first(loop):
    original = loop.model
    assert original is not None

    def interrupted(data, round_params):
        result = original(data, round_params)
        loop._shutdown_requested = True
        return result

    loop.model = interrupted


def test_default_and_false_preserve_round_artifacts(recorded_bars, tmp_path):
    config = _config()
    baseline = _loop(config, recorded_bars, tmp_path / 'default')
    explicit = _loop(config, recorded_bars, tmp_path / 'false')
    _run(baseline)
    _run(explicit, record_execution=False)
    assert (tmp_path / 'default/round_data.jsonl').read_bytes() == (tmp_path / 'false/round_data.jsonl').read_bytes()
    assert baseline.experiment_log.drop('execution_time').equals(explicit.experiment_log.drop('execution_time'))
    for path in (tmp_path / 'default', tmp_path / 'false'):
        assert all('execution' not in record for record in _records(path))
        assert 'record_execution' not in json.loads((path / 'metadata.json').read_text())


@pytest.mark.parametrize('kind', ('ridge', 'binary', 'rule'))
@pytest.mark.parametrize(('notional_rate', 'take_profit_bps'), ((1.0, None), (0.5, 75.0)))
def test_recorded_execution_is_the_evaluated_test_series(
    kind, notional_rate, take_profit_bps, recorded_bars, tmp_path, monkeypatch,
):
    observed = []
    original = _backtest_evaluation._snapshot_with_execution

    def capture(*args, **kwargs):
        metrics, execution = original(*args, **kwargs)
        observed.append(execution)
        return metrics, execution

    monkeypatch.setattr(_backtest_evaluation, '_snapshot_with_execution', capture)
    loop = _loop(_config(kind, notional_rate=notional_rate, take_profit_bps=take_profit_bps), recorded_bars, tmp_path)
    _run(loop, record_execution=True)
    records = _records(tmp_path)
    assert len(records) == 2
    test_executions = observed[2::3] if kind == 'rule' else observed
    assert len(test_executions) == len(records)
    for index, (record, expected) in enumerate(zip(records, test_executions, strict=True)):
        payload = record['execution']
        assert set(payload) == {'pos', 'gross', 'net'}
        for field in ExecutionResult._fields:
            assert len(payload[field]) == len(record['preds'])
            np.testing.assert_array_equal(payload[field], getattr(expected, field) * notional_rate)
        recorded = ExecutionResult(*(np.asarray(payload[field]) for field in ExecutionResult._fields))
        for metric, value in snapshot_ledger(recorded, 1.0).items():
            column = f'{metric}_test' if kind == 'rule' else f'backtest_{metric}'
            np.testing.assert_equal(loop.experiment_log[column][index], value)
    assert loop._alignment == []
    assert json.loads((tmp_path / 'metadata.json').read_text())['record_execution'] is True


@pytest.mark.parametrize('record_execution', (False, True))
def test_python_resume_preserves_setting_and_rounds(record_execution, recorded_bars, tmp_path):
    config = _config()
    first = _loop(config, recorded_bars, tmp_path)
    _stop_after_first(first)
    _run(first, record_execution=record_execution)
    before = (tmp_path / 'round_data.jsonl').read_bytes()
    assert len(_records(tmp_path)) == 1
    wrong = _loop(config, recorded_bars, tmp_path)
    with pytest.raises(ValueError, match='record_execution'):
        _run(wrong, record_execution=not record_execution, resume=True)
    assert (tmp_path / 'round_data.jsonl').read_bytes() == before
    resumed = _loop(config, recorded_bars, tmp_path)
    _run(resumed, record_execution=record_execution, resume=True, post_processing=True)
    records = _records(tmp_path)
    assert len(records) == 2
    assert records[0] == json.loads(before)
    assert len({record['round_id'] for record in records}) == 2
    assert all(('execution' in record) == record_execution for record in records)
    if record_execution:
        assert all(record['execution'] is not None for record in records)


def test_cli_recording_continues_after_resume(recorded_bars, tmp_path, monkeypatch):
    config = _config()
    path = tmp_path / 'experiment'
    config['uel'].update(output_path=str(path), record_execution=True)
    yaml_path = tmp_path / 'experiment.yaml'
    with yaml_path.open('w') as handle:
        YAML().dump(config, handle)
    monkeypatch.setattr(HistoricalData, 'get_spot_klines', staticmethod(_recorded_spot_klines))
    original_run = UniversalExperimentLoop.run

    def interrupted_run(loop, *args, **kwargs):
        _stop_after_first(loop)
        return original_run(loop, *args, **kwargs)

    runner = CliRunner()
    with monkeypatch.context() as context:
        context.setattr(UniversalExperimentLoop, 'run', interrupted_run)
        result = runner.invoke(cli, ['run', '--no-progress-bar', str(yaml_path)])
    assert result.exit_code == 0, result.output
    before = _records(path)
    assert len(before) == 1
    result = runner.invoke(cli, ['run', '--no-progress-bar', '--resume', str(path)])
    assert result.exit_code == 0, result.output
    records = _records(path)
    assert len(records) == 2
    assert records[0] == before[0]
    assert all(record['execution'] is not None for record in records)
    sensors = Trainer(path, data=recorded_bars).train([records[0]['round_id']])
    assert len(sensors) == 1
    assert sensors[0].predict(recorded_bars).reason is None
    cohort = Cohort(experiment_log_path=str(path), permutation_ids=[records[0]['round_id']])
    assert cohort.permutation_ids == [records[0]['round_id']]


@pytest.mark.parametrize('unavailable', ('prices', 'inline', 'custom'))
def test_unavailable_execution_is_explicit_null(unavailable, recorded_bars, tmp_path):
    config = _config()
    config['sfd']['manifest'].pop('backtest')
    loop = _loop(config, recorded_bars, tmp_path)
    if unavailable == 'prices':
        original = loop.prep
        assert original is not None

        def without_prices(data, round_params):
            prepared = original(data, round_params)
            prepared.pop('price_data_for_backtest')
            return prepared

        loop.prep = without_prices
    elif unavailable == 'inline':
        loop.model = lambda data, round_params: RidgeRegressor().train(data, **round_params).evaluate(data, inline_metrics=False)
    else:
        loop.model = lambda data, round_params: {'_preds': np.asarray(data['y_test'])}
    _run(loop, record_execution=True)
    assert all(record['execution'] is None for record in _records(tmp_path))


def test_event_execution_keeps_its_trade_ledger(recorded_bars, tmp_path):
    config = _config('rule')
    config['sfd']['manifest']['backtest'] = {
        'product': {'kind': 'cash_spot', 'instrument': 'BTCUSDT', 'base_currency': 'BTC',
                    'quote_currency': 'USDT', 'quantity_step': 1e-9, 'min_notional': 0},
    }
    baseline = _loop(config, recorded_bars, tmp_path / 'off')
    enabled = _loop(config, recorded_bars, tmp_path / 'on')
    _run(baseline)
    _run(enabled, record_execution=True)
    for off, on in zip(_records(tmp_path / 'off'), _records(tmp_path / 'on'), strict=True):
        assert on.pop('execution') is None
        assert 'trade_ledger' in on
        assert on == off
    assert enabled.experiment_log.drop('execution_time').equals(baseline.experiment_log.drop('execution_time'))


def test_reused_preparation_clears_previous_execution(recorded_bars, tmp_path):
    config = _config()
    config['sfd']['manifest'].pop('backtest')
    loop = _loop(config, recorded_bars, tmp_path)
    prepared = loop.manifest.prepare_data(recorded_bars, {'alpha': 0.1, 'solver': 'svd'})
    calls = 0

    def reuse(data, round_params):
        nonlocal calls
        calls += 1
        if calls == 2:
            prepared.pop('price_data_for_backtest')
        return prepared

    loop.prep = reuse
    _run(loop, record_execution=True)
    records = _records(tmp_path)
    assert records[0]['execution'] is not None
    assert records[1]['execution'] is None


@pytest.mark.parametrize('value', ('false', 0, 1, None))
def test_record_execution_requires_bool(value, recorded_bars, tmp_path):
    config = _config()
    config['uel']['record_execution'] = value
    result = validate(config)
    assert not result.valid
    assert any(error.path == 'uel.record_execution' for error in result.errors)
    loop = _loop(config, recorded_bars, tmp_path)
    with pytest.raises(ValueError, match='record_execution'):
        _run(loop, record_execution=value)
    assert not (tmp_path / 'results.csv').exists()


@pytest.mark.parametrize('missing', ('search_strategy', 'experiment_dir'))
def test_record_execution_requires_a_writer(missing, recorded_bars, tmp_path):
    loop = _loop(_config(), recorded_bars, None if missing == 'experiment_dir' else tmp_path,
                 search=missing != 'search_strategy')
    with pytest.raises(ValueError, match='record_execution'):
        _run(loop, record_execution=True)
    assert not (tmp_path / 'results.csv').exists()


def test_cli_resume_uses_python_recording_setting(recorded_bars, tmp_path, monkeypatch):
    config = _config()
    assert 'record_execution' not in config['uel']
    first = _loop(config, recorded_bars, tmp_path)
    _stop_after_first(first)
    _run(first, record_execution=True)
    before = _records(tmp_path)
    metadata_path = tmp_path / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    assert len(before) == 1
    assert metadata['record_execution'] is True
    assert metadata['yaml_reference'] == config

    monkeypatch.setattr(HistoricalData, 'get_spot_klines', staticmethod(_recorded_spot_klines))
    result = CliRunner().invoke(cli, ['run', '--no-progress-bar', '--resume', str(tmp_path)])
    assert result.exit_code == 0, result.output
    records = _records(tmp_path)
    assert len(records) == 2
    assert records[0] == before[0]
    assert all(record['execution'] is not None for record in records)
    assert json.loads(metadata_path.read_text())['yaml_reference'] == config
    assert 'record_execution' not in config['uel']

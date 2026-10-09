import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner
from ruamel.yaml import YAML

from limen.cli.main import cli
from limen.data import HistoricalData
from limen.experiment import CalibrationConfig, UniversalExperimentLoop
from limen.inference import Trainer
from limen.sfd.reference_architecture import LightGBMBinary, LogRegBinary, TabPFNBinary, XGBoostRegressor
from limen.sfd.reference_architecture.base import ReferenceModel
from limen.yaml import CompiledSFD, build_search_strategy, parse, validate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def recorded_bars():
    return recorded_source()


def recorded_source(**kwargs):
    return pl.read_parquet(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet').head(1200)


def _config(kind='logreg', *, calibrated=False):
    config, errors = parse((ROOT / 'limen/yaml/templates/ridge_regressor.yaml').read_text())
    assert not errors
    manifest = config['sfd']['manifest']
    manifest['split_dates'] = {
        'train_start': '2024-01-01', 'train_end': '2024-01-23',
        'val_start': '2024-01-24', 'val_end': '2024-01-30',
        'test_start': '2024-01-31', 'test_end': '2024-02-06',
    }
    architecture = {'logreg': 'logreg_binary', 'lightgbm': 'lightgbm_binary',
                    'random': 'random_binary', 'ridge': 'ridge_regressor', 'xgboost': 'xgboost_regressor'}[kind]
    manifest['reference_architecture'] = f'limen.sfd.reference_architecture.{architecture}'
    if kind in ('logreg', 'lightgbm', 'random'):
        manifest['target'] = {
            'name': 'quantile_flag', 'class': 'limen.targets.QuantileBinaryTarget',
            'fit_params': {'source_column': 'ret_1', 'quantile': 0.5},
            'transform_params': {'shift': -1},
        }
    config['sfd']['params'] = {
        'logreg': {'C': [0.1, 1.0], 'max_iter': [1000]},
        'lightgbm': {'learning_rate': [0.05, 0.1], 'n_estimators': [20], 'num_leaves': [5],
                     'early_stopping_rounds': [3], 'verbosity': [-1], 'n_jobs': [1]},
        'random': {'random_weights': [0.4, 0.6]},
        'ridge': {'alpha': [0.1, 1.0], 'solver': ['svd']},
        'xgboost': {'n_estimators': [10, 20], 'early_stopping_rounds': [3], 'n_jobs': [1]},
    }[kind]
    if calibrated:
        manifest['calibration'] = {
            'probability_calibration': {'func': 'limen.calibration.sklearn_probability_calibrator',
                                        'params': {'method': 'sigmoid'}},
            'threshold_function': {'func': 'limen.calibration.grid_threshold_optimizer',
                                   'params': {'metric': 'limen.metrics.balanced_metric.balanced_metric',
                                              'threshold_min': 0.4, 'threshold_max': 0.6, 'threshold_step': 0.05}},
        }
    config['uel'].update(n_permutations=2, search_strategy={'type': 'grid'}, checkpoint_interval=1)
    return config


def _loop(config, bars, path, *, search=True):
    return UniversalExperimentLoop(
        sfd=CompiledSFD(config), data=bars, experiment_dir=path, yaml_reference=config,
        search_strategy=build_search_strategy(config) if search else None, checkpoint_interval=1,
    )


def _run(loop, **kwargs):
    loop.run('recorded_model_outputs', n_permutations=2, prep_each_round=True, progress_bar=False, **kwargs)


def _records(path):
    return [json.loads(line) for line in (path / 'round_data.jsonl').read_text().splitlines()]


def _assert_predictions(record):
    probabilities = np.asarray(record['probs'])
    assert len(probabilities) == len(record['preds'])
    threshold = record['optimal_threshold']
    predictions = probabilities >= threshold if record['threshold_rule'] == '>=' else probabilities > threshold
    np.testing.assert_array_equal(predictions, record['preds'])


def _stop_after_first(loop):
    original = loop.model
    assert original is not None

    def interrupted(data, round_params):
        result = original(data, round_params)
        loop._shutdown_requested = True
        return result

    loop.model = interrupted


def test_default_and_false_preserve_outputs(recorded_bars, tmp_path):
    config = _config()
    baseline = _loop(config, recorded_bars, tmp_path / 'default')
    explicit = _loop(config, recorded_bars, tmp_path / 'false')
    _run(baseline)
    _run(explicit, record_model_outputs=False)
    assert (tmp_path / 'default/round_data.jsonl').read_bytes() == (tmp_path / 'false/round_data.jsonl').read_bytes()
    assert baseline.experiment_log.drop('execution_time').equals(explicit.experiment_log.drop('execution_time'))
    assert all('probs' not in row for row in _records(tmp_path / 'default'))
    assert 'best_iteration' not in baseline.experiment_log.columns
    assert 'record_model_outputs' not in json.loads((tmp_path / 'default/metadata.json').read_text())


@pytest.mark.parametrize(('kind', 'calibrated'), (('logreg', False), ('logreg', True),
                                               ('lightgbm', False), ('lightgbm', True), ('random', False)))
def test_python_records_original_probabilities(kind, calibrated, recorded_bars, tmp_path, monkeypatch):
    captured = []
    original = ReferenceModel._record_probabilities

    def capture(model, data, prediction):
        captured.append(np.asarray(prediction['_probs']).copy())
        original(model, data, prediction)

    monkeypatch.setattr(ReferenceModel, '_record_probabilities', capture)
    loop = _loop(_config(kind, calibrated=calibrated), recorded_bars, tmp_path)
    _run(loop, record_model_outputs=True)
    records = _records(tmp_path)
    assert len(captured) == len(records) == 2
    for record, expected in zip(records, captured, strict=True):
        np.testing.assert_array_equal(record['probs'], expected)
        _assert_predictions(record)
        assert record['threshold_rule'] == ('>=' if calibrated else '>')
        if not calibrated:
            assert record['optimal_threshold'] == 0.5
    assert '_probs' not in loop.experiment_log.columns
    assert 'probs' not in loop.experiment_log.columns
    assert loop.preds == loop._alignment == []
    assert json.loads((tmp_path / 'metadata.json').read_text())['record_model_outputs'] is True


@pytest.mark.parametrize('model_class', (LogRegBinary, LightGBMBinary, TabPFNBinary))
@pytest.mark.parametrize('calibrated', (False, True))
def test_binary_evaluation_records_without_inline_metrics(model_class, calibrated, recorded_bars):
    compiled = CompiledSFD(_config(calibrated=calibrated))
    manifest = compiled.manifest()
    data = manifest.prepare_data(recorded_bars, {'C': 0.1, 'max_iter': 1000})
    config = manifest.prediction_calibration_config
    model = model_class(prediction_calibration_config=config)
    if model_class is TabPFNBinary:
        model.model = LogRegBinary().train(data, max_iter=1000).model
    else:
        params = {'n_estimators': 20, 'verbosity': -1, 'n_jobs': 1} if model_class is LightGBMBinary else {'max_iter': 1000}
        model.train(data, **params)
    data['_record_model_outputs'] = True
    result = model.evaluate(data, inline_metrics=False)
    record = {**data['_alignment']['model_outputs'], 'preds': result['_preds']}
    _assert_predictions(record)
    assert record['threshold_rule'] == ('>=' if calibrated else '>')
    assert not any(key.startswith('backtest_') for key in result)
    expected = model.predict(data)['_probs']
    np.testing.assert_array_equal(record['probs'], expected)


@pytest.mark.parametrize('threshold_path', (False, True))
def test_exact_threshold_boundary_preserves_backend_rule(threshold_path, recorded_bars):
    manifest = CompiledSFD(_config()).manifest()
    data = manifest.prepare_data(recorded_bars, {'C': 0.1, 'max_iter': 1000})
    model = LogRegBinary().train(data, max_iter=1000)
    model.model.coef_.fill(0)
    model.model.intercept_.fill(0)
    if threshold_path:
        model.prediction_calibration_config = CalibrationConfig()
    data['_record_model_outputs'] = True
    result = model.evaluate(data, inline_metrics=False)
    record = {**data['_alignment']['model_outputs'], 'preds': result['_preds']}
    assert np.asarray(record['probs']).min() == np.asarray(record['probs']).max() == 0.5
    assert record['threshold_rule'] == ('>=' if threshold_path else '>')
    _assert_predictions(record)
    assert np.asarray(result['_preds']).all() == threshold_path


@pytest.mark.parametrize('fault', ('short', 'nan', 'shape'))
def test_recorded_probabilities_reject_invalid_arrays(fault, recorded_bars):
    manifest = CompiledSFD(_config()).manifest()
    data = manifest.prepare_data(recorded_bars, {'C': 0.1, 'max_iter': 1000})
    model = LogRegBinary().train(data, max_iter=1000)
    prediction = model.predict(data)
    if fault == 'short':
        prediction['_probs'] = prediction['_probs'][:-1]
    elif fault == 'nan':
        prediction['_probs'][0] = np.nan
    else:
        prediction['_probs'] = prediction['_probs'][:, None]
    data['_record_model_outputs'] = True
    with pytest.raises(ValueError, match='finite and aligned'):
        model._record_probabilities(data, prediction)


@pytest.mark.parametrize(('kind', 'stopping', 'booster'), (('lightgbm', True, None), ('lightgbm', False, None),
                                                        ('xgboost', True, 'gbtree'), ('xgboost', False, 'gbtree'),
                                                        ('xgboost', True, 'gblinear')))
def test_recorded_boosting_count_matches_prediction(kind, stopping, booster, recorded_bars):
    compiled = CompiledSFD(_config(kind))
    data = compiled.manifest().prepare_data(recorded_bars, {})
    params = {'n_estimators': 20, 'early_stopping_rounds': 3 if stopping else None, 'n_jobs': 1}
    if kind == 'lightgbm':
        model = LightGBMBinary().train(data, verbosity=-1, num_leaves=5, **params)
        expected = model.model.best_iteration_ if model.model.best_iteration_ > 0 else model.model.n_iter_
    else:
        model = XGBoostRegressor().train(data, booster=booster, **params)
        native = model.model.get_booster()
        expected = model.model.best_iteration + 1 if stopping and booster != 'gblinear' else native.num_boosted_rounds()
    off = model.evaluate(data, inline_metrics=False)
    assert 'best_iteration' not in off
    data['_record_model_outputs'] = True
    on = model.evaluate(data, inline_metrics=False)
    assert on['best_iteration'] == expected > 0
    if kind == 'xgboost' and booster != 'gblinear':
        np.testing.assert_array_equal(on['_preds'], model.model.predict(data['x_test'], iteration_range=(0, expected)))
    np.testing.assert_array_equal(off['_preds'], on['_preds'])


@pytest.mark.parametrize('record_outputs', (False, True))
def test_python_resume_rejects_changed_setting_without_rewriting(record_outputs, recorded_bars, tmp_path):
    config = _config('lightgbm')
    first = _loop(config, recorded_bars, tmp_path)
    _stop_after_first(first)
    _run(first, record_model_outputs=record_outputs)
    before = {name: (tmp_path / name).read_bytes() for name in ('results.csv', 'round_data.jsonl', 'metadata.json', 'checkpoint.json')}
    wrong = _loop(config, recorded_bars, tmp_path)
    with pytest.raises(ValueError, match='record_model_outputs'):
        _run(wrong, record_model_outputs=not record_outputs, resume=True)
    assert all((tmp_path / name).read_bytes() == value for name, value in before.items())
    resumed = _loop(config, recorded_bars, tmp_path)
    _run(resumed, record_model_outputs=record_outputs, resume=True)
    records = _records(tmp_path)
    assert len(records) == len({row['round_id'] for row in records}) == 2
    assert records[0] == json.loads(before['round_data.jsonl'])
    assert all(('probs' in row) == record_outputs for row in records)
    if record_outputs:
        for record in records:
            _assert_predictions(record)
        assert resumed.experiment_log['best_iteration'].is_not_null().all()


@pytest.mark.parametrize('record_execution', (False, True))
def test_cli_records_and_resumes_independently_of_execution(record_execution, recorded_bars, tmp_path, monkeypatch):
    config = _config('lightgbm', calibrated=True)
    path = tmp_path / 'experiment'
    config['uel'].update(output_path=str(path), record_model_outputs=True, record_execution=record_execution)
    yaml_path = tmp_path / 'experiment.yaml'
    with yaml_path.open('w') as handle:
        YAML().dump(config, handle)
    monkeypatch.setattr(HistoricalData, 'get_spot_klines', staticmethod(recorded_source))
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
    for record in records:
        _assert_predictions(record)
        assert ('execution' in record) == record_execution
    log = pl.read_csv(path / 'results.csv')
    assert log['best_iteration'].is_not_null().all()
    assert not any(column in log.columns for column in ('_probs', 'probs', 'threshold_rule'))
    sensors = Trainer(path, data=recorded_bars).train([records[0]['round_id']])
    assert len(sensors) == 1
    assert sensors[0].predict(recorded_bars).reason is None


@pytest.mark.parametrize('kind', ('ridge', 'xgboost'))
def test_architectures_without_probabilities_record_null(kind, recorded_bars, tmp_path):
    loop = _loop(_config(kind), recorded_bars, tmp_path)
    _run(loop, record_model_outputs=True)
    assert all(row['probs'] is None for row in _records(tmp_path))
    assert ('best_iteration' in loop.experiment_log.columns) == (kind == 'xgboost')


def test_reused_alignment_does_not_record_previous_probabilities(recorded_bars, tmp_path):
    loop = _loop(_config(), recorded_bars, tmp_path)
    prepared = loop.manifest.prepare_data(recorded_bars, {'C': 0.1, 'max_iter': 1000})
    loop.prep = lambda data, round_params: prepared
    original = loop.model
    calls = 0

    def first_only(data, round_params):
        nonlocal calls
        calls += 1
        return original(data, round_params) if calls == 1 else {'_preds': np.asarray(data['y_test'])}

    loop.model = first_only
    _run(loop, record_model_outputs=True)
    records = _records(tmp_path)
    assert records[0]['probs'] is not None
    assert records[1]['probs'] is None


@pytest.mark.parametrize('value', ('false', 0, 1, None))
def test_recording_requires_bool(value, recorded_bars, tmp_path):
    config = _config()
    config['uel']['record_model_outputs'] = value
    result = validate(config)
    assert not result.valid
    assert any(error.path == 'uel.record_model_outputs' for error in result.errors)
    loop = _loop(config, recorded_bars, tmp_path)
    with pytest.raises(TypeError, match='record_model_outputs'):
        _run(loop, record_model_outputs=value)
    assert not (tmp_path / 'results.csv').exists()


@pytest.mark.parametrize('missing', ('search_strategy', 'experiment_dir'))
def test_recording_requires_a_writer(missing, recorded_bars, tmp_path):
    loop = _loop(_config(), recorded_bars, None if missing == 'experiment_dir' else tmp_path,
                 search=missing != 'search_strategy')
    with pytest.raises(ValueError, match='record_model_outputs'):
        _run(loop, record_model_outputs=True)
    assert not (tmp_path / 'results.csv').exists()

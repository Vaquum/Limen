import copy
import inspect
import json
from dataclasses import replace
from functools import partial

import numpy as np
import polars as pl
import pytest

from limen.backtest.execution_events import with_predictions
from limen.backtest.trade_execution import trade_execution
from limen.sfd.reference_architecture import logreg_binary
from limen.yaml import validate
from limen.yaml import CompiledSFD, build_search_strategy
from limen.experiment import UniversalExperimentLoop, _round_storage
from limen.inference import Trainer
from tests.test_experiment_objective import COLUMN, _prepared, objective_config
from tests.test_record_model_outputs import recorded_source


class _Duck:
    prediction_mode = 'binary'

    def __init__(self, model, *, probabilities=True):
        self.model, self.probabilities = model, probabilities

    def predict(self, data):
        assert set(data) == {'x_test'}
        result = self.model.predict(data)
        return result if self.probabilities else {'_preds': result['_preds']}


def _original(data, params):
    return logreg_binary(data, **{key: value for key, value in params.items() if key in inspect.signature(logreg_binary).parameters})


def _duck(data, **params):
    result = _original(data, params)
    result['_model'] = _Duck(result['_model'], probabilities=False)
    return result


def _explicit(data, **params):
    result = _original(data, params)
    predictions = result['_model'].predict({'x_test': data['x_val']})['_preds']
    result['_validation'] = {**copy.deepcopy(data['_validation_context']), 'predictions': predictions}
    result.pop('_model')
    result['_prediction_mode'] = 'binary'
    return result


def _duck_probability(data, **params):
    result = _original(data, params)
    result['_model'] = _Duck(result['_model'])
    return result


def _explicit_probability(data, **params):
    result = _original(data, params)
    validation = result['_model'].predict({'x_test': data['x_val']})
    result['_validation'] = {**copy.deepcopy(data['_validation_context']), 'predictions': validation['_preds'], 'probabilities': validation['_probs']}
    return result


def _rule_config(direction):
    config = objective_config(direction, calibrated=False)
    manifest = config['sfd']['manifest']
    manifest.update(type='rule_based', reference_architecture='limen.sfd.reference_architecture.rule_based.rule_based',
                    strategy={'conditions': [{'id': 'up', 'name': 'Up bar', 'type': 'relative', 'column': 'close', 'operator': '>', 'other_column': 'open'}], 'entry': 'up'})
    for key in ('target', 'scaler', 'features', 'feature_shims', 'strict_mode'):
        manifest.pop(key, None)
    config['sfd']['params'] = {'sharpe_std_threshold': [0.5]}
    config['uel']['n_permutations'] = 1
    return config


@pytest.mark.parametrize('direction', ['maximize', 'minimize'])
@pytest.mark.parametrize('architecture', [partial(logreg_binary), _duck, _explicit])
def test_recorded_objective_admission(direction, architecture):
    config = objective_config(direction, calibrated=False)
    baseline, parameters, prepared = _prepared(config)
    expected = baseline.run_model(prepared, parameters)
    manifest, params, data = _prepared(config)
    manifest.with_reference_architecture(architecture)
    actual = manifest.run_model(data, params)
    assert actual[COLUMN] == expected[COLUMN]
    np.testing.assert_array_equal(actual['_preds'], expected['_preds'])
    assert '_validation' not in actual and '_validation_context' not in data


@pytest.mark.parametrize('direction', ['maximize', 'minimize'])
def test_recorded_objective_admission_rule(direction):
    config = _rule_config(direction)
    validation = validate(config)
    assert validation.valid
    assert not any('objective' in str(warning) for warning in validation.warnings)
    manifest, params, data = _prepared(config)
    result = manifest.run_model(data, params)
    predictions = result['_model'].predict({'test': data['val'], 'strategy': data['strategy']})['_preds']
    context = data['_trade_context']
    expected = trade_execution(with_predictions(context.partitions[1], predictions), context.policy).metrics['total_return']
    assert result[COLUMN] == expected
    assert 'val_backtest_total_return' in result


@pytest.mark.parametrize('direction', ['maximize', 'minimize'])
def test_recorded_objective_admission_signed_exposures(direction):
    from limen.scalers import RobustScaler
    from tests.test_direction_sizing import native_manifest, recorded_source as exposure_source

    manifest = native_manifest().set_scaler(RobustScaler).set_objective(direction=direction)
    params = {'holding': 1800, 'folds': 2, 'min_train_samples': 2, 'direction_params': {'max_iter': 1000}}
    data = manifest.prepare_data(exposure_source(limit=800), params)
    result = manifest.run_model(data, params)
    decisions = result['_model'].predict({'x_test': data['x_val']})['_preds']
    assert np.any(decisions < 0) and np.any(decisions > 0)
    context = data['_trade_context']
    expected = trade_execution(with_predictions(context.partitions[1], decisions), context.policy)
    assert result[COLUMN] == expected.metrics['total_return']


@pytest.mark.parametrize('fault', ['rows', 'source', 'mode', 'missing', 'nonfinite', 'length'])
def test_objective_evidence_refusals(fault):
    def invalid(data, **params):
        result = _explicit(data, **params)
        payload = result['_validation']
        if fault == 'rows':
            payload['row_ids'].reverse()
        elif fault == 'source':
            payload['source_binding'] = 'stale'
        elif fault == 'mode':
            payload['prediction_mode'] = 'target_exposure'
        elif fault == 'missing':
            payload.pop('predictions')
        elif fault == 'nonfinite':
            payload['predictions'] = np.asarray(payload['predictions'], dtype=float)
            payload['predictions'][0] = float('nan')
        else:
            payload['predictions'] = payload['predictions'][:-1]
        return result
    manifest, params, data = _prepared(objective_config(calibrated=False))
    manifest.with_reference_architecture(invalid)
    with pytest.raises(ValueError, match='Objective'):
        manifest.run_model(data, params)


@pytest.mark.parametrize('direction', ['maximize', 'minimize'])
@pytest.mark.parametrize('flags', [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize('architecture', [partial(logreg_binary), _duck_probability, _explicit_probability])
def test_recorded_objective_tuning(direction, flags, architecture):
    config = objective_config(direction)
    baseline, params, original = _prepared(config)
    params.update(zip(('use_calibration', 'use_threshold'), flags, strict=True))
    expected = baseline.run_model(original, params)
    manifest, _, data = _prepared(config)
    manifest.with_reference_architecture(architecture)
    result = manifest.run_model(data, params)
    assert result[COLUMN] == expected[COLUMN]
    assert result['optimal_threshold'] == expected['optimal_threshold']
    np.testing.assert_array_equal(result['_preds'], expected['_preds'])


@pytest.mark.parametrize('fault', ['held_out_labels', 'reordered_features', 'discarded_calibrator', 'raw_decisions'])
def test_objective_evidence_refusals_calibration(fault):
    def invalid(data, **params):
        if fault == 'held_out_labels':
            data['y_val'] = data['y_test']
        elif fault == 'reordered_features':
            data['x_val'] = data['x_val'].reverse()
        if fault == 'discarded_calibrator':
            config = params.pop('prediction_calibration_config')
            result = _original(data, params)
            _ = config.calibration_func(result['_model'].model, np.asarray(data['x_val']), data['y_val'], **config.calibration_params)
            return result
        result = _original(data, params)
        if fault == 'raw_decisions':
            model = result['_model']
            probabilities = model.predict({'x_test': data['x_val']})['_probs']
            raw = model.model.predict(np.asarray(data['x_val']))
            assert np.any(raw != (probabilities >= 0.5))
            result['_validation'] = {**copy.deepcopy(data['_validation_context']), 'predictions': raw, 'probabilities': probabilities}
        return result
    manifest, params, data = _prepared(objective_config('minimize'))
    assert len(data['y_val']) == len(data['y_test'])
    assert not np.array_equal(data['y_val'], data['y_test'])
    params['use_threshold'] = False
    manifest.with_reference_architecture(invalid)
    with pytest.raises(ValueError, match='Objective'):
        manifest.run_model(data, params)


@pytest.mark.parametrize('ml', [False, True, 'logreg'])
@pytest.mark.parametrize('direction', ['maximize', 'minimize'])
def test_recorded_objective_allocation(ml, direction):
    from tests.test_meta_allocation import _prepared as allocation_data

    baseline, _, params, original = allocation_data(ml)
    if ml == 'logreg':
        baseline.allocation_config.params['features'] = ['ret_1']
    expected = baseline.run_model(original, params)
    manifest, raw, _, _ = allocation_data(ml)
    if ml == 'logreg':
        manifest.allocation_config.params['features'] = ['ret_1']
    manifest.set_objective(direction=direction)
    data = manifest.prepare_data(raw, params)
    result = manifest.run_model(data, params)
    model = result['_model']
    decisions = model.predict({'allocation_batch': data['_allocation_batches']['val']})['_preds']
    if ml is not True:
        assert np.any((decisions > 0) & (decisions < 1))
    assert np.isfinite(decisions).all()
    context = data['_trade_context']
    oracle = trade_execution(with_predictions(context.partitions[1], decisions), context.policy)
    assert result[COLUMN] == oracle.metrics['total_return']
    np.testing.assert_array_equal(result['_preds'], expected['_preds'])
    altered = replace(data['_allocation_batches']['val'], partition_end_ns=context.partitions[1].partition_end_ns + 1)
    data['_allocation_batches']['val'] = altered
    with pytest.raises(ValueError, match='Objective allocation validation batch'):
        manifest.run_model(data, params)


@pytest.mark.parametrize('rule', [False, True])
def test_recorded_objective_folds(rule, tmp_path):
    from tests.test_walk_forward_uel import _GEOMETRY, _bars, _loop, _records

    config = _rule_config('minimize') if rule else objective_config('minimize', calibrated=False)
    declaration = config['sfd']['manifest']
    declaration.pop('split_dates')
    declaration['split_walk_forward'] = _GEOMETRY
    if not rule:
        config['sfd']['params']['C'] = [0.1]
    loop = _loop(config, _bars(), tmp_path, ratios=(4, 1, 0))
    loop.run('causal_objective_folds', n_permutations=1, prep_each_round=True, progress_bar=False,
             record_execution=True, record_model_outputs=True,
             recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True))
    scores = loop.fold_results[COLUMN]
    assert scores.len() == 2 and scores.is_finite().all()
    assert loop.experiment_log[COLUMN][0] == pytest.approx(scores.mean())
    records = _records(tmp_path)
    assert len(records) == 1 and len(records[0]['folds']) == 2
    assert '_validation' not in json.dumps(records)


@pytest.mark.parametrize('rule', [False, True])
def test_recorded_objective_lifecycle(rule, tmp_path):
    from click.testing import CliRunner
    from limen.cli.main import cli
    from tests.test_record_model_outputs import ROOT

    config = _rule_config('maximize') if rule else objective_config(calibrated=False)
    config['sfd']['manifest']['data_source'] = {'method': 'limen.data.HistoricalData.get_any_file', 'params': {'file_path_or_url': str(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet')}}
    config['sfd']['manifest']['backtest']['source_interval_seconds'] = 3600
    directory = tmp_path / 'results'
    config['uel'].update(output_path=str(directory), n_permutations=1)
    path = tmp_path / 'objective.yaml'
    path.write_text(json.dumps(config))
    runner = CliRunner()
    checked = runner.invoke(cli, ['validate', str(path)])
    assert checked.exit_code == 0, checked.output
    executed = runner.invoke(cli, ['run', '--no-progress-bar', str(path)])
    assert executed.exit_code == 0, executed.output
    assert 'objective is not used' not in executed.output and 'Unknown field' not in executed.output
    metadata = json.loads((directory / 'metadata.json').read_text())
    assert metadata['objective'] == config['sfd']['manifest']['objective']
    frame = pl.read_csv(directory / 'results.csv')
    assert frame.height == 1 and frame[COLUMN].is_finite().all()
    retained = _round_storage.text(_round_storage.round_path(directory))
    assert '_validation' not in retained
    sensor = Trainer(directory, data=recorded_source()).train(frame['id'].to_list())[0]
    assert sensor.predict(recorded_source()).reason is None


@pytest.mark.parametrize('kind', ['calibrator_inactive', 'optimizer_inactive', 'both_inactive'])
def test_recorded_objective_lifecycle_ordinary_empty_calibration(kind, tmp_path):
    from limen.experiment._holdout_freeze import validate_member
    from limen.experiment.holdout import reserve_holdout
    from tests.test_holdout_research import _project, _configuration

    root = _project(tmp_path / 'project')
    config = _configuration(root)
    reserve_holdout(root, '2024-01-05', '2024-01-07', interval_seconds=3600)
    config['sfd']['manifest']['calibration'] = {}
    if kind != 'optimizer_inactive':
        config['sfd']['manifest']['calibration']['probability_calibration'] = {'func': 'limen.calibration.sklearn_probability_calibrator', 'params': {'method': 'sigmoid'}}
    if kind != 'calibrator_inactive':
        config['sfd']['manifest']['calibration']['threshold_function'] = {'func': 'limen.calibration.grid_threshold_optimizer'}
    config['sfd']['params'].update(use_calibration=[kind == 'optimizer_inactive'], use_threshold=[kind == 'calibrator_inactive'])
    directory = root / 'results'
    loop = UniversalExperimentLoop(sfd=CompiledSFD(config, project_root=root), project_root=root, experiment_dir=directory,
                                   search_strategy=build_search_strategy(config), yaml_reference=config)
    loop.run(kind, n_permutations=1, progress_bar=False, post_processing=True)
    ids = loop.experiment_log['_id'].to_list()
    sensor = Trainer(directory, project_root=root).train(ids)[0]
    assert validate_member(sensor, None)
    model = sensor._model
    assert model._fitted_calibrator is (None if kind == 'both_inactive' else model.model)

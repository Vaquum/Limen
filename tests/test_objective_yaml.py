import copy
from pathlib import Path

import pytest

from limen.experiment.manifest_core import MLManifest
from limen.yaml.compiler import build_manifest, build_pruning_strategies
from limen.yaml.parser import parse
from limen.yaml.validator import validate

_TEMPLATES = Path(__file__).resolve().parents[1] / 'limen/yaml/templates'
_COLUMN = 'val_backtest_total_return'


def _configuration(template='logreg_binary'):
    yaml_dict, errors = parse(_TEMPLATES / f'{template}.yaml')
    assert not errors
    return yaml_dict


def _objective_configuration(direction='maximize'):
    yaml_dict = _configuration()
    manifest = yaml_dict['sfd']['manifest']
    manifest['objective'] = {'metric': 'backtest_total_return', 'direction': direction}
    manifest['calibration']['threshold_function']['params'].pop('metric')
    return yaml_dict


@pytest.mark.parametrize('direction', ['maximize', 'minimize'])
def test_compiled_objective_matches_native_manifest(direction):
    yaml_dict = _objective_configuration(direction)
    before = copy.deepcopy(yaml_dict)
    assert validate(yaml_dict).valid
    compiled = build_manifest(yaml_dict)
    assert isinstance(compiled, MLManifest)
    native = MLManifest().set_objective(metric='backtest_total_return', direction=direction)
    assert compiled.objective == native.objective
    assert compiled.objective.as_dict() == yaml_dict['sfd']['manifest']['objective']
    assert yaml_dict == before


@pytest.mark.parametrize('declaration', [
    None, 'backtest_total_return', [], {},
    {'metric': 'backtest_total_return'},
    {'metric': 'backtest_total_return', 'direction': 'maximize', 'extra': True},
    {'metric': 'auc', 'direction': 'maximize'},
    {'metric': 'backtest_total_return', 'direction': 'sideways'},
    {'metric': '{metric}', 'direction': 'maximize'},
    {'metric': 'backtest_total_return', 'direction': '{direction}'},
    {'metric': 1, 'direction': 'maximize'},
    {'metric': 'backtest_total_return', 'direction': True},
])
def test_invalid_objective_is_an_error(declaration):
    yaml_dict = _configuration()
    yaml_dict['sfd']['manifest']['objective'] = declaration
    result = validate(yaml_dict)
    assert not result.valid
    assert any(error.path == 'sfd.manifest.objective' for error in result.errors)
    with pytest.raises(ValueError, match='objective'):
        build_manifest(yaml_dict)


@pytest.mark.parametrize('location', ['root', 'uel'])
def test_misplaced_objective_is_an_error(location):
    yaml_dict = _configuration()
    container = yaml_dict if location == 'root' else yaml_dict['uel']
    container['objective'] = {'metric': 'backtest_total_return', 'direction': 'maximize'}
    result = validate(yaml_dict)
    assert not result.valid
    path = 'objective' if location == 'root' else 'uel.objective'
    assert any(error.path == path for error in result.errors)
    with pytest.raises(ValueError, match=r'only at sfd.manifest.objective'):
        build_manifest(yaml_dict)


def test_rule_based_objective_is_rejected():
    yaml_dict = _configuration('rule_based')
    yaml_dict['sfd']['manifest']['objective'] = {'metric': 'backtest_total_return', 'direction': 'maximize'}
    result = validate(yaml_dict)
    assert not result.valid
    assert any('ML manifest' in error.message for error in result.errors)
    with pytest.raises(ValueError, match='ML manifest'):
        build_manifest(yaml_dict)


@pytest.mark.parametrize('direction', ['maximize', 'minimize'])
def test_yaml_binds_supported_reducers_without_rewriting_declarations(direction):
    yaml_dict = _objective_configuration(direction)
    yaml_dict['uel']['pruning_strategies'] = [
        {'type': 'correlation'},
        {'type': 'focus', 'params': {'breakthrough_threshold': 0.0}},
        {'type': 'sanity'},
        {'type': 'saturation'},
        {'type': 'budget', 'params': {'trim_strategy': 'worst_first'}},
        {'type': 'budget', 'params': {'trim_strategy': 'random'}},
    ]
    before = copy.deepcopy(yaml_dict)
    assert validate(yaml_dict).valid
    reducers = build_pruning_strategies(yaml_dict)
    for reducer in reducers[:-1]:
        assert reducer._metric == _COLUMN
    for index in (0, 1, 4):
        assert reducers[index]._maximize is (direction == 'maximize')
    assert not hasattr(reducers[2], '_maximize')
    assert not hasattr(reducers[3], '_maximize')
    assert reducers[-1]._metric is None
    assert yaml_dict == before


@pytest.mark.parametrize('params', [
    {'metric': 'backtest_total_return'},
    {'metric': None},
    {'maximize': False},
    {'maximize': 'true'},
    {'maximize': 1},
])
def test_conflicting_reducer_binding_fails_validation_and_compilation(params):
    yaml_dict = _objective_configuration()
    yaml_dict['uel']['pruning_strategies'] = [{'type': 'correlation', 'params': params}]
    result = validate(yaml_dict)
    assert not result.valid
    assert any('conflicts with the declared objective' in error.message for error in result.errors)
    with pytest.raises(ValueError, match='conflicts with the declared objective'):
        build_pruning_strategies(yaml_dict)


@pytest.mark.parametrize('direction', ['maximize', 'minimize'])
def test_matching_explicit_reducer_binding_is_accepted(direction):
    yaml_dict = _objective_configuration(direction)
    yaml_dict['uel']['pruning_strategies'] = [{
        'type': 'correlation',
        'params': {'metric': _COLUMN, 'maximize': direction == 'maximize'},
    }]
    assert validate(yaml_dict).valid
    reducer = build_pruning_strategies(yaml_dict)[0]
    assert reducer._metric == _COLUMN
    assert reducer._maximize is (direction == 'maximize')


def test_random_budget_preserves_its_metric_free_behavior():
    yaml_dict = _objective_configuration('minimize')
    yaml_dict['uel']['pruning_strategies'] = [{
        'type': 'budget',
        'params': {'trim_strategy': 'random', 'metric': 'auc', 'maximize': True},
    }]
    assert validate(yaml_dict).valid
    reducer = build_pruning_strategies(yaml_dict)[0]
    assert reducer._trim_strategy == 'random'
    assert reducer._metric == 'auc'
    assert reducer._maximize is True


def test_no_objective_preserves_classification_scorer_and_reducer_metric():
    yaml_dict = _configuration()
    yaml_dict['uel']['pruning_strategies'] = [{'type': 'correlation', 'params': {'metric': 'auc', 'maximize': False}}]
    before = copy.deepcopy(yaml_dict)
    assert validate(yaml_dict).valid
    manifest = build_manifest(yaml_dict)
    assert isinstance(manifest, MLManifest)
    assert manifest.objective is None
    assert manifest.prediction_calibration_config is not None
    assert manifest.prediction_calibration_config.threshold_params['metric'].__name__ == 'balanced_metric'
    reducer = build_pruning_strategies(yaml_dict)[0]
    assert reducer._metric == 'auc'
    assert reducer._maximize is False
    assert yaml_dict == before


def test_lightgbm_model_objective_remains_a_parameter():
    yaml_dict = _configuration('lightgbm_binary')
    before = copy.deepcopy(yaml_dict['sfd']['params']['objective'])
    assert validate(yaml_dict).valid
    manifest = build_manifest(yaml_dict)
    assert isinstance(manifest, MLManifest)
    assert manifest.objective is None
    assert yaml_dict['sfd']['params']['objective'] == before

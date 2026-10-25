import copy
import inspect
from functools import partial

import numpy as np
import pytest

from limen.backtest.execution_events import with_predictions
from limen.backtest.trade_execution import trade_execution
from limen.sfd.reference_architecture import logreg_binary
from limen.yaml import validate
from tests.test_experiment_objective import COLUMN, _prepared, objective_config


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


def _rule_config(direction):
    config = objective_config(direction, calibrated=False)
    manifest = config['sfd']['manifest']
    manifest.update(type='rule_based', reference_architecture='limen.sfd.reference_architecture.rule_based.rule_based',
                    strategy={'conditions': [{'id': 'up', 'type': 'relative', 'column': 'close', 'operator': '>', 'other_column': 'open'}], 'entry': 'up'})
    for key in ('target', 'scaler', 'features', 'feature_shims'):
        manifest.pop(key, None)
    config['sfd']['params'] = {'sharpe_std_threshold': [0.5]}
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

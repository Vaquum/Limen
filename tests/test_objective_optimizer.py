from dataclasses import replace
from functools import partial
from unittest.mock import patch

import numpy as np
import pytest

from limen.backtest.execution_events import with_predictions
from limen.backtest.trade_execution import trade_execution
from limen.calibration import grid_threshold_optimizer
from tests.test_experiment_objective import COLUMN, _prepared, objective_config


def _decorated(y_val, val_proba, **params):
    return grid_threshold_optimizer(y_val, val_proba, **params)


class _Optimizer:
    def __call__(self, y_val, val_proba, **params):
        return grid_threshold_optimizer(y_val, val_proba, **params)


def _non_grid(y_val, val_proba, *, metric, _objective_maximize, **params):
    candidates = [2.0, 0.7, 0.3]
    scores = [metric(y_val, (val_proba >= threshold).astype(np.int8)) for threshold in candidates]
    best = max(scores) if _objective_maximize else min(scores)
    return candidates[scores.index(best)], best


@pytest.mark.parametrize('direction', ['maximize', 'minimize'])
@pytest.mark.parametrize('calibrated', [False, True])
@pytest.mark.parametrize('optimizer', [partial(grid_threshold_optimizer), _decorated, _Optimizer(), _non_grid])
def test_recorded_custom_objective_optimizer(direction, calibrated, optimizer):
    config = objective_config(direction, calibrated=True)
    manifest, params, data = _prepared(config)
    params['use_calibration'] = calibrated
    manifest.prediction_calibration_config = replace(manifest.prediction_calibration_config, threshold_func=optimizer)
    result = manifest.run_model(data, params)
    probabilities = result['_model'].predict({'x_test': data['x_val']})['_probs']
    candidates = [2.0, 0.7, 0.3] if optimizer is _non_grid else [2.0, 0.6, 0.55, 0.5, 0.45, 0.4]
    context = data['_trade_context']
    expected = [trade_execution(with_predictions(context.partitions[1], (probabilities >= threshold).astype(np.int8)), context.policy).metrics['total_return'] for threshold in candidates]
    optimum = max(expected) if direction == 'maximize' else min(expected)
    assert result[COLUMN] == pytest.approx(optimum)
    assert result['val_score'] == pytest.approx(optimum)
    assert result['optimal_threshold'] == pytest.approx(candidates[expected.index(optimum)])
    if direction == 'maximize':
        assert result['optimal_threshold'] == 2.0
    else:
        assert optimum < 0


@pytest.mark.parametrize('binding', ['metric', '_objective_maximize', 'y_val', 'val_proba'])
def test_objective_optimizer_refuses_bound_protected_inputs(binding):
    manifest, params, data = _prepared(objective_config())
    optimizer = partial(grid_threshold_optimizer, **{binding: object()})
    manifest.prediction_calibration_config = replace(manifest.prediction_calibration_config, threshold_func=optimizer)
    with patch('limen.sfd.reference_architecture.logreg_binary.LogRegBinary.train', side_effect=AssertionError('must refuse before fitting')), pytest.raises(ValueError, match='conflicts'):
        manifest.run_model(data, params)
    params['use_threshold'] = False
    assert np.isfinite(manifest.run_model(data, params)[COLUMN])


def _drops_metric(y_val, val_proba, **params):
    return 0.5, 0.0


def _false_score(y_val, val_proba, **params):
    threshold, score = _decorated(y_val, val_proba, **params)
    return threshold, score + 1


def _bad_result(y_val, val_proba, **params):
    return 0.5, float('nan')


@pytest.mark.parametrize('optimizer', [_drops_metric, _false_score, _bad_result])
def test_objective_optimizer_refuses_unused_or_false_objective(optimizer):
    manifest, params, data = _prepared(objective_config())
    manifest.prediction_calibration_config = replace(manifest.prediction_calibration_config, threshold_func=optimizer)
    with pytest.raises(ValueError, match='Objective optimizer'):
        manifest.run_model(data, params)


def test_objective_optimizer_preserves_partial_keyword_precedence():
    manifest, params, data = _prepared(objective_config('minimize'))
    config = manifest.prediction_calibration_config
    manifest.prediction_calibration_config = replace(config, threshold_func=partial(grid_threshold_optimizer, threshold_min=0.1, threshold_max=0.9))
    actual = manifest.run_model(data, params)
    ordinary, ordinary_params, ordinary_data = _prepared(objective_config('minimize'))
    expected = ordinary.run_model(ordinary_data, ordinary_params)
    assert actual['optimal_threshold'] == expected['optimal_threshold']
    assert actual[COLUMN] == expected[COLUMN]


def test_inactive_objective_optimizer_is_never_called():
    def fail(*args, **kwargs):
        raise AssertionError('inactive optimizer called')
    manifest, params, data = _prepared(objective_config())
    config = manifest.prediction_calibration_config
    manifest.prediction_calibration_config = replace(config, threshold_func=partial(fail, metric=object()))
    params['use_threshold'] = False
    assert np.isfinite(manifest.run_model(data, params)[COLUMN])

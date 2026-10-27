import copy
import json
from types import SimpleNamespace

import polars as pl
import pytest

from limen.experiment.msq import MSQ
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import GridStrategy
from limen.experiment.reducer import CorrelationReducer
from limen.log._experiment_parameter_importance import experiment_parameter_importance
from limen.yaml.compiler import build_pruning_strategies
from tests.test_parameter_importance import METRIC

pytest_plugins = ['tests.test_parameter_importance']


def queue(loop):
    domain = ParamDomain(loop.params)
    return MSQ(GridStrategy(domain), domain)


def reducer(**kwargs):
    return CorrelationReducer(metric=METRIC, analyzer='random_forest', min_observations=20,
                              n_repeats=2, random_state=7, min_model_score=.001, **kwargs)


def test_real_sweep_forest_suggestions_and_checkpoint(recorded_importance):
    loop, _, _ = recorded_importance
    msq = queue(loop)
    original = copy.deepcopy(msq.get_state())
    analysis = experiment_parameter_importance(SimpleNamespace(_captured_log=loop.experiment_log),
                    METRIC, parameters=msq.domain_keys, random_state=7, n_repeats=2)
    assert analysis['baseline_score'][0] > .001
    strategy = reducer()
    actions = strategy.analyze_and_intervene(loop.experiment_log, msq)
    assert actions and {'force_row_wise', 'random_state'} <= {action['param'] for action in actions}
    assert all(action['action'] == 'suggest' and action['op'] == 'keep_is' for action in actions)
    for action in actions:
        row = analysis.filter(pl.col('parameter') == action['param']).row(0, named=True)
        assert max(0, row['importance_mean']) + 2 * row['importance_std'] < .05
        assert action['value'] in loop.params[action['param']]
    assert msq.get_state() == original
    restored = reducer()
    restored.set_state(json.loads(json.dumps(strategy.get_state())))
    assert restored.analyze_and_intervene(loop.experiment_log, msq) == []
    assert reducer().analyze_and_intervene(loop.experiment_log, msq) == actions


def test_multi_metric_requires_agreement_and_rejects_weak_surrogates(recorded_importance):
    loop, _, _ = recorded_importance
    msq = queue(loop)
    measured = loop.experiment_log.with_columns(pl.col(METRIC).alias('opposing_direction_control'))
    actions = reducer(prune_threshold=1, metrics={'opposing_direction_control': False}).analyze_and_intervene(measured, msq)
    for action in actions:
        scores = measured.group_by(action['param']).agg(pl.col(METRIC).mean())[METRIC]
        assert scores.min() == scores.max()
    strict = CorrelationReducer(metric=METRIC, analyzer='random_forest', min_model_score=1, n_repeats=2)
    assert strict.analyze_and_intervene(measured, msq) == []


def test_forest_admission_and_failed_rows(recorded_importance):
    loop, _, _ = recorded_importance
    msq = queue(loop)
    clean = loop.experiment_log
    failed = clean.head(1).with_columns(pl.lit('retained failure control').alias('strict_mode_error'),
                                      pl.lit(None).cast(pl.Float64).alias(METRIC))
    expected = reducer().analyze_and_intervene(clean, msq)
    assert reducer().analyze_and_intervene(pl.concat([clean, failed], how='diagonal_relaxed'), msq) == expected
    assert reducer().analyze_and_intervene(clean.head(19), msq) == []
    with pytest.raises(ValueError, match='columns'):
        reducer(metrics={'missing_metric': True}).analyze_and_intervene(clean, msq)
    with pytest.raises(ValueError, match='columns'):
        reducer().analyze_and_intervene(clean.drop(METRIC), msq)


@pytest.mark.parametrize('settings', [
    {'analyzer': 'guess'}, {'analyzer': 'correlation', 'metrics': {'loss': False}},
    {'analyzer': 'random_forest', 'n_repeats': 1},
    {'analyzer': 'random_forest', 'metrics': {'loss': 'minimize'}},
    {'analyzer': 'random_forest', 'metrics': {METRIC: False}},
    {'analyzer': 'random_forest', 'min_model_score': 0},
])
def test_invalid_forest_controls_refuse(settings):
    with pytest.raises(ValueError):
        CorrelationReducer(metric=METRIC, **settings)


def test_yaml_factory_admits_forest_controls():
    strategies = build_pruning_strategies({'uel': {'pruning_strategies': [{
        'type': 'correlation', 'params': {'metric': METRIC, 'analyzer': 'random_forest',
        'metrics': {'validation_rmse': False}, 'n_repeats': 3, 'random_state': 7}}]}})
    assert len(strategies) == 1 and strategies[0]._forest.metrics == {METRIC: True, 'validation_rmse': False}

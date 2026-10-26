import json
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from limen.experiment import UniversalExperimentLoop
from limen.experiment.errors import StrictModeError
from limen.experiment.feedback_controller import FeedbackController
from limen.experiment.msq import MSQ
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import GridStrategy
from limen.experiment.reducer import BudgetReducer, CorrelationReducer, SanityReducer
from limen.yaml import CompiledSFD
from tests.test_record_model_outputs import _config, recorded_source


@pytest.fixture(scope='module')
def recorded_mixed_logs(tmp_path_factory):
    root = tmp_path_factory.mktemp('recorded-reducer-identities')
    results = {}
    for name, values, fail_int in (
        ('mixed', [1, 2.5], False), ('correlation', [1, 0.25], False),
        ('failed', [1, 2.5], True), ('ambiguous', [1, 1.0, 2.5], True),
    ):
        manifest = CompiledSFD(_config('logreg')).manifest()
        params = {'C': values, 'max_iter': [1000], 'solver': ['lbfgs', 'liblinear']}
        domain = ParamDomain(params)

        def model(data, round_params, *, native_manifest=manifest, fail=fail_int):
            result = native_manifest.run_model(data, round_params)
            probabilities = result['_model'].model.predict_proba(data['x_val'].to_numpy())[:, 1]
            result['validation_brier'] = float(np.mean((probabilities - data['y_val'].to_numpy().ravel()) ** 2))
            if fail and type(round_params['C']) is int:
                raise StrictModeError('Injected review control after a recorded model fit')
            return result

        sfd = SimpleNamespace(__name__=__name__, params=lambda values=params: values, prep=manifest.prepare_data, model=model)
        loop = UniversalExperimentLoop(sfd=sfd, data=recorded_source(), experiment_dir=root / name,
            search_strategy=GridStrategy(domain), checkpoint_interval=1, feedback_interval=1)
        loop.run('recorded_reducer_identity', n_permutations=domain.total_combinations,
            prep_each_round=True, progress_bar=False,
            recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True))
        assert loop.experiment_log.height == len(values) * 2 and loop.experiment_log['C'].dtype == pl.Float64
        results[name] = loop.experiment_log, params
    return results


def _queue(params):
    domain = ParamDomain(params)
    return MSQ(GridStrategy(domain), domain), domain


def test_sanity_prunes_the_original_integer_from_a_real_coerced_log(recorded_mixed_logs):
    log, params = recorded_mixed_logs['failed']
    queue, domain = _queue(params)
    reducer = SanityReducer(metric='validation_brier', nan_threshold=0.75, min_observations=2)
    actions = reducer.analyze_and_intervene(log, queue)
    removal = next(action for action in actions if action['param'] == 'C')
    assert removal['value'] == 1 and type(removal['value']) is int
    FeedbackController._apply_intervention(queue, removal)
    assert domain.values_for('C') == [2.5]
    restored = SanityReducer(metric='validation_brier', nan_threshold=0.75, min_observations=2)
    restored.set_state(json.loads(json.dumps(reducer.get_state())))
    assert not restored.analyze_and_intervene(log, queue)


def test_budget_prunes_the_original_integer_from_real_recorded_scores(recorded_mixed_logs):
    log, params = recorded_mixed_logs['mixed']
    queue, domain = _queue(params)
    means = log.group_by('C').agg(pl.col('validation_brier').mean())
    first = means.filter(pl.col('C') == 1)['validation_brier'][0]
    other = means.filter(pl.col('C') == 2.5)['validation_brier'][0]
    reducer = BudgetReducer(trim_strategy='worst_first', metric='validation_brier', maximize=first < other)
    actions = reducer._trim_worst_first(log, queue, 0)
    removal = next(action for action in actions if action['param'] == 'C')
    assert removal['value'] == 1 and type(removal['value']) is int
    FeedbackController._apply_intervention(queue, removal)
    assert domain.values_for('C') == [2.5]


def test_correlation_prunes_the_original_integer_from_real_recorded_scores(recorded_mixed_logs):
    log, params = recorded_mixed_logs['correlation']
    queue, domain = _queue(params)
    means = log.group_by('C').agg(pl.col('validation_brier').mean())
    first = means.filter(pl.col('C') == 1)['validation_brier'][0]
    other = means.filter(pl.col('C') == 0.25)['validation_brier'][0]
    reducer = CorrelationReducer(metric='validation_brier', method='pearson', n_boot=30,
        min_observations=2, sign_stability_threshold=0, negative_correlation_threshold=0,
        maximize=first < other)
    actions = reducer.analyze_and_intervene(log, queue)
    removal = next(action for action in actions if action['param'] == 'C' and action.get('action') is None)
    assert removal['value'] == 1 and type(removal['value']) is int
    FeedbackController._apply_intervention(queue, removal)
    assert domain.values_for('C') == [0.25]


def test_ambiguous_coerced_numeric_groups_refuse_an_exact_pruning_guess(recorded_mixed_logs):
    log, params = recorded_mixed_logs['ambiguous']
    queue, domain = _queue(params)
    before = domain.get_state(), domain.version
    with pytest.raises(ValueError, match='ambiguously represents distinct execution identities'):
        SanityReducer(metric='validation_brier').analyze_and_intervene(log, queue)
    assert (domain.get_state(), domain.version) == before


@pytest.mark.parametrize('kind', ['sanity', 'correlation'])
def test_reducer_checkpoint_pairs_preserve_distinct_numeric_identities(kind):
    reducer = SanityReducer(metric='score') if kind == 'sanity' else CorrelationReducer(metric='score')
    key = 'removed' if kind == 'sanity' else 'applied'
    state = {key: [['C', 1], ['C', 1.0]], 'suggested': []}
    reducer.set_state(state)
    actual = json.loads(json.dumps(reducer.get_state()))
    assert actual == state and [type(value) for _, value in actual[key]] == [int, float]

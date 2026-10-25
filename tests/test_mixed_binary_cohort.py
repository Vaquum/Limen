import copy
import hashlib
import json
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from limen import Cohort
from limen.cli.commands.run import run_experiment
from limen.data import HistoricalData
from limen.inference import Trainer
from limen.inference.sensor import BarPrediction
from limen.sfd.reference_architecture import LightGBMBinary, LogRegBinary
from tests.test_cohort_rounds import _FIXTURE, _after_training, _yaml


@pytest.fixture(scope='module')
def mixed_campaigns(tmp_path_factory):
    root = tmp_path_factory.mktemp('mixed-binary-cohort')
    recorded = HistoricalData().get_any_file(str(_FIXTURE))
    campaigns = []
    with patch('requests.sessions.Session.request', side_effect=AssertionError('recorded inputs only')):
        for architecture in ('logreg_binary', 'lightgbm_binary'):
            directory = root / architecture
            source = _yaml(directory, 42, narrowed=True).replace('limen.sfd.reference_architecture.logreg_binary', 'limen.sfd.reference_architecture.' + architecture)
            if architecture == 'lightgbm_binary':
                source = source.replace('C: [1.0]', 'n_estimators: [10]\n    num_leaves: [5]\n    n_jobs: [1]').replace('max_iter: [100]', 'verbosity: [-1]')
            path = root / f'{architecture}.yaml'
            path.write_text(source)
            assert run_experiment(path, progress_bar=False)
            trainer = Trainer(directory, data=recorded)
            sensors = trainer.train(pl.read_csv(directory / 'results.csv')['id'].to_list())
            campaigns.append((directory, sensors[0]))
    assert type(campaigns[0][1]._model) is LogRegBinary
    assert type(campaigns[1][1]._model) is LightGBMBinary
    return campaigns, _after_training(recorded)


def _cohort(campaigns, *, reverse=False):
    selected = list(reversed(campaigns)) if reverse else campaigns
    cohort = Cohort(rounds=[(path, sensor.permutation_id) for path, sensor in selected])
    assert cohort.cohort_id is cohort.aggregation_mode is cohort.supports_probabilities is None
    cohort.set_members([sensor for _, sensor in reversed(selected)])
    return cohort


def test_recorded_mixed_cohort_lifecycle(mixed_campaigns):
    campaigns, bars = mixed_campaigns
    cohort = _cohort(campaigns)
    reversed_cohort = _cohort(campaigns, reverse=True)
    assert cohort.cohort_id == reversed_cohort.cohort_id
    assert cohort.aggregation_mode == 'probability_weighted'
    assert cohort.supports_probabilities is True
    member_outputs = [sensor.predict_all(bars) for _, sensor in campaigns]
    actual = cohort.predict_all(bars)
    assert actual == reversed_cohort.predict_all(bars)
    compared = 0
    for index, result in enumerate(actual):
        operands = [outputs[index] for outputs in member_outputs]
        if all(bar.reason is None for bar in operands):
            expected = float(np.mean([bar.probability for bar in operands]))
            assert result.probability == expected
            assert result.prediction == int(expected > 0.5)
            assert result.trade_contract_digest is result.available_at_ns is None
            compared += 1
    assert compared > 50
    assert cohort.predict(bars) == actual[-1]
    policy = vars(cohort)['_mixed_policy']
    payload = cohort._round_sources.identity(cohort.aggregation_mode) | {'binary_contracts': policy}
    assert cohort.cohort_id == 'sha256:' + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class _VoteModel:
    inference_output = ('binary', False)
    prediction_mode = 'binary'
    deterministic = True

    def __init__(self, original):
        self.original = original

    def predict(self, data):
        return {'_preds': self.original.predict(data)['_preds']}


@pytest.mark.parametrize('reverse', [False, True])
def test_recorded_mixed_cohort_output_contracts(mixed_campaigns, reverse):
    campaigns, bars = mixed_campaigns
    vote = copy.copy(campaigns[1][1])
    vote._model = _VoteModel(vote._model)
    selected = [campaigns[0], (campaigns[1][0], vote)]
    cohort = _cohort(selected, reverse=reverse)
    assert cohort.aggregation_mode == 'majority_vote'
    outputs = [member.predict(bars) for member in cohort._members]
    assert all(output.reason is None for output in outputs)
    assert cohort.predict(bars).prediction == int(np.mean([output.prediction for output in outputs]) > 0.5)
    assert cohort.predict(bars).probability is None
    promising = next(index for index, member in enumerate(cohort._members) if member is campaigns[0][1])
    malformed = list(outputs)
    malformed[promising] = replace(malformed[promising], probability=None)
    with pytest.raises(ValueError, match='promised probability'):
        cohort._aggregate_bar_predictions(malformed)
    for value in (None, float('nan'), '1', 0.25):
        malformed = list(outputs)
        malformed[promising] = replace(malformed[promising], prediction=value)
        with pytest.raises(ValueError, match='binary decisions'):
            cohort._aggregate_bar_predictions(malformed)
    excluded = list(outputs)
    excluded[promising] = BarPrediction(datetime=None, prediction=None, probability=None, reason='sensor-error')
    assert cohort._aggregate_bar_predictions(excluded).prediction == outputs[1 - promising].prediction


def test_recorded_mixed_cohort_refusals(mixed_campaigns):
    campaigns, bars = mixed_campaigns
    cohort = _cohort(campaigns)
    state = (cohort.cohort_id, cohort.member_ids, cohort.manifest_id, cohort.aggregation_mode, vars(cohort)['_mixed_policy'], cohort._members)
    for fault in ('manifest', 'architecture', 'declaration'):
        invalid = copy.copy(campaigns[1][1])
        if fault == 'manifest':
            invalid.manifest_id = 'wrong-source'
        elif fault == 'architecture':
            invalid._round_params = {**invalid.round_params, 'architecture': 'limen.sfd.reference_architecture.ridge_regressor'}
        else:
            invalid._model = object()
        with pytest.raises(ValueError):
            cohort.set_members([campaigns[0][1], invalid])
        assert state == (cohort.cohort_id, cohort.member_ids, cohort.manifest_id, cohort.aggregation_mode, vars(cohort)['_mixed_policy'], cohort._members)
    assert cohort.predict(bars).reason is None


def test_recorded_mixed_cohort_freeze(mixed_campaigns):
    from limen.experiment._holdout_freeze import owned_copy, state_digest

    campaigns, bars = mixed_campaigns
    cohort = _cohort(campaigns)
    snapshot = owned_copy(cohort)
    second = owned_copy(snapshot)
    assert snapshot.predict_all(bars) == second.predict_all(bars) == cohort.predict_all(bars)
    digest = state_digest(snapshot)
    vars(cohort)['_mixed_policy'] = ()
    assert state_digest(snapshot) == digest
    vars(second)['_mixed_policy'] = ()
    assert state_digest(second) != digest

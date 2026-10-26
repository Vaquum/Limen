import copy
import hashlib
import inspect
import json
import shutil
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
from limen.sfd.reference_architecture import lightgbm_binary, logreg_binary
from limen.yaml import CompiledSFD, build_search_strategy, parse, validate
from limen.experiment import UniversalExperimentLoop
from tests.test_cohort_rounds import _FIXTURE, _after_training, _yaml
from tests.test_holdout_assessment import assessment_case

__all__ = ['assessment_case']


@pytest.fixture(scope='module')
def mixed_campaigns(tmp_path_factory):
    root = tmp_path_factory.mktemp('mixed-binary-cohort')
    recorded = HistoricalData().get_any_file(str(_FIXTURE))
    campaigns = []
    with patch('requests.sessions.Session.request', side_effect=AssertionError('recorded inputs only')):
        for architecture in ('logreg_binary', 'lightgbm_binary'):
            directory = root / 'results/dev' / architecture
            source = _yaml(directory, 42, narrowed=True).replace('limen.sfd.reference_architecture.logreg_binary', 'limen.sfd.reference_architecture.' + architecture)
            if architecture == 'lightgbm_binary':
                source = source.replace('C: [1.0]', 'n_estimators: [10]\n    num_leaves: [5]\n    n_jobs: [1]').replace('max_iter: [100]', 'verbosity: [-1]')
            path = root / f'{architecture}.yaml'
            path.write_text(source)
            assert run_experiment(path, results_base=root, progress_bar=False)
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


def _dispatch(data, architecture, **params):
    function = logreg_binary if architecture.endswith('logreg_binary') else lightgbm_binary
    return function(data, **{key: value for key, value in params.items() if key in inspect.signature(function).parameters})


@pytest.fixture(scope='module')
def same_directory(tmp_path_factory):
    root = tmp_path_factory.mktemp('mixed-selected-ids')
    reference, errors = parse(_yaml(root / 'results/dev/mixed', 42, narrowed=True))
    assert not errors
    reference['metadata']['name'] = 'mixed_dispatch'
    reference['sfd']['params'].update(architecture=['limen.sfd.reference_architecture.logreg_binary', 'limen.sfd.reference_architecture.lightgbm_binary'], n_estimators=[10], n_jobs=[1])
    reference['uel']['n_permutations'] = 2
    assert [error.path for error in validate(reference).errors] == ['sfd.params.architecture', 'sfd.params.n_estimators']
    original = CompiledSFD.manifest
    def manifest(sfd):
        result = original(sfd)
        if sfd._yaml['metadata']['name'] == 'mixed_dispatch':
            result.with_reference_architecture(_dispatch)
        return result
    with patch.object(CompiledSFD, 'manifest', manifest):
        directory = root / 'results/dev/mixed'
        loop = UniversalExperimentLoop(sfd=CompiledSFD(reference), data=HistoricalData().get_any_file(str(_FIXTURE)),
            experiment_dir=directory, yaml_reference=reference, search_strategy=build_search_strategy(reference))
        loop.run('mixed_dispatch', n_permutations=2, prep_each_round=True, progress_bar=False,
            record_execution=True, record_model_outputs=True,
            recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True))
        ids = pl.read_csv(directory / 'results.csv')['id'].to_list()
        sensors = Trainer(directory).train(ids)
        assert {type(sensor._model) for sensor in sensors} == {LogRegBinary, LightGBMBinary}
        yield directory, ids, sensors


@pytest.mark.parametrize('selection', ['selected', 'default', 'selector', 'rounds'])
def test_recorded_mixed_cohort_lifecycle_selection(same_directory, selection):
    directory, ids, sensors = same_directory
    options = {'permutation_ids': ids} if selection == 'selected' else {'selector': 'all'} if selection == 'selector' else {}
    cohort = Cohort(rounds=[(directory, pid) for pid in ids]) if selection == 'rounds' else Cohort(experiment_log_path=str(directory), **options)
    cohort.set_members(list(reversed(sensors)))
    expected = Cohort(experiment_log_path=str(directory), permutation_ids=ids)
    expected.set_members(sensors)
    assert cohort.cohort_id == expected.cohort_id


def test_recorded_mixed_cohort_lifecycle_relocation_and_alias(mixed_campaigns, tmp_path):
    campaigns, bars = mixed_campaigns
    expected = _cohort(campaigns)
    relocated = []
    for index, (directory, sensor) in enumerate(campaigns):
        path = tmp_path / str(index)
        shutil.copytree(directory, path)
        alias = copy.copy(sensor)
        architecture = alias._yaml_reference['sfd']['manifest']['reference_architecture']
        alias._round_params = {**alias.round_params, 'architecture': architecture + '.' + architecture.rsplit('.', 1)[1]}
        relocated.append((path, alias))
    actual = _cohort(relocated, reverse=True)
    assert actual.cohort_id == expected.cohort_id
    assert actual.predict_all(bars) == expected.predict_all(bars)


def test_recorded_mixed_cohort_lifecycle_folds(tmp_path):
    from tests.fold_components import FIXTURE, FULL, config

    source = tmp_path / 'recorded.parquet'
    history = pl.read_parquet(FIXTURE)
    bars = history.head(384)
    bars.write_parquet(source)
    selections, sensors = [], []
    for architecture in ('logreg_binary', 'lightgbm_binary'):
        reference = config('custom', source)
        reference['metadata']['name'] = 'mixed_fold_' + architecture
        reference['sfd']['manifest']['reference_architecture'] = 'limen.sfd.reference_architecture.' + architecture
        if architecture == 'lightgbm_binary':
            reference['sfd']['params'] = {'n_estimators': [10], 'num_leaves': [5], 'n_jobs': [1]}
        directory = tmp_path / architecture
        loop = UniversalExperimentLoop(sfd=CompiledSFD(reference), data=bars, experiment_dir=directory, yaml_reference=reference, search_strategy=build_search_strategy(reference))
        loop.run('mixed_folds', n_permutations=1, prep_each_round=True, progress_bar=False, recording=FULL, record_execution=True, record_model_outputs=True)
        trainer = Trainer(directory, data=bars)
        pid = next(iter(trainer._round_data))
        sensor = trainer.train([pid], fold=0)[0]
        selections.append((directory, pid, 0))
        sensors.append(sensor)
    cohort = Cohort(rounds=selections)
    cohort.set_members(list(reversed(sensors)))
    reverse = Cohort(rounds=list(reversed(selections)))
    reverse.set_members(sensors)
    assert cohort.cohort_id == reverse.cohort_id
    probe = history.head(480)
    assert cohort.predict_all(probe) == reverse.predict_all(probe)
    expected = float(np.mean([sensor.predict(probe).probability for sensor in sensors]))
    assert cohort.predict(probe).probability == expected


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
    last = cohort.predict(bars)
    assert replace(last, probability=None) == replace(actual[-1], probability=None)
    np.testing.assert_allclose(last.probability, actual[-1].probability, rtol=1e-14)
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


class _InvalidBinary(_VoteModel):
    inference_output = ('binary', True)

    def __init__(self, original, value):
        super().__init__(original)
        self.value = value

    def predict(self, data):
        result = self.original.predict(data)
        return {**result, '_preds': np.full_like(result['_preds'], self.value, dtype=float)}


@pytest.mark.parametrize('value', [0.25, float('nan')])
def test_recorded_mixed_cohort_output_contracts_sensor_boundary(mixed_campaigns, value):
    campaigns, bars = mixed_campaigns
    invalid = copy.copy(campaigns[1][1])
    invalid._model = _InvalidBinary(invalid._model, value)
    cohort = _cohort([campaigns[0], (campaigns[1][0], invalid)])
    if np.isfinite(value):
        assert invalid.predict(bars).reason is None
        with pytest.raises(ValueError, match='binary decisions'):
            cohort.predict(bars)
    else:
        assert invalid.predict(bars).reason == 'sensor-error'
        assert cohort.predict(bars).probability == campaigns[0][1].predict(bars).probability


@pytest.mark.parametrize('reverse', [False, True])
def test_recorded_mixed_cohort_output_contracts_excluded_middle(mixed_campaigns, same_directory, reverse):
    campaigns, bars = mixed_campaigns
    directory, _, sensors = same_directory
    selected = [campaigns[0], (directory, sensors[0]), campaigns[1]]
    cohort = _cohort(selected, reverse=reverse)
    outputs = [member.predict(bars) for member in cohort._members]
    outputs[1] = BarPrediction(datetime=None, prediction=None, probability=None, reason='sensor-error')
    expected = float(np.mean([outputs[0].probability, outputs[2].probability]))
    assert cohort._aggregate_bar_predictions(outputs).probability == expected
    for index in (0, 2):
        invalid = list(outputs)
        invalid[index] = replace(invalid[index], probability=None)
        with pytest.raises(ValueError, match='promised probability'):
            cohort._aggregate_bar_predictions(invalid)


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
    for fault in ('manifest', 'architecture', 'unresolvable_architecture', 'declaration'):
        invalid = copy.copy(campaigns[1][1])
        if fault == 'manifest':
            invalid.manifest_id = 'wrong-source'
        elif fault == 'architecture':
            invalid._round_params = {**invalid.round_params, 'architecture': 'limen.sfd.reference_architecture.ridge_regressor'}
        elif fault == 'unresolvable_architecture':
            invalid._round_params = {**invalid.round_params, 'architecture': 'limen.sfd.reference_architecture.missing'}
        else:
            invalid._model = object()
        with pytest.raises(ValueError):
            cohort.set_members([campaigns[0][1], invalid])
        assert state == (cohort.cohort_id, cohort.member_ids, cohort.manifest_id, cohort.aggregation_mode, vars(cohort)['_mixed_policy'], cohort._members)
    assert cohort.predict(bars).reason is None


@pytest.mark.parametrize('fault', ['missing', 'blank', 'wrong'])
def test_recorded_mixed_cohort_refusals_legacy_identity(same_directory, tmp_path, fault):
    directory, ids, sensors = same_directory
    path = tmp_path / 'source'
    shutil.copytree(directory, path)
    if fault == 'missing':
        metadata = json.loads((path / 'metadata.json').read_text())
        metadata.pop('manifest_id')
        (path / 'metadata.json').write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match='manifest_id'):
            Cohort(experiment_log_path=str(path), permutation_ids=ids)
        return
    cohort = Cohort(experiment_log_path=str(path), permutation_ids=ids)
    changed = copy.copy(sensors[0])
    changed.manifest_id = '' if fault == 'blank' else 'wrong-source'
    with pytest.raises(ValueError, match='manifest_id'):
        cohort.set_members([changed, sensors[1]])
    assert cohort.cohort_id is cohort.aggregation_mode is cohort.supports_probabilities is None
    assert cohort._members == []


def test_recorded_mixed_cohort_freeze(assessment_case):
    from limen.experiment._holdout_freeze import owned_copy, state_digest
    from tests.test_holdout_assessment import _freeze, _report

    root, _, trained, _ = assessment_case
    campaigns = [(root / name, members[0]) for name, members in zip(('logreg', 'lightgbm'), trained[:2], strict=True)]
    bars = pl.read_parquet(campaigns[0][1]._yaml_reference['sfd']['manifest']['data_source']['params']['file_path_or_url']).head(120)
    cohort = _cohort(campaigns)
    snapshot = owned_copy(cohort)
    second = owned_copy(snapshot)
    assert snapshot.predict_all(bars) == second.predict_all(bars) == cohort.predict_all(bars)
    digest = state_digest(snapshot)
    vars(cohort)['_mixed_policy'] = ()
    assert state_digest(snapshot) == digest
    vars(second)['_mixed_policy'] = ()
    assert state_digest(second) != digest
    frozen = _freeze(snapshot)
    report = _report(frozen, assessment_case)
    assert report['state'] == 'completed'
    rows = pl.read_parquet(root / report['rows_path']).filter(pl.col('role') == 'final')
    assert rows.height == 24
    for row in rows.iter_rows(named=True):
        prefix = bars.filter(pl.col('datetime').dt.epoch('ns') <= row['open_ns'])
        outputs = [sensor.predict(prefix) for _, sensor in campaigns]
        expected = float(np.mean([output.probability for output in outputs]))
        assert row['aggregate_probability'] == expected

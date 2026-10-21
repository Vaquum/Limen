import copy
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from limen.cohort import Cohort
from limen.experiment import UniversalExperimentLoop, _round_storage
from limen.inference import ReconstructionError, Trainer
from limen.sfd.reference_architecture.rule_based import RuleBasedStrategy
from limen.yaml import CompiledSFD, build_search_strategy
from tests.test_temporal_parameters import _config, _conditions

FIXTURE = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'


@pytest.fixture(scope='module', params=['default', 'recorded', 'fold'])
def rule_campaign(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp('rule-sensor-' + request.param)
    bars = pl.read_parquet(FIXTURE, n_rows=336)
    source = directory / 'source.parquet'
    bars.head(168).write_parquet(source)
    config = _config(_conditions(), {'hold_bars': [1, 3], 'recent_bars': [1], 'fee': [3.0], 'size': [1.0]})
    manifest = config['sfd']['manifest']
    manifest['data_source']['params']['file_path_or_url'] = str(source)
    if request.param == 'fold':
        manifest.pop('split_dates')
        manifest['split_walk_forward'] = {'n_folds': 2, 'test_bars': 24, 'purge_bars': 2, 'embargo_bars': 1, 'anchored': True}
    config['uel']['n_permutations'] = 2
    output = directory / 'experiment'
    loop = UniversalExperimentLoop(sfd=CompiledSFD(config), data=bars.head(168), experiment_dir=output,
                                   yaml_reference=config, search_strategy=build_search_strategy(config), checkpoint_interval=1)
    options = {} if request.param == 'default' else {'recording': dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True),
                                                   'record_execution': True, 'record_model_outputs': True}
    loop.run('recorded_rule_sensor', n_permutations=2, prep_each_round=True, progress_bar=False, **options)
    trainer = Trainer(output, data=bars.head(168))
    ids = list(trainer._round_data)
    fold = {'fold': 0} if request.param == 'fold' else {}
    sensors = trainer.train(ids, **fold)
    records = [json.loads(line) for line in _round_storage.text(_round_storage.round_path(output)).splitlines()]
    return output, config, bars, trainer, ids, sensors, records


def test_recorded_rule_sensor_promotion(rule_campaign):
    _, config, bars, trainer, ids, sensors, _ = rule_campaign
    assert len(ids) == len(sensors) == 2
    assert sensors[0].round_params != sensors[1].round_params
    assert all(sensor.permutation_id == pid and sensor.manifest_id == trainer._manifest_id for sensor, pid in zip(sensors, ids, strict=True))
    assert all(type(sensor._model) is RuleBasedStrategy and sensor._model.inference_output == ('binary', False) for sensor in sensors)
    before = copy.deepcopy([sensor._model._strategy for sensor in sensors])
    history = bars.slice(168, 120)
    first = sensors[0].predict_all(history)
    other = sensors[1].predict_all(history)
    assert first == sensors[0].predict_all(history)
    assert any(row.reason is None for row in first) and any(row.reason is None for row in other)
    assert [row.prediction for row in first] != [row.prediction for row in other]
    assert all(row.probability is None and row.reason != 'sensor-error' for row in first + other)
    assert [sensor._model._strategy for sensor in sensors] == before
    sensors[0]._yaml_reference['sfd']['manifest']['strategy']['entry'] = 'changed-after-construction'
    assert sensors[0]._model._strategy == before[0]
    sensors[0]._yaml_reference = copy.deepcopy(config)


def test_recorded_rule_sensor_decisions(rule_campaign):
    _, _, bars, _, _, sensors, _ = rule_campaign
    for sensor in sensors:
        history = bars.slice(168, 120)
        manifest = sensor._get_manifest()
        prepared, lookback = manifest.sensor_input_prep(history, sensor._fitted_params, sensor.round_params)
        columns = [name for name in prepared.columns if name not in ('datetime', '__trade_available_at_ns__')]
        valid = prepared.drop_nulls(columns)
        expected = sensor._model.predict({'test': valid, 'strategy': sensor._model._strategy})['_preds'].tolist()
        actual = sensor.predict_all(history)
        rows = [row for row in actual if row.reason is None]
        assert [row.datetime for row in rows] == valid['datetime'].to_list()
        assert [row.prediction for row in rows] == expected
        assert sensor.predict(history) == actual[-1]
        assert lookback == 0 and len(rows) == history.height
        for size in (1, 5, 31):
            window = history.head(size)
            assert sensor.predict(window) == sensor.predict_all(window)[-1]
        matrix = valid.select(columns).to_numpy().astype(np.float64)
        prediction = sensor._model.predict({'x_test': matrix})
        assert prediction['_preds'].tolist() == expected and '_probs' not in prediction


def test_recorded_rule_sensor_consumers(rule_campaign):
    output, config, bars, _, ids, sensors, _ = rule_campaign
    selections = [(output, pid, 0) for pid in ids] if 'split_walk_forward' in config['sfd']['manifest'] else [(output, pid) for pid in ids]
    cohorts = [Cohort(rounds=selections)]
    if len(selections[0]) == 2:
        cohorts.append(Cohort(experiment_log_path=str(output), permutation_ids=ids))
    history = bars.slice(168, 120)
    native = [sensor.predict_all(history) for sensor in sensors]
    expected = [(float(left.prediction) + float(right.prediction)) / 2 > 0.5 for left, right in zip(*native, strict=True)]
    for cohort in cohorts:
        cohort.set_members(list(reversed(sensors)))
        assert cohort.supports_probabilities is False and cohort.aggregation_mode == 'majority_vote'
        actual = cohort.predict_all(history)
        assert [row.prediction for row in actual] == [int(value) for value in expected]
        assert all(row.reason is None and row.probability is None for row in actual)
        identity = cohort.cohort_id
        cohort.set_members(sensors)
        assert cohort.cohort_id == identity and cohort.predict(history) == actual[-1]
    guarded = copy.deepcopy(sensors[0])
    if hasattr(guarded, 'selection_cutoff'):
        guarded_rows = guarded.predict_all(bars.head(168))
        assert all(row.prediction is None for row in guarded_rows)
        assert all(row.reason == 'inside-training-window' for row in guarded_rows)


def test_rule_sensor_evidence_refusals(rule_campaign):
    _, _, bars, trainer, ids, sensors, _ = rule_campaign
    sensor = sensors[0]
    original = copy.deepcopy(sensor._model._strategy)
    for value in (np.zeros(3), np.full((3, len(sensor._model._predicate_ids)), np.nan), np.full((3, len(sensor._model._predicate_ids)), 2.0)):
        with pytest.raises(ValueError, match='finite binary matrix'):
            sensor._model.predict({'x_test': value})
    fresh = RuleBasedStrategy()
    with pytest.raises(ValueError, match='captured selected strategy'):
        fresh.predict({'x_test': np.zeros((1, 1))})
    assert sensor._model._strategy == original
    if hasattr(sensor, 'selection_cutoff'):
        changed = bars.head(168).with_columns((pl.col('close') * 1.01).alias('close'))
        with pytest.raises(ReconstructionError):
            Trainer(trainer._experiment_dir, data=changed).train(ids, fold=0)
    else:
        with pytest.raises(ValueError, match='not found'):
            trainer.train(['not-a-recorded-round'])

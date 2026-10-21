import copy
import json
from pathlib import Path
from datetime import timedelta

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


def _gap_feature(data):
    gap = (pl.col('datetime').dt.day() == 9) & (pl.col('datetime').dt.hour() == 8)
    return data.with_columns(pl.when(gap).then(None).otherwise(pl.col('close').shift(2)).alias('retained_close'))


@pytest.fixture(scope='module', params=['default', 'recorded', 'fold', 'leaves', 'formed', 'common', 'nulls', 'policy', 'sql-window'])
def rule_campaign(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp('rule-sensor-' + request.param)
    bars = pl.read_parquet(FIXTURE, n_rows=336)
    source = directory / 'source.parquet'
    bars.head(168).write_parquet(source)
    config = _config(_conditions(), {'hold_bars': [1, 3], 'recent_bars': [1], 'fee': [3.0], 'size': [1.0]})
    manifest = config['sfd']['manifest']
    config['metadata']['name'] = 'rule_sensor_' + request.param.replace('-', '_')
    if request.param == 'leaves':
        manifest['indicators'] = [{'func': 'limen.indicators.ema', 'params': {'period': '{period}'}, 'include_if': 'use_ema'}]
        config['sfd']['params'].update(period=[3], use_ema=[True])
        manifest['strategy']['conditions'] += [
            {'id': 'above', 'type': 'threshold', 'column': 'close', 'operator': '>', 'value': '{price_floor}'},
            {'id': 'cross', 'type': 'crossover', 'column': 'close', 'other_column': 'ema_{period}'},
            {'id': 'slope', 'type': 'slope', 'column': 'close', 'lookback': 2},
            {'id': 'sql', 'type': 'sql_expr', 'expr': 'close > open'},
            {'id': 'not_above', 'operator': 'not', 'operands': ['above']},
            {'id': 'choice', 'operator': 'or', 'operands': ['cross', 'slope', 'sql', 'not_above']},
            {'id': 'combined', 'operator': 'and', 'operands': ['entry', 'choice']},
        ]
        config['sfd']['params']['price_floor'] = [float(bars['close'].median())]
        manifest['strategy']['entry'] = 'combined'
    if request.param == 'formed':
        manifest['bar_formation'] = {'func': 'limen.data.bars.trade_bars', 'params': {'trade_threshold': 100000}}
        config['sfd']['params']['bar_type'] = ['trade']
    if request.param == 'common':
        manifest['indicators'] = [{'func': 'limen.features.common_strategy_positions.common_strategy_positions', 'params': {'strategy': '{strategy}'}}]
        manifest['strategy'] = {'conditions': [{'id': 'entry', 'type': 'threshold', 'column': 'common_strategy_position', 'operator': '>', 'value': 0}], 'entry': 'entry'}
        config['sfd']['params'] = {'strategy': ['buy_and_hold', 'time_series_momentum'], 'fee': [3.0], 'size': [1.0]}
    if request.param == 'nulls':
        manifest['indicators'] = [{'func': 'tests.test_rule_sensor._gap_feature', 'include_if': 'use_gap'}]
        config['sfd']['params']['use_gap'] = [True]
    if request.param == 'policy':
        manifest['backtest'].update(product={'kind': 'cash_spot', 'instrument': 'BTCUSDT', 'base_currency': 'BTC', 'quote_currency': 'USDT', 'quantity_step': 1e-9, 'min_notional': 0}, source_interval_seconds=3600)
    if request.param == 'sql-window':
        manifest['strategy'] = {'conditions': [{'id': 'entry', 'type': 'sql_expr', 'expr': 'datetime = MAX(datetime)'}], 'entry': 'entry'}
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
    return output, config, bars, trainer, ids, sensors, records, request.param


def test_recorded_rule_sensor_promotion(rule_campaign):
    _, config, bars, trainer, ids, sensors, _, kind = rule_campaign
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
    if kind != 'sql-window':
        assert [row.prediction for row in first] != [row.prediction for row in other]
    assert all(row.probability is None and row.reason != 'sensor-error' for row in first + other)
    assert [sensor._model._strategy for sensor in sensors] == before
    sensors[0]._yaml_reference['sfd']['manifest']['strategy']['entry'] = 'changed-after-construction'
    assert sensors[0]._model._strategy == before[0]
    sensors[0]._yaml_reference = copy.deepcopy(config)


def test_recorded_rule_sensor_decisions(rule_campaign):
    _, _, bars, _, _, sensors, _, kind = rule_campaign
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
        assert len(actual) == prepared.height
        assert len(rows) == valid.height
        if kind == 'formed':
            assert len(actual) < history.height
        if kind == 'nulls':
            assert lookback == 2 and actual[0].reason == 'warm-up'
            assert actual[32].reason == 'null-features'
        if kind == 'policy':
            assert all(row.available_at_ns == int(valid['datetime'].dt.epoch('ns')[index]) + 3600_000_000_000 and row.trade_contract_digest == sensor._trade_digest for index, row in enumerate(rows))
        if kind not in ('fold', 'policy'):
            native = copy.deepcopy(manifest)
            first_day = history['datetime'][0].date()
            last_day = history['datetime'][-1].date() + timedelta(days=1)
            native.set_split_dates(first_day, last_day, last_day, last_day, last_day, last_day)
            raw_prepared = native.prepare_data(history, sensor.round_params)
            oracle = sensor._model.predict({'test': raw_prepared['train'], 'strategy': sensor._model._strategy})['_preds'].tolist()
            assert oracle == expected
        if kind == 'sql-window':
            short = sensor.predict_all(history.head(5))
            assert short[-1].prediction == 1 and actual[4].prediction == 0
        for size in (1, 5, 31):
            window = history.head(size)
            batch = sensor.predict_all(window)
            if batch:
                assert sensor.predict(window) == batch[-1]
            else:
                assert sensor.predict(window).reason == 'warm-up'
        matrix = valid.select(columns).to_numpy().astype(np.float64)
        prediction = sensor._model.predict({'x_test': matrix})
        assert prediction['_preds'].tolist() == expected and '_probs' not in prediction


def test_recorded_rule_sensor_consumers(rule_campaign):
    output, config, bars, _, ids, sensors, _, _ = rule_campaign
    selections = [(output, pid, 0) for pid in ids] if 'split_walk_forward' in config['sfd']['manifest'] else [(output, pid) for pid in ids]
    cohorts = [Cohort(rounds=selections)]
    if len(selections[0]) == 2:
        cohorts.append(Cohort(experiment_log_path=str(output), permutation_ids=ids))
    history = bars.slice(168, 120)
    native = [sensor.predict_all(history) for sensor in sensors]
    expected = [None if left.reason is not None or right.reason is not None else int((float(left.prediction) + float(right.prediction)) / 2 > 0.5) for left, right in zip(*native, strict=True)]
    for cohort in cohorts:
        cohort.set_members(list(reversed(sensors)))
        assert cohort.supports_probabilities is False and cohort.aggregation_mode == 'majority_vote'
        actual = cohort.predict_all(history)
        assert [row.prediction for row in actual] == expected
        assert all(row.reason != 'sensor-error' and row.probability is None and row.available_at_ns is None and row.trade_contract_digest is None for row in actual)
        identity = cohort.cohort_id
        cohort.set_members(sensors)
        assert cohort.cohort_id == identity and cohort.predict(history) == actual[-1]
    guarded = copy.deepcopy(sensors[0])
    if hasattr(guarded, 'selection_cutoff'):
        guarded_rows = guarded.predict_all(bars.head(168))
        assert all(row.prediction is None for row in guarded_rows)
        assert all(row.reason == 'inside-training-window' for row in guarded_rows)


def test_rule_sensor_evidence_refusals(rule_campaign):
    _, _, bars, trainer, ids, sensors, _, _ = rule_campaign
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

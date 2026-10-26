import copy
from datetime import timedelta
from importlib import import_module
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from limen.cohort import Cohort
from limen.data.bars import trade_bars
from limen.experiment import UniversalExperimentLoop, _round_storage
from limen.inference import ReconstructionError, Trainer
from limen.sfd.reference_architecture.rule_based import RuleBasedStrategy, rule_based
from limen.yaml import CompiledSFD, build_search_strategy, validate
from limen.yaml.resolver import resolve
from tests.test_temporal_parameters import _config, _conditions

FIXTURE = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'
KINDS = ['default', 'recorded', 'fold', 'leaves', 'already-formed', 'common', 'nulls', 'policy', 'sql-window', 'neutral-alias', 'logreg-alias', 'optional-off']


@pytest.fixture(scope='module')
def admitted_rule_aliases():
    module = import_module('limen.sfd.reference_architecture.rule_based')
    with pytest.MonkeyPatch.context() as patch:
        for alias in ('rule_sensor_native', 'rule_sensor_logreg'):
            patch.setattr(module, alias, rule_based, raising=False)
            assert resolve(module.__name__ + '.' + alias) is rule_based
        yield


@pytest.fixture(scope='module', params=KINDS)
def rule_campaign(request, tmp_path_factory, admitted_rule_aliases):
    kind = request.param
    directory = tmp_path_factory.mktemp('rule-sensor-' + kind)
    bars = pl.read_parquet(FIXTURE, n_rows=336)
    if kind == 'already-formed':
        threshold = int(bars['no_of_trades'].max()) * 2
        bars = trade_bars(pl.read_parquet(FIXTURE), threshold).head(336)
        assert bars.height == 336 and bars['bar_count'].max() > 1
    if kind == 'nulls':
        bars = bars.with_columns(pl.when(pl.col('datetime') == bars['datetime'][200]).then(None).otherwise(pl.col('close')).alias('nullable_observation'))
    source = directory / 'source.parquet'
    bars.head(168).write_parquet(source)
    config = _config(_conditions(), {'hold_bars': [1, 3], 'recent_bars': [1], 'fee': [3.0], 'size': [1.0]})
    manifest = config['sfd']['manifest']
    config['metadata']['name'] = 'rule_sensor_' + kind.replace('-', '_')
    if kind == 'leaves':
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
        config['sfd']['params']['price_floor'] = [float(bars.head(168)['close'].median())]
        manifest['strategy']['entry'] = 'combined'
    if kind == 'optional-off':
        manifest['indicators'] = [{'func': 'limen.indicators.ema', 'params': {'period': '{period}'}, 'include_if': 'use_ema'}]
        config['sfd']['params'].update(period=[3], use_ema=[False])
    if kind == 'common':
        manifest['indicators'] = [{'func': 'limen.features.common_strategy_positions.common_strategy_positions', 'params': {'strategy': '{strategy}'}}]
        manifest['strategy'] = {'conditions': [{'id': 'entry', 'type': 'threshold', 'column': 'common_strategy_position', 'operator': '>', 'value': 0}], 'entry': 'entry'}
        config['sfd']['params'] = {'strategy': ['buy_and_hold', 'time_series_momentum'], 'fee': [3.0], 'size': [1.0]}
    if kind == 'nulls':
        manifest['indicators'] = [{'func': 'limen.features.lagged_features.lag_column', 'params': {'col': 'close', 'lag': 2}, 'include_if': 'use_lag'}]
        config['sfd']['params']['use_lag'] = [True]
    if kind == 'policy':
        manifest['backtest'].update(product={'kind': 'cash_spot', 'instrument': 'BTCUSDT', 'base_currency': 'BTC', 'quote_currency': 'USDT', 'quantity_step': 1e-9, 'min_notional': 0}, source_interval_seconds=3600)
    if kind == 'sql-window':
        manifest['strategy'] = {'conditions': [{'id': 'entry', 'type': 'sql_expr', 'expr': 'close > AVG(close)'}], 'entry': 'entry'}
        config['sfd']['params'] = {'fee': [3.0, 10.0], 'size': [1.0]}
    if kind.endswith('-alias'):
        alias = 'rule_sensor_logreg' if kind == 'logreg-alias' else 'rule_sensor_native'
        manifest['reference_architecture'] = 'limen.sfd.reference_architecture.rule_based.' + alias
    for condition in manifest['strategy']['conditions']:
        condition.setdefault('name', condition['id'])
    manifest['data_source']['params']['file_path_or_url'] = str(source)
    if kind == 'fold':
        manifest.pop('split_dates')
        manifest['split_walk_forward'] = {'n_folds': 2, 'test_bars': 24, 'purge_bars': 2, 'embargo_bars': 1, 'anchored': True}
    config['uel']['n_permutations'] = 2
    checked = validate(config)
    assert checked.valid, checked.errors
    output = directory / 'experiment'
    loop = UniversalExperimentLoop(sfd=CompiledSFD(config), data=bars.head(168), experiment_dir=output,
                                   yaml_reference=config, search_strategy=build_search_strategy(config), checkpoint_interval=1)
    options = {} if kind == 'default' else {'recording': dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True),
                                          'record_execution': True, 'record_model_outputs': True}
    loop.run('recorded_rule_sensor', n_permutations=2, prep_each_round=True, progress_bar=False, **options)
    trainer = Trainer(output, data=bars.head(168))
    ids = list(trainer._round_data)
    sensors = trainer.train(ids, **({'fold': 0} if kind == 'fold' else {}))
    records = [json.loads(line) for line in _round_storage.text(_round_storage.round_path(output)).splitlines()]
    return output, config, bars, trainer, ids, sensors, records, kind


def _whole_window(manifest, history, params):
    native = copy.deepcopy(manifest)
    first = history['datetime'][0].date()
    last = history['datetime'][-1].date() + timedelta(days=1)
    native.set_split_dates(first, last, last, last, last, last)
    return native.prepare_data(history, params)


def test_recorded_rule_sensor_promotion(rule_campaign):
    _, config, bars, trainer, ids, sensors, _, kind = rule_campaign
    assert len(ids) == len(sensors) == 2 and sensors[0].round_params != sensors[1].round_params
    assert all(sensor.permutation_id == pid and sensor.manifest_id == trainer._manifest_id for sensor, pid in zip(sensors, ids, strict=True))
    assert all(type(sensor._model) is RuleBasedStrategy and sensor._model.inference_output == ('binary', False) for sensor in sensors)
    before = copy.deepcopy([sensor._model._strategy for sensor in sensors])
    history = bars.slice(168, 120)
    first, other = sensors[0].predict_all(history), sensors[1].predict_all(history)
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
        assert tuple(columns) == sensor._model._predicate_ids
        valid = prepared.drop_nulls(columns)
        expected = sensor._model.predict({'test': valid, 'strategy': sensor._model._strategy})['_preds'].tolist()
        actual = sensor.predict_all(history)
        rows = [row for row in actual if row.reason is None]
        assert rows and [row.datetime for row in rows] == valid['datetime'].to_list()
        assert [row.prediction for row in rows] == expected and sensor.predict(history) == actual[-1]
        assert len(actual) == history.height and len(rows) == valid.height
        if kind == 'optional-off':
            assert lookback == 0
        if kind == 'nulls':
            assert lookback == 2 and actual[0].reason == 'warm-up' and actual[32].reason == 'null-features'
        if kind == 'policy':
            assert all(row.available_at_ns == int(valid['datetime'].dt.epoch('ns')[index]) + 3600_000_000_000 and row.trade_contract_digest == sensor._trade_digest for index, row in enumerate(rows))
        if kind not in ('fold', 'policy'):
            research = _whole_window(manifest, history, sensor.round_params)
            assert sensor._model.predict({'test': research['train'], 'strategy': sensor._model._strategy})['_preds'].tolist() == expected
            assert research['x_train'].shape[1] == len(columns) and research['x_train'].dtype == np.float64
        for size in (1, 5, 31):
            window = history.head(size)
            batch = sensor.predict_all(window)
            assert sensor.predict(window) == batch[-1]
            if kind != 'sql-window':
                assert batch == actual[:size]
        if kind == 'sql-window':
            assert any(sensor.predict_all(history.head(size)) != actual[:size] for size in range(2, len(history)))
        matrix = valid.select(columns).to_numpy().astype(np.float64)
        prediction = sensor._model.predict({'x_test': matrix})
        assert prediction['_preds'].tolist() == expected and '_probs' not in prediction
        if kind == 'already-formed':
            assert manifest.bar_formation is None and bars['bar_count'].max() > 1
            raw = pl.read_parquet(FIXTURE, n_rows=336)
            threshold = int(raw['no_of_trades'].max()) * 2
            native = copy.deepcopy(manifest).set_bar_formation(trade_bars, trade_threshold=threshold)
            params = sensor.round_params | {'bar_type': 'trade'}
            formed, _ = native.sensor_input_prep(raw, {}, params)
            assert 0 < formed.height < raw.height and formed['datetime'].equals(trade_bars(raw, threshold)['datetime'])
            first = raw['datetime'][0].date()
            middle, later, end = first + timedelta(days=7), first + timedelta(days=10), first + timedelta(days=14)
            native.set_split_dates(first, middle, middle, later, later, end)
            research = native.prepare_data(raw, params)
            train_input = raw.filter(pl.col('datetime').dt.date() < middle)
            prepared, _ = native.sensor_input_prep(train_input, {}, params)
            assert prepared.height == research['train'].height > 0
            assert prepared['datetime'].equals(trade_bars(train_input, threshold)['datetime'])
            assert sensor._model.predict({'test': prepared, 'strategy': sensor._model._strategy})['_preds'].tolist() == sensor._model.predict({'test': research['train'], 'strategy': sensor._model._strategy})['_preds'].tolist()


def test_recorded_rule_sensor_consumers(rule_campaign):
    output, config, bars, _, ids, sensors, _, kind = rule_campaign
    selections = [(output, pid, 0) for pid in ids] if kind == 'fold' else [(output, pid) for pid in ids]
    cohorts = [Cohort(rounds=selections)]
    if kind != 'fold':
        cohorts.append(Cohort(experiment_log_path=str(output), permutation_ids=ids))
    history = bars.slice(168, 120)
    native = [sensor.predict_all(history) for sensor in sensors]
    expected = [None if left.reason is not None or right.reason is not None else int((float(left.prediction) + float(right.prediction)) / 2 > 0.5) for left, right in zip(*native, strict=True)]
    for cohort in cohorts:
        if kind == 'logreg-alias':
            assert cohort.supports_probabilities is True
        cohort.set_members(list(reversed(sensors)))
        assert cohort.supports_probabilities is False and cohort.aggregation_mode == 'majority_vote'
        actual = cohort.predict_all(history)
        assert any(row.reason is None for row in actual) and [row.prediction for row in actual] == expected
        assert all(row.reason != 'sensor-error' and row.probability is None and row.available_at_ns is None and row.trade_contract_digest is None for row in actual)
        identity = cohort.cohort_id
        cohort.set_members(sensors)
        assert cohort.cohort_id == identity and cohort.predict(history) == actual[-1]
    guarded = copy.deepcopy(sensors[0])
    manifest = guarded._get_manifest()
    manifest.val_predict_guard = manifest.test_predict_guard = True
    guarded_rows = guarded.predict_all(bars.head(168))
    if kind == 'fold':
        assert all(row.reason == 'inside-training-window' and row.prediction is None for row in guarded_rows)
    else:
        dates = config['sfd']['manifest']['split_dates']
        historical = [row for row in guarded_rows if dates['train_start'] <= row.datetime.date().isoformat() < dates['test_end']]
        assert historical and all(row.reason == 'inside-training-window' and row.prediction is None for row in historical)
        assert any(row.reason is None for row in guarded_rows if row.datetime.date().isoformat() >= dates['test_end'])


def test_rule_sensor_evidence_refusals(rule_campaign):
    _, _, bars, trainer, ids, sensors, _, kind = rule_campaign
    sensor = sensors[0]
    original = copy.deepcopy(sensor._model._strategy)
    width = len(sensor._model._predicate_ids)
    for value in (np.zeros(3), np.zeros((3, width + 1)), np.full((3, width), np.nan), np.full((3, width), 2.0)):
        with pytest.raises(ValueError, match='finite binary matrix'):
            sensor._model.predict({'x_test': value})
    with pytest.raises(ValueError, match='captured selected strategy'):
        RuleBasedStrategy().predict({'x_test': np.zeros((1, 1))})
    assert sensor._model._strategy == original
    if kind == 'fold':
        changed = bars.head(168).with_columns((pl.col('close') * 1.01).alias('close'))
        with pytest.raises(ReconstructionError):
            Trainer(trainer._experiment_dir, data=changed).train(ids, fold=0)
    else:
        with pytest.raises(ValueError, match='not found'):
            trainer.train(['not-a-recorded-round'])
        altered = copy.deepcopy(trainer)
        selected = altered._original_log[ids[0]]
        assert isinstance(selected['position_rate_test'], (int, float)) and np.isfinite(selected['position_rate_test'])
        selected['position_rate_test'] += 1.0
        with pytest.raises(ReconstructionError, match='metric mismatch'):
            altered.train([ids[0]])

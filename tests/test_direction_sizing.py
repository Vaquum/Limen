from pathlib import Path
from dataclasses import replace
from datetime import date
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch
import json
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from limen.experiment.manifest_core import FundingConfig, MLManifest, ProductConfig
from limen.sfd.reference_architecture.direction_sizing import direction_sizing
from limen.targets import TradeOutcomeTarget


def recorded_source(klines_size=900, limit=120):
    return pl.read_parquet(Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet').head(limit)


def recorded_yaml_source(**params):
    return recorded_source(limit=288)


def native_manifest():
    manifest = MLManifest().set_data_source(recorded_source, params={'klines_size':900})
    manifest.set_split_config(6,2,2)
    manifest.with_target_label('outcome', TradeOutcomeTarget)
    manifest.with_reference_architecture(direction_sizing)
    manifest.set_backtest_config(prediction_mode='target_exposure', product=ProductConfig('linear_perpetual','BTCUSDT','BTC','USDT',1e-9,0), max_holding_seconds=1800, funding=FundingConfig(preset='binance_btcusdt', params={'rate':0.0001,'settlement_phase_utc_seconds':1800}))
    return manifest


@pytest.mark.parametrize('reference', ('{interval}', 'interval'))
def test_sensor_preparation_resolves_recorded_interval(reference):
    source = recorded_source()
    manifest = native_manifest().set_data_source(recorded_source, params={'klines_size': reference})
    params = {'interval': 900}
    data = manifest.prepare_data(source, params)
    prepared, _ = manifest.sensor_input_prep(source, data['_fitted_params'], params)
    assert prepared.height == source.height
    assert prepared['__trade_available_at_ns__'].equals(source['datetime'].dt.epoch('ns') + 900000000000, check_names=False)


class ConstantComponent:
    def __init__(self, value):
        self.value = value

    def fit(self, x, y):
        assert len(x) == len(y) > 0
        self.targets = np.asarray(y).copy()
        self.fit_x = np.asarray(x).copy()
        return self

    def predict(self, x):
        return np.full(len(x), self.value, dtype=np.float64)


def direction_factory(*, seed, sign=1):
    return ConstantComponent(sign)


def sizing_factory(*, seed, size=0.25):
    return ConstantComponent(size)


direction_factory.deterministic = True
sizing_factory.deterministic = True


def component_params(**kwargs):
    return {'direction_factory':direction_factory,'sizing_factory':sizing_factory,'min_train_samples':2,**kwargs}


def test_native_components_and_tunable_parameters():
    manifest = native_manifest()
    source = recorded_source()
    data = manifest.prepare_data(source, {})
    for sign, size in ((1,.25),(-1,.25),(-1,.75)):
        result = manifest.run_model(data, component_params(direction_params={'sign':sign}, sizing_params={'size':size}))
        assert np.all(result['_preds'] == sign * size)
        assert len(result['_preds']) == len(data['x_test'])
        assert result['_model'].prediction_mode == 'target_exposure'
        assert result['_model'].deterministic
    with pytest.raises(ValueError, match='feature'):
        manifest.run_model(data, component_params(direction_features=['missing']))
    manifest.set_backtest_config(**{**manifest.backtest_config.__dict__, 'max_holding_seconds':'holding'})
    first = manifest.prepare_data(source, {'holding':1800})
    with pytest.raises(ValueError, match='Cached trade preparation'):
        manifest.run_model(first, component_params(holding=3600))
    second = manifest.prepare_data(source, {'holding':3600})
    assert second['trade_contract_digest'] != first['trade_contract_digest']


class ChangingComponent(ConstantComponent):
    def __init__(self, value):
        super().__init__(value)
        self.calls = 0

    def predict(self, x):
        self.calls += 1
        return np.full(len(x), self.value if self.calls == 1 else 0, dtype=np.float64)


def changing_direction_factory(*, seed):
    return ChangingComponent(-1)


def changing_sizing_factory(*, seed):
    return ChangingComponent(0.25)


changing_direction_factory.deterministic = False
changing_sizing_factory.deterministic = False


def test_evaluation_scores_the_returned_component_predictions():
    manifest = native_manifest()
    data = manifest.prepare_data(recorded_source(), {})
    result = manifest.run_model(data, component_params(direction_factory=changing_direction_factory, sizing_factory=changing_sizing_factory))
    model = result['_model']
    assert not model.deterministic
    assert model.direction_model.calls == model.sizing_model.calls == 1
    assert np.all(result['_preds'] == -0.25)
    from limen.sfd.reference_architecture._backtest_evaluation import compute_backtest
    expected = compute_backtest(np.full(len(data['x_test']), -0.25), data)
    assert all(result[key] == pytest.approx(value) for key, value in expected.items())


class MemoryDirection(ConstantComponent):
    fit_sizes: ClassVar[list[int]] = []

    def fit(self, x, y):
        self.fit_sizes.append(len(x))
        return super().fit(x, y)

    def predict(self, x):
        return np.array([1.0 if np.equal(self.fit_x, row).all(axis=1).any() else -1.0 for row in np.asarray(x)])


def memory_direction_factory(*, seed):
    return MemoryDirection(1)


memory_direction_factory.deterministic = True


def test_stacking_predictions_are_causal():
    from limen.scalers import RobustScaler

    manifest = native_manifest().set_scaler(RobustScaler).set_pca_compression()
    params = component_params(direction_factory=memory_direction_factory, conditional_size=True, folds=3, auto_pca=True, pca_k=2)
    data = manifest.prepare_data(recorded_source(), params)
    preparation = data['_fold_preparation']
    calls = []

    def observe(train, predict):
        calls.append((tuple(train), tuple(predict)))
        return preparation.transform(train, predict)

    data['_fold_preparation'] = replace(preparation, transform=observe)
    MemoryDirection.fit_sizes = []
    result = manifest.run_model(data, params)
    model = result['_model']
    assert np.all(model.sizing_model.fit_x[:, -1] == -1)
    assert np.all(model.direction_model.predict(model.direction_model.fit_x) == 1)
    assert len(calls) == 2
    positions = {identity:i for i, identity in enumerate(preparation.row_ids)}
    for train, predict in calls:
        assert max(positions[row] for row in train) < min(positions[row] for row in predict)
    private = data['_trade_labels'].rows
    signals = data['_trade_context'].partitions[0].signals
    completion = signals.join(private, on='row_id', how='left').with_columns(pl.max_horizontal('long_label_available_ns','short_label_available_ns').alias('completed'))
    for index, (train, predict) in enumerate(calls):
        start = signals.filter(pl.col('row_id') == predict[0])['available_at_ns'][0]
        eligible = completion.filter(pl.col('row_id').is_in(train) & pl.col('long_available') & pl.col('short_available') & (pl.col('completed') < start))
        assert MemoryDirection.fit_sizes[index] == eligible.height < len(train)
    assert model.sizing_model.fit_x.shape[0] <= data['x_train'].height - len(np.array_split(np.arange(data['x_train'].height),3)[0])
    # Recorded outer-validation/test features are swapped. OOF fitting sees neither.
    isolated = dict(data)
    isolated['x_val'], isolated['x_test'] = data['x_test'], data['x_val']
    repeated = manifest.run_model(isolated, params)['_model']
    assert np.array_equal(repeated.sizing_model.fit_x, model.sizing_model.fit_x)
    with pytest.raises(ValueError, match='fold 1'):
        manifest.run_model(data, {**params, 'folds':8, 'min_train_samples':20})


YAML_SPEC = '''schema_version: "1.0"
metadata:
  name: trade_outcome_direction_sizing
  mode: development
sfd:
  manifest:
    type: ml
    data_source:
      method: limen.data.HistoricalData.get_spot_klines
      params:
        klines_size: 900
    split_dates:
      train_start: "2025-01-01"
      train_end: "2025-01-02"
      val_start: "2025-01-02"
      val_end: "2025-01-03"
      test_start: "2025-01-03"
      test_end: "2025-01-04"
    target:
      name: outcome
      class: limen.targets.TradeOutcomeTarget
    scaler:
      from_params: scaler_type
    reference_architecture: limen.sfd.reference_architecture.direction_sizing
    backtest:
      prediction_mode: target_exposure
      product:
        kind: linear_perpetual
        instrument: BTCUSDT
        base_currency: BTC
        quote_currency: USDT
        quantity_step: 0.000000001
        min_notional: 0
      max_holding_seconds: "{holding}"
      fee_bps: 1
      funding:
        preset: binance_btcusdt
        params:
          rate: "{funding_rate}"
  params:
    holding: [1800, 3600]
    funding_rate: [0.0001]
    scaler_type: [robust]
    direction_factory: [limen.sfd.reference_architecture.direction_sizing._direction_factory]
    sizing_factory: [limen.sfd.reference_architecture.direction_sizing._sizing_factory]
    direction_params: [{max_iter: 1000, C: 0.5}]
    sizing_params: [{alpha: 1.0}]
    min_train_samples: [2]
    seed: [42]
uel:
  n_permutations: 2
  search_strategy:
    type: grid
  output_format: csv
'''


def yaml_experiment(tmp_path):
    from limen.cli.commands.run import run_experiment
    from limen.data.historical_data import HistoricalData

    yaml_path = tmp_path / 'strategy.yaml'
    yaml_path.write_text(YAML_SPEC)
    with patch.object(HistoricalData, 'get_spot_klines', new=staticmethod(recorded_yaml_source)):
        assert run_experiment(yaml_path, results_base=tmp_path, progress_bar=False)
    return next(tmp_path.rglob('metadata.json')).parent


def test_trainer_sensor_preserve_signed_exposure(tmp_path):
    from limen.inference import Trainer, ReconstructionError
    from limen.cohort import Cohort

    directory = yaml_experiment(tmp_path)
    trainer = Trainer(directory, data=recorded_source(limit=288))
    ids = list(trainer._round_data)
    assert len(ids) == 2
    sensors = trainer.train(ids)
    for sensor, identity in zip(sensors, ids, strict=True):
        stored = trainer._round_data[identity]
        assert sensor.prediction_mode == 'target_exposure'
        assert sensor.trade_contract == stored['trade_contract']
        bars = sensor.predict_all(recorded_source(limit=384))
        valid = [bar for bar in bars if bar.reason is None]
        assert valid and all(bar.trade_contract_digest == stored['trade_contract_digest'] for bar in valid)
        assert all(bar.available_at_ns == int(bar.datetime.timestamp() * 1_000_000_000) + 900_000_000_000 for bar in valid)
        prepared, _ = sensor._manifest.sensor_input_prep(recorded_source(limit=384), sensor._fitted_params, sensor.round_params)
        features = prepared.filter(pl.col('datetime').dt.date() >= date(2025,1,4)).drop('datetime', '__trade_available_at_ns__')
        assert [bar.prediction for bar in valid] == sensor._model.predict({'x_test':features})['_preds'].tolist()
        owned = sensor.trade_contract
        owned['rule_version'] = 'changed'
        assert sensor.trade_contract != owned
    from limen.inference import Sensor

    manifest = sensors[0]._manifest
    round_params = {**sensors[0].round_params, **component_params(direction_params={'sign':-1}, sizing_params={'size':.25})}
    data = manifest.prepare_data(recorded_source(limit=288), round_params)
    model = manifest.run_model(data, round_params)['_model']
    sized = Sensor(sensors[0]._yaml_reference, model, data['_fitted_params'], round_params, trade_contract=data['trade_contract'])
    assert all(bar.prediction == -.25 for bar in sized.predict_all(recorded_source(limit=384)) if bar.reason is None)
    assert sized.predict(recorded_source(limit=384)).prediction == -.25
    cohort = Cohort(experiment_log_path=str(directory), permutation_ids=[ids[0]])
    with pytest.raises(ValueError, match='target-exposure'):
        cohort.set_members([sensors[0]])
    changed = Trainer(directory, data=recorded_source(limit=288).tail(287))
    with pytest.raises(ReconstructionError, match='identity changed'):
        changed.train([ids[0]])
    changed = Trainer(directory, data=recorded_source(limit=288))
    changed._round_data[ids[0]]['learning_binding']['model']['seed'] = 999
    with pytest.raises(ReconstructionError, match='learning/source/factory'):
        changed.train([ids[0]])
    code = 'from pathlib import Path; from limen.inference import Trainer; from tests.test_direction_sizing import recorded_source; t=Trainer(Path(__import__("sys").argv[1]),data=recorded_source(limit=288)); assert t.train(list(t._round_data))[0].prediction_mode == "target_exposure"'
    subprocess.run([sys.executable, '-c', code, str(directory)], check=True)


def test_inline_log_and_exported_contract_agree(tmp_path):
    from limen.log._snapshot_backtest_round import snapshot_backtest_round
    from limen.inference import Trainer

    manifest = native_manifest()
    source = recorded_source()
    data = manifest.prepare_data(source, {})
    params = component_params(direction_params={'sign':-1}, sizing_params={'size':.5})
    result = manifest.run_model(data, params)
    log = SimpleNamespace(manifest=manifest, data=source, round_params=[params], preds=[result['_preds']], _alignment=[data['_alignment']])
    replay = snapshot_backtest_round(log, 0, lambda frame: pytest.fail('signed predictions must retain magnitude'))
    assert all(result[f'backtest_{name}'] == pytest.approx(value) for name, value in replay.items())
    # One timed short episode: independent net arithmetic over the retained fills.
    ledger = data['_trade_ledger']
    fill = ledger.fills
    assert fill.height == 2
    quantity = -fill['quantity_delta'][0]
    expected_funding = quantity * fill['fill_price'][0] / (1-.0005) * .0001
    assert ledger.funding['recognized_delta'].sum() == pytest.approx(expected_funding)
    expected = quantity * (fill['fill_price'][0] - fill['fill_price'][1]) - fill['fee'].sum() + ledger.funding['recognized_delta'].sum()
    assert result['backtest_net_pnl'] == pytest.approx(expected)
    directory = yaml_experiment(tmp_path)
    rows = [json.loads(line) for line in (directory/'round_data.jsonl').read_text().splitlines()]
    assert len(rows) == 2 and rows[0]['trade_contract_digest'] != rows[1]['trade_contract_digest']
    for row in rows:
        assert row['learning_binding']['round_id'] == row['round_id']
        assert row['learning_binding']['manifest_id'] is not None
        assert row['learning_binding']['model']['trade_contract_digest'] == row['trade_contract_digest']
    assert Trainer(directory, data=recorded_source(limit=288)).train([rows[0]['round_id']])
    from limen.sfd.reference_architecture import logreg_binary
    from limen.targets import NextBarUpTarget
    from limen.backtest import backtest_snapshot

    legacy = MLManifest().set_data_source(recorded_source, params={'klines_size':900}).set_split_config(6,2,2)
    legacy.with_target_label('next_up', NextBarUpTarget).with_reference_architecture(logreg_binary)
    prepared = legacy.prepare_data(source, {})
    control = legacy.run_model(prepared, {})
    assert 'trade_contract' not in prepared and '_trade_labels' not in prepared
    assert set(control['_preds']) <= {0,1} and prepared['y_test'].null_count() == 0
    prices = prepared['price_data_for_backtest'].with_columns(pl.Series('predictions',control['_preds'])).with_columns((pl.col('close')-pl.col('open')).alias('price_change'))
    expected = backtest_snapshot(prices, execution_lag_bars=1, fee_bps=5, slip_bps=5, notional_rate=1)
    assert all(control[f'backtest_{name}'] == pytest.approx(value, nan_ok=True) for name,value in expected.items())


def test_supervision_masks_mapping_and_prediction_validation():
    from limen.sfd.reference_architecture.direction_sizing import _labels
    from limen.targets import OutcomeLabels

    data = native_manifest().prepare_data(recorded_source(), {})
    labels = data['_trade_labels']
    train_ids = data['_trade_context'].partitions[0].signals['row_id'].to_list()
    rows = labels.rows
    # Equal recorded side outcomes, neither-positive and missing-side cases.
    rows = rows.with_columns(pl.when(pl.col('row_id') == train_ids[0]).then(pl.col('long_return')).otherwise(pl.col('short_return')).alias('short_return'))
    rows = rows.with_columns(pl.when(pl.col('row_id') == train_ids[1]).then(False).otherwise(pl.col('short_available')).alias('short_available'), pl.when(pl.col('row_id') == train_ids[1]).then(None).otherwise(pl.col('short_return')).alias('short_return'))
    data['_trade_labels'] = OutcomeLabels(rows, labels.contract_digest)
    _, direction, size, mask = _labels(data,0,.01,1)
    assert direction[0] == 0 and size[0] == 0
    assert not mask[1] and mask[0]
    for wrong in (np.nan, np.inf):
        with pytest.raises(ValueError, match='Nonfinite'):
            native_manifest().run_model(data, component_params(sizing_params={'size':wrong}))
    with pytest.raises(ValueError, match='different execution contracts'):
        _labels({**data, '_trade_labels':OutcomeLabels(rows,'changed')},0,.01,1)
    with pytest.raises(ValueError, match='finite scalars'):
        native_manifest().run_model(data, component_params(sizing_factory=nonfinite_sizing_factory))


def nonfinite_sizing_factory(*, seed):
    return ConstantComponent(np.nan)


nonfinite_sizing_factory.deterministic = True

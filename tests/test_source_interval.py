from pathlib import Path

import polars as pl
import pytest

from limen.backtest.trade_contract import NANOSECONDS
from limen.backtest.execution_events import with_predictions
from limen.backtest.trade_execution import trade_execution
from limen.data import HistoricalData
from limen.experiment._prepare_trade_context import _load
from limen.experiment.manifest_core import DataSourceConfig, MLManifest, ProductConfig
from limen.targets import IdentityTarget
from limen.yaml import parse, validate
from limen.yaml.compiler import build_manifest


@pytest.fixture
def recorded_history(monkeypatch):
    source = pl.read_parquet(Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet').head(480)
    monkeypatch.setattr('limen.data.historical_data._read_any_file', lambda *args, **kwargs: source)
    return source


def event_manifest(method, params):
    manifest = MLManifest().set_data_source(method, params=params)
    manifest.set_split_config(6, 2, 2).with_target_label('close', IdentityTarget)
    manifest.set_backtest_config(prediction_mode='target_exposure', product=ProductConfig('cash_spot', 'BTCUSDT', 'BTC', 'USDT', 1e-9, 0), max_holding_seconds=86400)
    return manifest


def test_historical_interval_reaches_preparation_and_sensor(recorded_history):
    manifest = event_manifest(HistoricalData.get_spot_klines, {'kline_size': 3600})
    source = manifest.fetch_data()
    assert 'base_interval' not in source.columns
    assert source['datetime'].diff().drop_nulls().dt.total_seconds().unique().to_list() == [3600]
    data = manifest.prepare_data(source, {})
    inputs = data['_trade_inputs']
    assert (inputs.observations['end_ns'] - inputs.observations['start_ns']).unique().to_list() == [3600 * NANOSECONDS]
    prepared, _ = manifest.sensor_input_prep(source, data['_fitted_params'], {})
    assert prepared['__trade_available_at_ns__'].equals(source['datetime'].dt.epoch('ns') + 3600 * NANOSECONDS, check_names=False)


@pytest.mark.parametrize('key', ('kline_size', 'klines_size'))
@pytest.mark.parametrize('interval', (900, '{interval}', 'interval'))
def test_declared_interval_preserves_round_references(recorded_history, key, interval):
    manifest = event_manifest(lambda **params: recorded_history, {key: interval})
    params = {'interval': 900}
    data = manifest.prepare_data(recorded_history, params)
    prepared, _ = manifest.sensor_input_prep(recorded_history, data['_fitted_params'], params)
    assert prepared['__trade_available_at_ns__'].equals(recorded_history['datetime'].dt.epoch('ns') + 900 * NANOSECONDS, check_names=False)


@pytest.mark.parametrize('params', ({'kline_size': None, 'klines_size': 900}, {'kline_size': 900, 'klines_size': 3600}))
def test_canonical_interval_retains_legacy_fallback(recorded_history, params):
    manifest = event_manifest(lambda **kwargs: recorded_history, params)
    data = manifest.prepare_data(recorded_history, {})
    prepared, _ = manifest.sensor_input_prep(recorded_history, data['_fitted_params'], {})
    assert prepared['__trade_available_at_ns__'].equals(recorded_history['datetime'].dt.epoch('ns') + 900 * NANOSECONDS, check_names=False)


@pytest.mark.parametrize('interval', (900, '{execution_interval}'))
@pytest.mark.parametrize('bound', (False, True))
def test_historical_execution_source_uses_declared_interval(recorded_history, interval, bound):
    manifest = event_manifest(HistoricalData.get_spot_klines, {'kline_size': 3600})
    method = HistoricalData().get_spot_klines if bound else HistoricalData.get_spot_klines
    manifest.backtest_config.execution_data_source = DataSourceConfig(method, {'kline_size': interval})
    data = manifest.prepare_data(manifest.fetch_data(), {'execution_interval': 900})
    inputs = data['_trade_inputs']
    assert (inputs.observations['end_ns'] - inputs.observations['start_ns']).unique().to_list() == [900 * NANOSECONDS]
    assert data['_trade_context'].model_rows[2].height < inputs.observations.height


@pytest.mark.parametrize('execution_interval', (900, 1800))
def test_yaml_historical_execution_source_uses_finer_recorded_fills(recorded_history, execution_interval):
    config, errors = parse('''schema_version: "1.0"
metadata:
  name: historical_execution
  mode: development
sfd:
  manifest:
    type: ml
    data_source:
      method: limen.data.HistoricalData.get_spot_klines
      params:
        kline_size: 3600
    split_dates:
      train_start: "2025-01-01"
      train_end: "2025-01-02"
      val_start: "2025-01-02"
      val_end: "2025-01-03"
      test_start: "2025-01-03"
      test_end: "2025-01-04"
    target:
      name: close
      class: limen.targets.IdentityTarget
    reference_architecture: limen.sfd.reference_architecture.ridge_regressor
    backtest:
      prediction_mode: target_exposure
      product:
        kind: cash_spot
        instrument: BTCUSDT
        base_currency: BTC
        quote_currency: USDT
        quantity_step: 0.000000001
        min_notional: 0
      fee_bps: 0
      slip_bps: 0
      max_holding_seconds: 900
      timer_interval_seconds: 900
      execution_data_source:
        method: limen.data.HistoricalData.get_spot_klines
        params:
          kline_size: "{execution_interval}"
          start_date_limit: "2025-01-01"
          end_date_limit: "2025-01-04"
  params:
    execution_interval: [900, 1800]
uel:
  n_permutations: 2
''')
    assert errors == []
    validation = validate(config)
    assert validation.valid, [error.message for error in validation.errors]
    manifest = build_manifest(config)
    assert manifest.backtest_config.execution_data_source.method is HistoricalData.get_spot_klines
    source = manifest.fetch_data()
    params = {'execution_interval': execution_interval}
    data = manifest.prepare_data(source, params)
    inputs = data['_trade_inputs']
    policy = data['_trade_policy']
    assert (inputs.observations['end_ns'] - inputs.observations['start_ns']).unique().to_list() == [execution_interval * NANOSECONDS]
    assert data['_trade_context'].source_settings[0][1]['kline_size'] == execution_interval
    assert inputs.observations.height > inputs.signals.height
    ledger = trade_execution(with_predictions(inputs, [0.5] * inputs.signals.height), policy)
    entered = inputs.signals['available_at_ns'][0]
    closed = entered + execution_interval * NANOSECONDS
    assert ledger.episodes['first_fill_ns'].to_list() == [entered]
    assert ledger.episodes['closed_at_ns'].to_list() == [closed]
    assert ledger.intents.filter(pl.col('reason') == 'time_stop')['requested_at_ns'].to_list() == [entered + 900 * NANOSECONDS]
    assert closed < inputs.signals['available_at_ns'][1]
    expected_price = recorded_history.filter(pl.col('datetime').dt.epoch('ns') == closed)['open'][0]
    assert ledger.fills['reference_price'][1] == expected_price


def _recorded_trade_source(*, rows):
    return pl.read_parquet(Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet').head(rows)


class _RecordedTradeSource:
    @staticmethod
    def static(*, rows):
        return _recorded_trade_source(rows=rows)

    def load(self, *, rows):
        self.data = _recorded_trade_source(rows=rows)
        return self.data

    def __call__(self, *, rows):
        return _recorded_trade_source(rows=rows)


def test_trade_source_keeps_custom_callable_results_and_errors():
    expected = _recorded_trade_source(rows=3)
    methods = (_recorded_trade_source, _RecordedTradeSource.load, _RecordedTradeSource().load,
               _RecordedTradeSource.static, _RecordedTradeSource(), lambda **kwargs: _recorded_trade_source(**kwargs))
    for method in methods:
        assert _load(DataSourceConfig(method, {'rows': '{count}'}), {'count': 3}).equals(expected)

    def unavailable():
        raise RuntimeError('Recorded source unavailable')

    with pytest.raises(RuntimeError, match='Recorded source unavailable'):
        _load(DataSourceConfig(unavailable), {})
    with pytest.raises(ValueError, match='Polars DataFrame'):
        _load(DataSourceConfig(lambda: []), {})

from pathlib import Path

import polars as pl
import pytest

from limen.backtest.trade_contract import NANOSECONDS
from limen.data import HistoricalData
from limen.experiment.manifest_core import DataSourceConfig, MLManifest, ProductConfig
from limen.targets import IdentityTarget


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
def test_historical_execution_source_uses_declared_interval(recorded_history, interval):
    manifest = event_manifest(HistoricalData.get_spot_klines, {'kline_size': 3600})
    manifest.backtest_config.execution_data_source = DataSourceConfig(HistoricalData().get_spot_klines, {'kline_size': interval})
    data = manifest.prepare_data(manifest.fetch_data(), {'execution_interval': 900})
    inputs = data['_trade_inputs']
    assert (inputs.observations['end_ns'] - inputs.observations['start_ns']).unique().to_list() == [900 * NANOSECONDS]
    assert data['_trade_context'].model_rows[2].height < inputs.observations.height

from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from limen.backtest.execution_events import with_predictions
from limen.backtest.trade_contract import contract_digest, export_trade_contract
from limen.backtest.trade_execution import trade_execution
from limen.data.bars import volume_bars
from limen.experiment._prepare_trade_context import normalize_observations, prepare_trade_context, sensor_decisions
from limen.experiment._resolve_trade_policy import BacktestConfig, ProductConfig, resolve_trade_policy
from limen.experiment.manifest_core import DataSourceConfig, MLManifest, RuleBasedManifest
from limen.targets import IdentityTarget
from tests.test_elapsed_trade_exits import _ticks


@pytest.fixture(scope='module')
def recorded():
    source = pl.read_parquet(Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet', n_rows=32)
    observations = normalize_observations(source, interval_seconds=900)
    raw = source.hstack(observations.select('start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns'))
    return raw, observations


def _prepare(raw, execution, *, bars=None, lag=0):
    config = BacktestConfig(fee_bps=0, slip_bps=0, prediction_mode='target_exposure',
        product=ProductConfig('linear_perpetual', 'BTCUSDT', 'BTC', 'USDT', 1e-9, 0),
        execution_lag_seconds=lag,
        execution_data_source=None if execution is None else DataSourceConfig(lambda: execution))
    policy = resolve_trade_policy(config, {})
    assert policy is not None
    formed = volume_bars(raw, volume_threshold=500) if bars is None else bars
    context = prepare_trade_context(config, policy, [raw], [formed], {}, frozen_sources=(execution, None))
    return context, formed


def _delay(raw, available):
    return raw.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(available).otherwise(pl.col('available_at_ns')).alias('available_at_ns'))


@pytest.mark.parametrize('lag', (0, 900))
def test_complete_support_drives_recorded_fills_and_sensor(recorded, lag):
    raw, execution = recorded
    original = raw.clone()
    original_execution = execution.clone()
    delayed = _delay(raw, raw['available_at_ns'][4])
    context, formed = _prepare(delayed, execution, lag=lag)
    assert formed['bar_count'].to_list() == [4, 4, 9, 10]
    mapping = context.model_rows[0]
    assert mapping['available_at_ns'][0] == raw['available_at_ns'][4]
    assert mapping['native_first'].to_list() == [0, 4, 8, 17]
    assert mapping['native_last'].to_list() == [3, 7, 16, 26]
    assert sensor_decisions(delayed, formed)['__trade_available_at_ns__'].equals(mapping['available_at_ns'], check_names=False)
    partition = context.partitions[0]
    ledger = trade_execution(with_predictions(partition, [1.0] * formed.height), context.policy)
    first_fill = raw['available_at_ns'][4] + lag * 1_000_000_000
    assert ledger.fills['time_ns'][0] == first_fill
    assert ledger.fills['reference_price'][0] == execution.filter(pl.col('start_ns') == first_fill)['open'][0]
    assert context.native_support[0].drop('row_id').equals(delayed.select('start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns'))
    assert context.native_support[0]['row_id'].to_list() == execution['row_id'].to_list()
    assert raw.equals(original)
    assert execution.equals(original_execution)


def test_nonmax_native_clock_remains_bound(recorded):
    raw, execution = recorded
    original, _ = _prepare(raw, execution)
    changed, _ = _prepare(_delay(raw, raw['available_at_ns'][1]), execution)
    assert changed.model_rows[0].equals(original.model_rows[0])
    assert changed.native_support[0]['available_at_ns'][0] != original.native_support[0]['available_at_ns'][0]
    assert contract_digest(export_trade_contract(changed.policy, changed.partitions[0])) != contract_digest(export_trade_contract(original.policy, original.partitions[0]))


def test_equal_decision_clocks_preserve_model_identity(recorded):
    raw, execution = recorded
    context, formed = _prepare(_delay(raw, raw['available_at_ns'][7]), execution)
    mapping = context.model_rows[0]
    assert mapping['available_at_ns'][0] == mapping['available_at_ns'][1]
    assert mapping['row_id'].to_list() == [f'partition:0:bar:{index}' for index in range(formed.height)]
    ledger = trade_execution(with_predictions(context.partitions[0], [1.0] * formed.height), context.policy)
    assert ledger.fills['time_ns'][0] == mapping['available_at_ns'][0]


@pytest.mark.parametrize('boundary', ('overtaking', 'outside'))
def test_unsupported_derived_order_refused_by_both_producers(recorded, boundary):
    raw, execution = recorded
    end = raw['available_at_ns'][7] if boundary == 'overtaking' else raw['available_at_ns'][-1]
    delayed = _delay(raw, end + 900_000_000_000)
    formed = volume_bars(delayed, volume_threshold=500)
    with pytest.raises(ValueError, match='Derived decisions'):
        _prepare(delayed, execution, bars=formed)
    with pytest.raises(ValueError, match='Derived decisions'):
        sensor_decisions(delayed, formed)


def test_independent_execution_is_required_for_nonmonotone_receipts(recorded):
    raw, _ = recorded
    with pytest.raises(ValueError, match='Execution availability is out of order'):
        _prepare(_delay(raw, raw['available_at_ns'][4]), None)


@pytest.mark.parametrize('fault', ('null', 'fractional_clock', 'before_end', 'before_open'))
def test_native_clocks_validated_before_maximum(recorded, fault):
    raw, execution = recorded
    if fault == 'null':
        invalid = raw.with_columns(pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col('available_at_ns')).alias('available_at_ns'))
    elif fault == 'fractional_clock':
        invalid = raw.with_columns(pl.col('available_at_ns').cast(pl.Float64))
    elif fault == 'before_end':
        invalid = _delay(raw, raw['end_ns'][0] - 1)
    else:
        invalid = raw.with_columns((pl.col('available_at_ns') + 1).alias('open_available_at_ns'))
    formed = volume_bars(invalid, volume_threshold=500)
    for produce in (lambda: _prepare(invalid, execution, bars=formed), lambda: sensor_decisions(invalid, formed)):
        with pytest.raises(ValueError, match='Native support'):
            produce()


@pytest.mark.parametrize('count', (0, -1, 1.5, True, None))
def test_invalid_constituent_counts_are_not_coerced(recorded, count):
    raw, execution = recorded
    formed = volume_bars(raw, volume_threshold=500)
    values = [count, *formed['bar_count'].to_list()[1:]]
    if isinstance(count, bool):
        values = [count] * formed.height
    invalid = formed.with_columns(pl.Series('bar_count', values, strict=False))
    for produce in (lambda: _prepare(raw, execution, bars=invalid), lambda: sensor_decisions(raw, invalid)):
        with pytest.raises(ValueError, match='membership'):
            produce()


@pytest.mark.parametrize('fault', ('missing_count', 'overlap', 'foreign_anchor'))
def test_ambiguous_support_is_not_guessed(recorded, fault):
    raw, execution = recorded
    formed = volume_bars(raw, volume_threshold=500)
    if fault == 'missing_count':
        invalid = formed.drop('bar_count')
    elif fault == 'overlap':
        invalid = pl.concat([formed.head(1), formed])
    else:
        invalid = formed.with_columns(pl.col('datetime').dt.offset_by('1s'))
    for produce in (lambda: _prepare(raw, execution, bars=invalid), lambda: sensor_decisions(raw, invalid)):
        with pytest.raises(ValueError, match=r'support|membership'):
            produce()


@pytest.mark.parametrize('kind', ('ml', 'rule'))
def test_active_formation_matches_manifest_and_sensor_clocks(recorded, kind):
    source = pl.read_parquet(Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet', n_rows=120)
    execution = normalize_observations(source, interval_seconds=900)
    raw = source.hstack(execution.select('start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns'))
    delayed = _delay(raw, raw['available_at_ns'][4])
    manifest = MLManifest() if kind == 'ml' else RuleBasedManifest().with_strategy(
        [{'id': 'up', 'type': 'relative', 'column': 'high', 'operator': '>', 'other_column': 'open'}], 'up')
    manifest.set_data_source(lambda: delayed, params={'kline_size': 900}).set_split_config(6, 2, 2)
    manifest.with_target_label('close', IdentityTarget)
    manifest.set_bar_formation(volume_bars, volume_threshold=500)
    manifest.set_backtest_config(prediction_mode='target_exposure',
        product=ProductConfig('linear_perpetual', 'BTCUSDT', 'BTC', 'USDT', 1e-9, 0),
        execution_data_source=DataSourceConfig(lambda: execution))
    params = {'bar_type': 'volume'}
    prepared = manifest.prepare_data(delayed, params)
    context = prepared['_trade_context']
    native_train = delayed.head(context.native_support[0].height)
    sensor, _ = manifest.sensor_input_prep(native_train, prepared['_fitted_params'], params)
    expected = context.model_rows[0].select('datetime', 'available_at_ns')
    actual = sensor.select('datetime', pl.col('__trade_available_at_ns__').alias('available_at_ns'))
    assert actual.equals(expected)
    assert actual['available_at_ns'][0] >= delayed['available_at_ns'][0]
    assert all(name not in sensor.columns for name in ('native_first', 'native_last', 'start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns'))
    invalid = _delay(raw, raw['available_at_ns'][-1] + 1)
    with patch('limen.targets.IdentityTarget.__init__', side_effect=AssertionError('Invalid timing reached target fitting')), pytest.raises(ValueError, match='Derived decisions'):
        manifest.prepare_data(invalid, params)


def test_custom_formation_cannot_hide_a_constituent(recorded):
    raw, execution = recorded
    called = []

    def hidden_selection(frame):
        called.append(True)
        return volume_bars(frame.filter(pl.int_range(pl.len()) != 2), volume_threshold=500)

    manifest = MLManifest().set_data_source(lambda: raw, params={'kline_size': 900}).set_split_config(6, 2, 2)
    manifest.with_target_label('close', IdentityTarget).set_bar_formation(hidden_selection)
    manifest.set_backtest_config(prediction_mode='target_exposure',
        product=ProductConfig('linear_perpetual', 'BTCUSDT', 'BTC', 'USDT', 1e-9, 0),
        execution_data_source=DataSourceConfig(lambda: execution))
    with pytest.raises(ValueError, match='recoverable contiguous native support'):
        manifest.prepare_data(raw, {'bar_type': 'custom'})
    assert called == []


def test_recorded_point_support_uses_integer_nanosecond_maxima():
    inputs, _ = _ticks()
    points = inputs.observations.head(6)
    raw = points.with_columns(pl.from_epoch('start_ns', time_unit='ns').dt.replace_time_zone('UTC').alias('datetime'))
    paired = pl.DataFrame([
        {'datetime': raw['datetime'][index], 'bar_count': 2, 'open': raw['open'][index],
         'high': raw['high'].slice(index, 2).max(), 'low': raw['low'].slice(index, 2).min(), 'close': raw['close'][index + 1]}
        for index in (0, 2, 4)])
    delayed = _delay(raw, points['available_at_ns'][2])
    context, _ = _prepare(delayed, inputs.observations, bars=paired)
    assert context.model_rows[0]['available_at_ns'][0] == points['available_at_ns'][2]
    assert sensor_decisions(delayed, paired)['__trade_available_at_ns__'].equals(context.model_rows[0]['available_at_ns'], check_names=False)
    ledger = trade_execution(with_predictions(context.partitions[0], [1.0] * paired.height), context.policy)
    assert ledger.fills['time_ns'][0] == points['available_at_ns'][2]
    assert ledger.fills['reference_price'][0] == points['close'][2]


@pytest.mark.parametrize('role', ('start', 'end'))
def test_recorded_gap_and_ordered_clocks_keep_their_native_values(recorded, role):
    original, execution = recorded
    # A declared omission preserves the actual intervals on either side.
    raw = original.filter(pl.int_range(pl.len()) != 2)
    if role == 'end':
        raw = raw.with_columns(pl.from_epoch('end_ns', time_unit='ns').dt.replace_time_zone('UTC').alias('datetime'))
    formed = volume_bars(raw, volume_threshold=500)
    decisions = sensor_decisions(raw, formed, timestamp_role=role)
    native_positions = {instant: index for index, instant in enumerate(raw['datetime'].dt.epoch('ns'))}
    last = [native_positions[instant] + count - 1 for instant, count in zip(formed['datetime'].dt.epoch('ns'), formed['bar_count'], strict=True)]
    assert decisions['__trade_available_at_ns__'].to_list() == [raw['available_at_ns'][index] for index in last]
    config = BacktestConfig(prediction_mode='target_exposure', source_timestamp_role=role,
        source_interval_seconds=900 if role == 'end' else None,
        product=ProductConfig('linear_perpetual', 'BTCUSDT', 'BTC', 'USDT', 1e-9, 0),
        execution_data_source=DataSourceConfig(lambda: execution))
    policy = resolve_trade_policy(config, {})
    context = prepare_trade_context(config, policy, [raw], [formed], {}, frozen_sources=(execution, None))
    assert context.model_rows[0]['available_at_ns'].equals(decisions['__trade_available_at_ns__'], check_names=False)
    assert context.partitions[0].partition_end_ns == original['available_at_ns'][-1]

from pathlib import Path

import polars as pl
import pytest

from limen.backtest.execution_events import with_predictions
from limen.backtest.trade_contract import contract_digest, export_trade_contract
from limen.backtest.trade_execution import trade_execution
from limen.data.bars import volume_bars
from limen.experiment._prepare_trade_context import normalize_observations, prepare_trade_context, sensor_decisions
from limen.experiment._resolve_trade_policy import BacktestConfig, ProductConfig, resolve_trade_policy
from limen.experiment.manifest_core import DataSourceConfig


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

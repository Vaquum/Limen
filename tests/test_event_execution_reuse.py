from dataclasses import replace
import hashlib
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import polars as pl
import numpy as np
import pytest
from polars.testing import assert_frame_equal

from limen.backtest._event_plan import event_plan
from limen.backtest.funding_adapter import funding_adapter, prepare_funding
from limen.backtest.trade_contract import ProductSpec, TradeInputs, TradePolicy, source_binding
from limen.backtest.trade_execution import trade_execution
from limen.experiment._prepare_trade_context import normalize_observations
from limen.experiment._resolve_trade_policy import FundingConfig, resolve_funding
from limen.experiment.manifest_core import DataSourceConfig
from limen.sfd.rule_based.common_strategies import strategy_signals
from tests.backtest_reuse.event_reference import trade_execution as reference_trade_execution
from tests.test_fixed_trade_sources import _funding_frame

ADAPTER_CALLS = []


def recorded_adapter(context, *, params):
    ADAPTER_CALLS.append(context)
    return funding_adapter(context, params=params)


def recorded_inputs():
    path = Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet'
    recorded = pl.read_parquet(path, n_rows=256)
    targets = strategy_signals(recorded, 'ema_cross')
    offset = max(0, int(np.flatnonzero(targets)[0]) - 2)
    market = recorded.slice(offset, 12)
    observations = normalize_observations(market, interval_seconds=900)
    signals = pl.DataFrame({'row_id': [f'signal:{i}' for i in range(market.height)],
                           'available_at_ns': observations['available_at_ns'], 'datetime': market['datetime'],
                           'target': targets[offset:offset + market.height].astype(float)})
    start, end = int(observations['start_ns'][0]), int(observations['available_at_ns'][-1])
    binding = source_binding(observations, str(path), start, end, 900000000000, 'recorded OHLC')
    inputs = TradeInputs(10000.0, start, end, signals, observations, None, (binding,))
    return inputs, TradePolicy(ProductSpec('linear_perpetual', 'BTCUSDT', 'BTC', 'USDT', 1e-12, 0.0))


def test_event_reference_sources_match_prechange():
    expected = {
        'event_reference.py': '3caaa90ff0fb32c47bfda8fa113db615d97241c9f4836ed79e1ce57d65229cb5',
        'execution_events_reference.py': 'c8522b4b45d7bc12a56a11e740b2a556b4dd7c574a13453b59304831e1a57688',
    }
    for name, checksum in expected.items():
        source = (Path(__file__).parent / 'backtest_reuse' / name).read_text().split('# BEGIN FROZEN SOURCE' + chr(10), 1)[1]
        original = source.replace('from tests.backtest_reuse.execution_events_reference import execution_events', 'from limen.backtest.execution_events import execution_events')
        assert hashlib.sha256(original.encode()).hexdigest() == checksum


def assert_ledgers(actual, expected):
    for name in ('states', 'intents', 'fills', 'episodes', 'funding'):
        assert_frame_equal(getattr(actual, name), getattr(expected, name), check_exact=False, rel_tol=0, abs_tol=1e-10)
    assert actual.contract_digest == expected.contract_digest
    assert dict(actual.metrics) == pytest.approx(dict(expected.metrics), rel=0, abs=1e-10)


@pytest.mark.parametrize('holding,barrier,phase', [(None, None, 0), (65, None, 7), (1800, 10, 0)])
def test_timer_blocks_preserve_every_sequential_state(holding, barrier, phase):
    inputs, policy = recorded_inputs()
    policy = replace(policy, timer_interval_seconds=60, timer_phase_utc_seconds=phase,
                     max_holding_seconds=holding, take_profit_bps=barrier, stop_loss_bps=barrier)
    expected = reference_trade_execution(inputs, policy)
    actual = trade_execution(inputs, policy)
    assert_ledgers(actual, expected)
    assert actual.states.height > inputs.observations.height * 10


def recorded_funded_inputs():
    path = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'
    market = pl.read_parquet(path, n_rows=168)
    observations = normalize_observations(market, interval_seconds=3600)
    signals = pl.DataFrame({'row_id': [f'signal:{i}' for i in range(market.height)],
                           'available_at_ns': observations['available_at_ns'], 'target': strategy_signals(market, 'buy_and_hold').astype(float)})
    start, end = int(observations['start_ns'][0]), int(observations['available_at_ns'][-1])
    history = _funding_frame()
    funding = resolve_funding(FundingConfig(preset='binance_btcusdt', data_source=DataSourceConfig(lambda: history)), {})
    assert funding is not None
    events = prepare_funding(funding, history, start, end)
    sources = (source_binding(observations, str(path), start, end, 3600000000000, 'recorded OHLC'),
               source_binding(events, 'recorded Binance Jan2024 funding', start, end, 1, 'recorded realized settlement'))
    inputs = TradeInputs(10000, start, end, signals, observations, events, sources)
    policy = TradePolicy(ProductSpec('linear_perpetual', 'BTCUSDT', 'BTC', 'USDT', 1e-12, 0.0), funding=funding, timer_interval_seconds=600)
    return inputs, policy


@pytest.mark.parametrize('custom', [False, True])
def test_recorded_funding_keeps_every_original_clock_and_cashflow(custom):
    inputs, policy = recorded_funded_inputs()
    if custom:
        assert policy.funding is not None
        policy = replace(policy, funding=replace(policy.funding, adapter_ref=f'{__name__}.recorded_adapter', adapter_version='recorded-wrapper-v1'))
    ADAPTER_CALLS.clear()
    expected = reference_trade_execution(inputs, policy)
    calls = ADAPTER_CALLS.copy()
    ADAPTER_CALLS.clear()
    actual = trade_execution(inputs, policy)
    assert_ledgers(actual, expected)
    assert calls == ADAPTER_CALLS
    assert actual.states.height > inputs.observations.height
    if custom:
        assert len(calls) == actual.funding.height > 0


def test_equal_sources_reuse_plan_but_targets_remain_fresh():
    inputs, policy = recorded_inputs()
    module = import_module('limen.backtest._event_plan')
    module._PLANS.clear()
    with patch.object(module, 'source_execution_events', wraps=module.source_execution_events) as compile_events:
        original = trade_execution(inputs, policy)
        clone = replace(inputs, observations=inputs.observations.clone(), signals=inputs.signals.clone())
        assert_ledgers(trade_execution(clone, policy), original)
        changed = replace(clone, signals=clone.signals.with_columns(-pl.col('target')))
        assert_ledgers(trade_execution(changed, policy), reference_trade_execution(changed, policy))
        assert compile_events.call_count == 1
        invalid = replace(inputs, signals=inputs.signals.with_columns(pl.lit(float('nan')).alias('target')))
        with pytest.raises(ValueError, match='finite'):
            trade_execution(invalid, policy)


def test_cached_plan_revalidates_source_content_and_identity():
    inputs, policy = recorded_inputs()
    original = trade_execution(inputs, policy)
    changed_identity = replace(inputs, sources=(replace(inputs.sources[0], identity='other recorded source identity'),))
    changed = trade_execution(changed_identity, policy)
    assert changed.contract_digest != original.contract_digest
    assert changed.states['equity'].equals(original.states['equity'])
    invalid = replace(inputs, sources=(replace(inputs.sources[0], checksum='changed'),))
    with pytest.raises(ValueError, match='fingerprint changed'):
        trade_execution(invalid, policy)
    observations = inputs.observations.clone()
    trade_execution(replace(inputs, observations=observations), policy)
    opened, closed = observations['open'].clone(), observations['close'].clone()
    observations.replace_column(observations.get_column_index('open'), closed.rename('open'))
    observations.replace_column(observations.get_column_index('close'), opened.rename('close'))
    with pytest.raises(ValueError, match='fingerprint changed'):
        trade_execution(replace(inputs, observations=observations), policy)


@pytest.mark.parametrize('field,invalid,message', [
    ('initial_equity', '10000.0', 'initial equity must be a finite number'),
    ('partition_start_ns', 'numpy', 'Partition timestamps must be integer'),
    ('partition_end_ns', 'numpy', 'Partition timestamps must be integer'),
])
def test_cached_plan_revalidates_scalar_types(field, invalid, message):
    inputs, policy = recorded_inputs()
    trade_execution(inputs, policy)
    value = np.int64(getattr(inputs, field)) if invalid == 'numpy' else invalid
    with pytest.raises(ValueError, match=message):
        trade_execution(replace(inputs, **{field: value}), policy)


def test_narrow_partition_retains_only_referenced_source_rows():
    inputs, policy = recorded_inputs()
    end = int(inputs.observations['available_at_ns'][1])
    signals = inputs.signals.filter(pl.col('available_at_ns') <= end)
    plan = event_plan(replace(inputs, partition_end_ns=end, signals=signals), policy)
    referenced = {event.source_row_id for event in plan.events if event.kind == 'observation'}
    assert set(plan.observations) == referenced
    assert set(plan.observation_order) == referenced
    assert len(plan.observations) < inputs.observations.height


def assert_rejected_like_reference(inputs, policy):
    with pytest.raises(ValueError) as expected:
        reference_trade_execution(inputs, policy)
    with pytest.raises(ValueError) as actual:
        trade_execution(inputs, policy)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize('change', ['clock_order', 'partition_end', 'precision'])
def test_warm_plan_rejects_changed_clock_partition_and_precision(change):
    inputs, policy = recorded_inputs()
    trade_execution(inputs, policy)
    if change == 'clock_order':
        inputs = replace(inputs, signals=inputs.signals.reverse())
    elif change == 'partition_end':
        inputs = replace(inputs, partition_end_ns=inputs.partition_end_ns - 1)
    else:
        policy = replace(policy, max_price_gap_seconds=899)
    assert_rejected_like_reference(inputs, policy)


def test_warm_plan_rejects_changed_recorded_funding():
    inputs, policy = recorded_funded_inputs()
    trade_execution(inputs, policy)
    events = inputs.funding_events
    assert events is not None
    events.replace_column(events.get_column_index('rate_decimal'), events['rate_decimal'].reverse())
    assert_rejected_like_reference(inputs, policy)


@pytest.mark.parametrize('source_index', [0, 1])
def test_uncached_plan_preserves_source_validation_and_existing_cache(source_index):
    inputs, policy = recorded_funded_inputs()
    module = import_module('limen.backtest._event_plan')
    expected = event_plan(inputs, policy)
    cached = list(module._PLANS.items())
    with patch.object(module, '_plan_key', side_effect=AssertionError('single-use plan hashed for caching')):
        actual = event_plan(inputs, policy, cache=False)
        assert actual == expected and actual is not expected
        sources = tuple(replace(source, checksum='changed') if index == source_index else source
                        for index, source in enumerate(inputs.sources))
        with pytest.raises(ValueError, match='fingerprint changed'):
            event_plan(replace(inputs, sources=sources), policy, cache=False)
    assert list(module._PLANS.items()) == cached

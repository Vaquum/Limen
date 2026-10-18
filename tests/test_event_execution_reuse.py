from dataclasses import replace
from importlib import import_module
from itertools import groupby
from pathlib import Path
from unittest.mock import patch

import polars as pl
import numpy as np
import pytest
from polars.testing import assert_frame_equal

from limen.backtest._event_plan import _event_plan
from limen.backtest.execution_events import execution_events
from limen.backtest.funding_adapter import prepare_funding, resolve_adapter
from limen.backtest.trade_contract import ProductSpec, TradeInputs, TradeLedger, TradePolicy, source_binding
from limen.backtest.trade_execution import _Account, _EPISODE_SCHEMA, _FILL_SCHEMA, _FUNDING_SCHEMA, _INTENT_SCHEMA, _STATE_SCHEMA, _execute_batch, _frame, _metrics, trade_execution
from limen.experiment._prepare_trade_context import normalize_observations
from limen.experiment._resolve_trade_policy import FundingConfig, resolve_funding
from limen.experiment.manifest_core import DataSourceConfig
from limen.features import dollar_bar_crash_reversal
from tests.test_fixed_trade_sources import _funding_frame


def recorded_inputs():
    path = Path(__file__).parent / 'fixtures/dollar_bar_crash_reversal_15m.parquet'
    featured = dollar_bar_crash_reversal(pl.read_parquet(path), -575.0, -0.5, 60)
    assert isinstance(featured, pl.DataFrame)
    active = featured.with_row_index('index').filter(pl.col('dollar_bar_crash_reversal_position') > 0)['index'][0]
    market = featured.slice(max(0, int(active) - 2), 12)
    observations = normalize_observations(market, interval_seconds=900)
    signals = pl.DataFrame({'row_id': [f'signal:{i}' for i in range(market.height)],
                           'available_at_ns': observations['available_at_ns'], 'datetime': market['datetime'],
                           'target': market['dollar_bar_crash_reversal_position'].cast(pl.Float64)})
    start, end = int(observations['start_ns'][0]), int(observations['available_at_ns'][-1])
    binding = source_binding(observations, str(path), start, end, 900000000000, 'recorded OHLC')
    inputs = TradeInputs(10000.0, start, end, signals, observations, None, (binding,))
    return inputs, TradePolicy(ProductSpec('linear_perpetual', 'BTCUSDT', 'BTC', 'USDT', 1e-12, 0.0))


def sequential_ledger(inputs, policy):
    plan = _event_plan(inputs, policy)
    account = _Account(policy, inputs.initial_equity, plan.binding[:16])
    signals = dict(zip(inputs.signals['row_id'], inputs.signals['target'], strict=True))
    adapter = resolve_adapter(policy.funding) if policy.funding is not None else None
    active, previous, price = None, inputs.partition_start_ns, 0.0
    for time_ns, batch in groupby(execution_events(inputs, policy), key=lambda event: event.time_ns):
        group = tuple(batch)
        active, price = _execute_batch(account, plan, signals, adapter, active, previous, price, time_ns, group)
        previous = time_ns
        account.record(time_ns, price, group[-1].event_id, any(event.kind == 'signal' for event in group))
        account.states[-1]['absolute_exposure'] = abs(account.quantity) * price / account.equity(price)
    states = _frame(account.states, {**_STATE_SCHEMA, 'absolute_exposure': pl.Float64})
    states = states.with_columns((pl.col('equity') / pl.col('equity').shift(1).fill_null(inputs.initial_equity) - 1).alias('net_return'))
    return TradeLedger(states, _frame(account.intents, _INTENT_SCHEMA), _frame(account.fills, _FILL_SCHEMA), _frame(account.episodes, _EPISODE_SCHEMA), _frame(account.funding, _FUNDING_SCHEMA), _metrics(account, inputs.initial_equity, states), plan.binding)


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
    expected = sequential_ledger(inputs, policy)
    actual = trade_execution(inputs, policy)
    assert_ledgers(actual, expected)
    assert actual.states.height > inputs.observations.height * 10


def test_recorded_funding_keeps_every_original_clock_and_cashflow():
    path = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'
    market = pl.read_parquet(path, n_rows=168)
    observations = normalize_observations(market, interval_seconds=3600)
    signals = pl.DataFrame({'row_id': [f'signal:{i}' for i in range(market.height)],
                           'available_at_ns': observations['available_at_ns'], 'target': [1.0] * market.height})
    start, end = int(observations['start_ns'][0]), int(observations['available_at_ns'][-1])
    history = _funding_frame()
    funding = resolve_funding(FundingConfig(preset='binance_btcusdt', data_source=DataSourceConfig(lambda: history)), {})
    assert funding is not None
    events = prepare_funding(funding, history, start, end)
    sources = (source_binding(observations, str(path), start, end, 3600000000000, 'recorded OHLC'),
               source_binding(events, 'recorded Binance Jan2024 funding', start, end, 1, 'recorded realized settlement'))
    inputs = TradeInputs(10000, start, end, signals, observations, events, sources)
    policy = TradePolicy(ProductSpec('linear_perpetual', 'BTCUSDT', 'BTC', 'USDT', 1e-12, 0.0), funding=funding, timer_interval_seconds=600)
    assert_ledgers(trade_execution(inputs, policy), sequential_ledger(inputs, policy))


def test_equal_sources_reuse_plan_but_targets_remain_fresh():
    inputs, policy = recorded_inputs()
    module = import_module('limen.backtest._event_plan')
    module._PLANS.clear()
    with patch.object(module, '_execution_events', wraps=module._execution_events) as compile_events:
        original = trade_execution(inputs, policy)
        clone = replace(inputs, observations=inputs.observations.clone(), signals=inputs.signals.clone())
        assert_ledgers(trade_execution(clone, policy), original)
        changed = replace(clone, signals=clone.signals.with_columns(-pl.col('target')))
        assert_ledgers(trade_execution(changed, policy), sequential_ledger(changed, policy))
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
    plan = _event_plan(replace(inputs, partition_end_ns=end, signals=signals), policy)
    referenced = {event.source_row_id for event in plan.events if event.kind == 'observation'}
    assert set(plan.observations) == referenced
    assert set(plan.observation_order) == referenced
    assert len(plan.observations) < inputs.observations.height

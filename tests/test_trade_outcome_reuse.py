from bisect import bisect_left
from dataclasses import replace
from importlib import import_module
from unittest.mock import patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from limen.backtest.trade_contract import contract_digest, export_trade_contract, source_binding
from limen.backtest.trade_execution import trade_execution
from limen.targets.trade_outcome import TradeOutcomeTarget, TradeTargetContext
from tests.backtest_reuse.event_reference import trade_execution as reference_trade_execution
from tests.test_event_execution_reuse import assert_ledgers, recorded_inputs

OUTCOME_SCHEMA = {'row_id': pl.String}
for _side in ('long', 'short'):
    OUTCOME_SCHEMA.update({f'{_side}_return': pl.Float64, f'{_side}_available': pl.Boolean,
                           f'{_side}_entry_ns': pl.Int64, f'{_side}_exit_ns': pl.Int64,
                           f'{_side}_label_available_ns': pl.Int64, f'{_side}_exit_reason': pl.String})


def full_episode(inputs, policy, index, side):
    available = inputs.signals['available_at_ns'].to_list()
    first = max(0, bisect_left(inputs.observations['available_at_ns'].to_list(), available[index]) - 1)
    observations = inputs.observations.slice(first)
    start = max(inputs.partition_start_ns, int(observations['start_ns'][0]))
    binding = source_binding(observations, 'candidate:prices', start, inputs.partition_end_ns, 1, 'recorded_candidate_execution')
    candidate = replace(inputs, partition_start_ns=start, observations=observations,
                        signals=inputs.signals.slice(index).with_columns(pl.lit(side).alias('target')),
                        sources=(*inputs.sources, binding))
    ledger = reference_trade_execution(candidate, policy)
    completed = ledger.episodes.filter(pl.col('closed_at_ns').is_not_null())
    if not completed.height:
        return {'return': None, 'available': False, 'entry_ns': None if not ledger.episodes.height else ledger.episodes['first_fill_ns'][0], 'exit_ns': None, 'label_available_ns': None, 'exit_reason': None}
    assert completed.height == 1
    episode = completed.row(0, named=True)
    return {'return': episode['net_pnl'] / episode['entry_equity'], 'available': True,
            'entry_ns': episode['first_fill_ns'], 'exit_ns': episode['closed_at_ns'],
            'label_available_ns': episode['closed_at_ns'], 'exit_reason': episode['exit_reason']}


def expected_outcomes(inputs, policy):
    expected = []
    for index, identity in enumerate(inputs.signals['row_id']):
        row = {'row_id': identity}
        for side, sign in (('long', 1.0), ('short', -1.0)):
            row.update({f'{side}_{name}': value for name, value in full_episode(inputs, policy, index, sign).items()})
        expected.append(row)
    return pl.DataFrame(expected, schema=OUTCOME_SCHEMA)


@pytest.mark.parametrize('holding,timer,lag', [(None, None, 0), (1800, None, 0), (65, 60, 7)])
def test_early_episode_labels_equal_complete_suffix_execution(holding, timer, lag):
    inputs, policy = recorded_inputs()
    policy = replace(policy, take_profit_bps=10, stop_loss_bps=10,
                     max_holding_seconds=holding, timer_interval_seconds=timer, execution_lag_seconds=lag)
    context = TradeTargetContext(policy, inputs, inputs.partition_start_ns, inputs.partition_end_ns, contract_digest(export_trade_contract(policy, inputs)))
    frame = inputs.signals.select('datetime')
    target = TradeOutcomeTarget(frame, 'outcome', trade_context=context)
    result = target.transform(frame, trade_context=context)
    expected = expected_outcomes(inputs, policy)
    assert_frame_equal(target.outcomes.rows, expected, check_exact=False, rel_tol=0, abs_tol=1e-10)
    assert_frame_equal(result, frame.with_columns((expected['long_return'] > 0).cast(pl.Int8).alias('outcome')))
    assert target.outcomes.contract_digest == context.contract_digest


def test_candidate_does_not_rebuild_or_materialize_suffix_ledgers():
    inputs, policy = recorded_inputs()
    policy = replace(policy, take_profit_bps=10, stop_loss_bps=10)
    context = TradeTargetContext(policy, inputs, inputs.partition_start_ns, inputs.partition_end_ns, contract_digest(export_trade_contract(policy, inputs)))
    frame = inputs.signals.select('datetime')
    module = import_module('limen.backtest.trade_execution')
    with patch.object(module, '_metrics', side_effect=AssertionError('unnecessary full ledger')):
        target = TradeOutcomeTarget(frame, 'outcome', trade_context=context)
        target.transform(frame, trade_context=context)
    assert target.outcomes.rows['long_available'].sum() > 0
    assert target.outcomes.rows['short_available'].sum() > 0


@pytest.mark.parametrize('mechanism', ['discrete', 'continuous', 'recorded'])
@pytest.mark.parametrize('case', ['closed', 'open_tail', 'never_filled', 'barriers'])
def test_funded_candidate_plan_reuse_preserves_full_ledger(mechanism, case):
    from tests.test_funding_adapter import _funded
    from tests.test_event_execution_reuse import ADAPTER_CALLS, recorded_adapter, recorded_funded_inputs

    if mechanism == 'recorded':
        inputs, policy = recorded_funded_inputs()
        observations = inputs.observations.head(12)
        end = int(observations['available_at_ns'][-1])
        funding = inputs.funding_events.filter(pl.col('time_ns') <= end)
        times = pl.from_epoch(observations['start_ns'], time_unit='ns').alias('datetime')
        sources = (*inputs.sources, source_binding(observations, 'recorded first twelve hours', inputs.partition_start_ns, end, 1, 'recorded OHLC'), source_binding(funding, 'recorded funding', inputs.partition_start_ns, end, 1, 'recorded realized settlement'))
        inputs = replace(inputs, partition_end_ns=end, observations=observations, signals=inputs.signals.head(12).with_columns(times), funding_events=funding, sources=sources)
        policy = replace(policy, max_holding_seconds=1800)
    else:
        inputs, policy = recorded_inputs()
        inputs, policy = _funded(inputs, replace(policy, max_holding_seconds=1800), mechanism=mechanism)
    policy = replace(policy, prediction_mode='target_exposure', notional_rate=1.0, funding=replace(policy.funding, adapter_ref=f'{recorded_adapter.__module__}.recorded_adapter', adapter_version='recorded-wrapper-v1'))
    if case == 'open_tail':
        policy = replace(policy, max_holding_seconds=1e9)
    elif case == 'never_filled':
        policy = replace(policy, product=replace(policy.product, min_notional=1e9))
    elif case == 'barriers':
        policy = replace(policy, take_profit_bps_column='tp', stop_loss_bps_column='sl')
        inputs = replace(inputs, signals=inputs.signals.with_columns(pl.lit(10.0).alias('tp'), pl.lit(10.0).alias('sl'), pl.lit(True).alias('__barrier_ready__')))
    frame = inputs.signals.select('datetime')
    context = TradeTargetContext(policy, inputs, inputs.partition_start_ns, inputs.partition_end_ns, contract_digest(export_trade_contract(policy, inputs)))
    # Compare each original clipped candidate's complete ledger and adapter calls.
    from limen.targets.trade_outcome import _Timeline, _candidate
    from limen.backtest._event_plan import event_plan
    module = import_module('limen.targets.trade_outcome')
    original_execute = module.execute_event_plan
    expected = []
    calls = []
    required_plans = []

    def full_ledger(account, plan, signals, start, end, adapter, **kwargs):
        required_plans.append(any(row['kind'] == 'accrual' for row in inputs.funding_events.iter_rows(named=True)))
        observations = inputs.observations.filter(pl.col('row_id').is_in(plan.observation_order))
        funding = inputs.funding_events.filter(pl.col('event_id').is_in(plan.funding))
        selected = inputs.signals.filter(pl.col('row_id').is_in(plan.signal_order)).with_columns(pl.Series('target', [signals[str(identity)]['target'] if isinstance(signals[str(identity)], dict) else signals[str(identity)] for identity in inputs.signals['row_id'] if str(identity) in plan.signal_order]))
        sources = (*inputs.sources, source_binding(observations, 'oracle:prices', start, end, 1, 'recorded candidate'), source_binding(funding, 'oracle:funding', start, end, 1, 'recorded candidate'))
        candidate = replace(inputs, partition_start_ns=start, partition_end_ns=end, observations=observations, funding_events=funding, signals=selected, sources=sources)
        ADAPTER_CALLS.clear()
        ledger = trade_execution(candidate, policy)
        expected_calls = ADAPTER_CALLS.copy()
        calls.extend(expected_calls)
        ADAPTER_CALLS.clear()
        original_execute(account, plan, signals, start, end, adapter, **kwargs)
        assert expected_calls == ADAPTER_CALLS
        for actual, expected_episode in zip(account.episodes, ledger.episodes.to_dicts(), strict=True):
            # Open episodes have no outcome label; only the full ledger marks
            # their unused PnL to the final state in its aggregate metrics.
            omitted = {'episode_id'} if expected_episode['closed_at_ns'] is not None else {'episode_id', 'net_pnl'}
            assert {key: value for key, value in actual.items() if key not in omitted} == {key: value for key, value in expected_episode.items() if key not in omitted}

    plan = event_plan(inputs, policy)
    timeline = _Timeline(sorted({event.time_ns for event in plan.events if event.kind == 'observation'}), inputs.signals['available_at_ns'].to_list(), inputs.observations['start_ns'].to_list(), inputs.observations['available_at_ns'].to_list(), plan)
    with patch.object(_Timeline, 'has_accrual', True), patch.object(module, 'execute_event_plan', side_effect=full_ledger):
        for index, identity in enumerate(inputs.signals['row_id']):
            row = {'row_id': identity}
            for side, sign in [('long', 1.0), ('short', -1.0)]:
                row.update({f'{side}_{key}': value for key, value in _candidate(inputs, policy, index, sign, timeline).items()})
            expected.append(row)
    engine = import_module('limen.backtest.trade_execution')
    ADAPTER_CALLS.clear()
    plans = import_module('limen.backtest._event_plan')
    with patch.object(module, 'event_plan', wraps=module.event_plan) as compile_plan, patch.object(engine, '_metrics', side_effect=AssertionError('unnecessary full ledger')), patch.object(plans, '_plan_key', wraps=plans._plan_key) as cache_key:
        target = TradeOutcomeTarget(frame, 'outcome', trade_context=context)
        target.transform(frame, trade_context=context)
    assert compile_plan.call_count == 1 + sum(required_plans) // 2
    assert cache_key.call_count == 1
    assert calls == ADAPTER_CALLS
    assert_frame_equal(target.outcomes.rows, pl.DataFrame(expected, schema=target.outcomes.rows.schema), check_exact=True)


def test_label_fast_path_still_validates_late_source_rows():
    inputs, policy = recorded_inputs()
    policy = replace(policy, take_profit_bps=10, stop_loss_bps=10)
    prices = inputs.observations.with_columns(pl.when(pl.int_range(pl.len()) == pl.len() - 1).then(pl.col('high')).otherwise(pl.col('low')).alias('low'))
    binding = source_binding(prices, 'recorded source with invalid bounds', inputs.partition_start_ns, inputs.partition_end_ns, 1, 'invalid OHLC contract')
    inputs = replace(inputs, observations=prices, sources=(binding,))
    context = TradeTargetContext(policy, inputs, inputs.partition_start_ns, inputs.partition_end_ns, contract_digest(export_trade_contract(policy, inputs)))
    frame = inputs.signals.select('datetime')
    with pytest.raises(ValueError, match='OHLC'):
        TradeOutcomeTarget(frame, 'outcome', trade_context=context).transform(frame, trade_context=context)


@pytest.mark.parametrize('side', [1.0, -1.0])
def test_constant_candidate_signals_cannot_reopen_forced_exit(side):
    inputs, policy = recorded_inputs()
    policy = replace(policy, take_profit_bps=10, stop_loss_bps=10)
    inputs = replace(inputs, signals=inputs.signals.with_columns(pl.lit(side).alias('target')))
    expected = reference_trade_execution(inputs, policy)
    actual = trade_execution(inputs, policy)
    assert_ledgers(actual, expected)
    assert expected.episodes.height == expected.metrics['completed_trades'] == 1
    closed = int(expected.episodes['closed_at_ns'][0])
    later = expected.states.filter(pl.col('time_ns') > closed)
    assert later.height > 1
    assert later['forced_exit_latched'].all()
    assert later['quantity'].eq(0).all()
    assert expected.intents.filter(pl.col('reason') == 'signal').height == 1


@pytest.mark.parametrize('case', ['never_filled', 'open_tail', 'sparse_lag'])
def test_candidate_edge_outcomes_equal_frozen_reference(case):
    inputs, policy = recorded_inputs()
    policy = replace(policy, take_profit_bps=10, stop_loss_bps=10)
    if case == 'never_filled':
        policy = replace(policy, product=replace(policy.product, min_notional=1e9))
    elif case == 'open_tail':
        policy = replace(policy, take_profit_bps=None, stop_loss_bps=None, max_holding_seconds=1e9)
    else:
        observations = inputs.observations[[0, 3, 6, 9, 11]]
        binding = source_binding(observations, 'recorded sparse OHLC', inputs.partition_start_ns, inputs.partition_end_ns, 900000000000, 'recorded sparse intervals')
        inputs = replace(inputs, observations=observations, sources=(binding,))
        policy = replace(policy, execution_lag_seconds=1800)
    context = TradeTargetContext(policy, inputs, inputs.partition_start_ns, inputs.partition_end_ns, contract_digest(export_trade_contract(policy, inputs)))
    frame = inputs.signals.select('datetime')
    target = TradeOutcomeTarget(frame, 'outcome', trade_context=context, side='short', output='net_return')
    result = target.transform(frame, trade_context=context)
    expected = expected_outcomes(inputs, policy)
    assert_frame_equal(target.outcomes.rows, expected, check_exact=False, rel_tol=0, abs_tol=1e-10)
    assert_frame_equal(result, frame.with_columns(expected['short_return'].alias('outcome')), check_exact=False, rel_tol=0, abs_tol=1e-10)
    assert target.outcomes.contract_digest == context.contract_digest
    if case == 'never_filled':
        assert expected['short_entry_ns'].null_count() == expected.height
    elif case == 'open_tail':
        assert expected['short_available'].sum() == 0
        assert expected['short_entry_ns'].null_count() < expected.height
    else:
        assert expected['short_entry_ns'].null_count() > 0
        complete = expected.filter(pl.col('short_available'))
        assert complete.height > 0
        assert complete['short_label_available_ns'].equals(complete['short_exit_ns'].rename('short_label_available_ns'))

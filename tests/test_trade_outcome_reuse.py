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
from tests.test_event_execution_reuse import recorded_inputs


def full_episode(inputs, policy, index, side):
    available = inputs.signals['available_at_ns'].to_list()
    first = max(0, bisect_left(inputs.observations['available_at_ns'].to_list(), available[index]) - 1)
    observations = inputs.observations.slice(first)
    start = max(inputs.partition_start_ns, int(observations['start_ns'][0]))
    binding = source_binding(observations, 'candidate:prices', start, inputs.partition_end_ns, 1, 'recorded_candidate_execution')
    candidate = replace(inputs, partition_start_ns=start, observations=observations,
                        signals=inputs.signals.slice(index).with_columns(pl.lit(side).alias('target')),
                        sources=(*inputs.sources, binding))
    ledger = trade_execution(candidate, policy)
    completed = ledger.episodes.filter(pl.col('closed_at_ns').is_not_null())
    if not completed.height:
        return {'return': None, 'available': False, 'entry_ns': None if not ledger.episodes.height else ledger.episodes['first_fill_ns'][0], 'exit_ns': None, 'label_available_ns': None, 'exit_reason': None}
    episode = completed.row(0, named=True)
    return {'return': episode['net_pnl'] / episode['entry_equity'], 'available': True,
            'entry_ns': episode['first_fill_ns'], 'exit_ns': episode['closed_at_ns'],
            'label_available_ns': episode['closed_at_ns'], 'exit_reason': episode['exit_reason']}


@pytest.mark.parametrize('holding,timer,lag', [(None, None, 0), (1800, None, 0), (65, 60, 7)])
def test_early_episode_labels_equal_complete_suffix_execution(holding, timer, lag):
    inputs, policy = recorded_inputs()
    policy = replace(policy, take_profit_bps=10, stop_loss_bps=10,
                     max_holding_seconds=holding, timer_interval_seconds=timer, execution_lag_seconds=lag)
    context = TradeTargetContext(policy, inputs, inputs.partition_start_ns, inputs.partition_end_ns, contract_digest(export_trade_contract(policy, inputs)))
    frame = inputs.signals.select('datetime')
    target = TradeOutcomeTarget(frame, 'outcome', trade_context=context)
    target.transform(frame, trade_context=context)
    expected = []
    for index, identity in enumerate(inputs.signals['row_id']):
        row = {'row_id': identity}
        for side, sign in (('long', 1.0), ('short', -1.0)):
            row.update({f'{side}_{name}': value for name, value in full_episode(inputs, policy, index, sign).items()})
        expected.append(row)
    assert_frame_equal(target.outcomes.rows, pl.DataFrame(expected, schema=target.outcomes.rows.schema), check_exact=False, rel_tol=0, abs_tol=1e-10)


def test_candidate_does_not_rebuild_or_materialize_suffix_ledgers():
    inputs, policy = recorded_inputs()
    policy = replace(policy, take_profit_bps=10, stop_loss_bps=10)
    context = TradeTargetContext(policy, inputs, inputs.partition_start_ns, inputs.partition_end_ns, contract_digest(export_trade_contract(policy, inputs)))
    frame = inputs.signals.select('datetime')
    module = import_module('limen.targets.trade_outcome')
    with patch.object(module, 'trade_execution', side_effect=AssertionError('unnecessary full ledger')):
        target = TradeOutcomeTarget(frame, 'outcome', trade_context=context)
        target.transform(frame, trade_context=context)
    assert target.outcomes.rows['long_available'].sum() > 0
    assert target.outcomes.rows['short_available'].sum() > 0


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

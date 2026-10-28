from pathlib import Path
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from limen.backtest._snapshot_execution import snapshot_with_execution
from limen.backtest.backtest_snapshot import backtest_snapshot
from limen.backtest.long_flat_strategy import long_flat_strategy
from limen.backtest._event_plan import event_plan
from limen.backtest.trade_contract import contract_digest, export_trade_contract
from limen.backtest.trade_execution import trade_execution
from tests.test_event_execution_reuse import recorded_inputs


@pytest.fixture(scope='module')
def market():
    return pl.read_parquet(Path(__file__).parent / 'fixtures/spot_1h_20240101_20240116.parquet').head(12)


def _execute(market, entries, exits, **options):
    columns = {**market.head(len(entries)).select('open', 'high', 'low', 'close').to_dict(as_series=False),
               'predictions': entries, 'exit': exits}
    columns['price_change'] = np.asarray(columns['close'], dtype=float) - np.asarray(columns['open'], dtype=float)
    return snapshot_with_execution(columns, exit_col='exit', fee_bps=0.0, slip_bps=0.0, **options)


def test_snapshot_entry_holds_until_independent_exit(market):
    _, result = _execute(market, [1, 0, 0, 0, 0, 0], [False, False, False, True, False, False])
    assert result.pos.tolist() == [0, 1, 1, 1, 0, 0]
    assert result.gross[1:4] == pytest.approx(market['close'].to_numpy()[1:4] / market['close'].to_numpy()[:3] - 1)


def test_snapshot_exit_requires_a_post_closure_false_before_reentry(market):
    _, result = _execute(market, [1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
                         [False, False, True, False, False, True, False, False, False, False])
    assert result.pos.tolist() == [0, 1, 1, 0, 0, 1, 0, 0, 0, 1]


def test_snapshot_pending_entry_cancellation_and_rule_lag(market):
    _, result = _execute(market, [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                         [False, True, False, False, False, False, False, True, False, False, False, False],
                         execution_lag_bars=3)
    assert result.pos.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]


def test_snapshot_flat_exit_veto_preserves_initial_arming(market):
    _, result = _execute(market, [1, 1, 1, 0], [True, False, False, False])
    assert result.pos.tolist() == [0, 0, 1, 1]


@pytest.mark.parametrize('lag', [0, -1, True, 1.5])
def test_snapshot_explicit_exit_lag_refusals(market, lag):
    with pytest.raises(ValueError, match='positive integer'):
        _execute(market, [1, 0], [False, False], execution_lag_bars=lag)


def test_snapshot_explicit_exit_evidence_and_price_refusals(market):
    for values in ([0, 1], [False, None], [False]):
        with pytest.raises(ValueError, match=r'boolean|length'):
            _execute(market, [1, 0], values)
    with pytest.raises(ValueError, match='finite positive prices'):
        _execute(market.with_columns(pl.when(pl.int_range(pl.len()) == 1).then(None).otherwise(pl.col('close')).alias('close')), [1, 0], [False, False])
    columns = {**market.head(2).to_dict(as_series=False), 'predictions': [1, 0], 'exit': [False, False]}
    columns['price_change'] = np.asarray(columns['close']) - np.asarray(columns['open'])
    with pytest.raises(ValueError, match='default long-flat strategy'):
        backtest_snapshot(columns, exit_col='exit', strategy=lambda *args, **kwargs: long_flat_strategy(*args, **kwargs))


_RULES = {'version': 'limen-rule-exit-v1', 'signal_semantics': 'entry_until_exit',
          'conditions': [{'id': 'exit', 'type': 'threshold', 'column': 'exit_flag', 'operator': '==', 'value': True}], 'exit': 'exit'}


def _event_case(entries, exits, **options):
    inputs, policy = recorded_inputs()
    signals = inputs.signals.with_columns(pl.Series('target', entries, dtype=pl.Float64), pl.Series('exit', exits, dtype=pl.Boolean))
    return replace(inputs, signals=signals), replace(policy, prediction_mode='binary', exit_rules=dict(_RULES), **options)


def test_event_binary_entry_holds_until_exit():
    inputs, policy = _event_case([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [False] * 3 + [True] + [False] * 8)
    ledger = trade_execution(inputs, policy)
    assert ledger.fills.height == 2
    assert ledger.episodes.height == 1
    assert ledger.episodes['closed_at_ns'][0] == inputs.signals['available_at_ns'][3]
    assert ledger.episodes['exit_reason'][0] == 'rule_exit'


def test_event_pending_entry_cancellation_and_rule_exit_lag():
    inputs, policy = _event_case([1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                                [False, True, False, False, False, False, True, False, False, False, False, False],
                                execution_lag_seconds=1800)
    ledger = trade_execution(inputs, policy)
    assert ledger.intents['status'][0] == 'cancelled'
    assert ledger.fills.height == 2
    assert ledger.episodes['first_fill_ns'][0] == inputs.signals['available_at_ns'][5]
    assert ledger.episodes['closed_at_ns'][0] == inputs.signals['available_at_ns'][8]


def test_event_exit_rearm_ignores_threshold_suppression():
    inputs, policy = _event_case([1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
                                [False, False, True, False, False, True, False, False, False, False, False, False],
                                flat_threshold=0.9, signal_change_bps=9999)
    ledger = trade_execution(inputs, policy)
    assert ledger.episodes.height == 3
    assert ledger.episodes['first_fill_ns'].to_list() == [inputs.signals['available_at_ns'][i] for i in (0, 4, 8)]


def test_event_exit_cache_binds_each_ruleset_and_evidence():
    inputs, policy = _event_case([1] * 12, [False] * 3 + [True] + [False] * 8)
    changed = replace(inputs, signals=inputs.signals.with_columns(pl.Series('exit', [False] * 6 + [True] + [False] * 5)))
    changed_policy = replace(policy, exit_rules={**_RULES, 'conditions': [{**_RULES['conditions'][0], 'value': False}]})
    assert event_plan(inputs, policy) is event_plan(changed, changed_policy)
    first, second = trade_execution(inputs, policy), trade_execution(changed, changed_policy)
    assert first.contract_digest != second.contract_digest
    assert first.contract_digest == contract_digest(export_trade_contract(policy, inputs))
    assert second.contract_digest == contract_digest(export_trade_contract(changed_policy, changed))
    assert first.episodes['closed_at_ns'][0] != second.episodes['closed_at_ns'][0]
    assert first.intents['intent_id'][0].split(':')[0] != second.intents['intent_id'][0].split(':')[0]


def test_event_exit_declaration_evidence_refusals():
    inputs, policy = _event_case([1] * 12, [False] * 12)
    for bad, configured in ((replace(inputs, signals=inputs.signals.drop('exit')), policy),
                            (replace(inputs, signals=inputs.signals.with_columns(pl.lit(None, dtype=pl.Boolean).alias('exit'))), policy),
                            (inputs, replace(policy, exit_rules=None))):
        with pytest.raises(ValueError, match=r'Explicit exit'):
            trade_execution(bad, configured)
    with pytest.raises(ValueError, match='binary'):
        replace(policy, prediction_mode='target_exposure')

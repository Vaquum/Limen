from __future__ import annotations

import copy
import json
from dataclasses import replace
from decimal import Decimal

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from limen.backtest import _cost_comparison
from limen.backtest.observed_costs import CostCalibration, CostCandidate
from limen.backtest.trade_contract import ProductSpec, TradeInputs, TradePolicy, source_binding
from limen.backtest.trade_execution import trade_execution
from tests.test_observed_costs import CUTOFF, FIXTURE, _history


def _recorded_candidate(mode, hour=1):
    benchmark = pl.DataFrame(json.loads((FIXTURE / 'benchmark.json').read_text()), schema={
        'row_id': pl.String, 'phase': pl.String, 'time_ns': pl.Int64,
        'replay_available_at_ns': pl.Int64, 'price': pl.Float64,
        'reference_id': pl.String, 'observed_receipt_ns': pl.Int64})
    first = benchmark['time_ns'][0] + (hour - 1) * 3_600_000_000_000
    selected = benchmark.filter(pl.col('time_ns').is_between(first + 180_000_000_000, first + 420_000_000_000))
    observations = selected.select('row_id', pl.col('time_ns').alias('start_ns'), pl.col('time_ns').alias('end_ns'),
        pl.col('replay_available_at_ns').alias('open_available_at_ns'), pl.col('replay_available_at_ns').alias('available_at_ns'),
        *(pl.col('price').alias(name) for name in ('open', 'high', 'low', 'close')))
    # Hypothetical strategy targets/equity; every execution price is recorded.
    targets = [1.0, 0.0, 1.0, 0.0, 0.0] if mode == 'binary' else [1.0, -0.5, 0.25, 0.0, 0.0]
    signals = selected.select('row_id', pl.col('time_ns').alias('available_at_ns')).with_columns(pl.Series('target', targets))
    start, end = selected['time_ns'][0], selected['time_ns'][-1]
    binding = source_binding(observations, 'recorded_midpoints', start, end, 1, 'retrospective_assumed_exchange_label')
    inputs = TradeInputs(10_000.0, start, end, signals, observations, None, (binding,))
    policy = TradePolicy(ProductSpec('linear_perpetual', 'BTC', 'BTC', 'USDC', 0.00001, 1.0),
                         prediction_mode=mode, fee_bps=5.0, slip_bps=5.0)
    conditions = {'venue': 'hyperliquid', 'environment': 'mainnet', 'product_kind': 'linear_perpetual',
                  'instrument': 'BTC', 'base_currency': 'BTC', 'quote_currency': 'USDC',
                  'reference_id': benchmark['reference_id'][0], 'reference_convention': 'minute_midpoint_label',
                  'clock': 'assumed_exchange_label'}
    return inputs, policy, conditions, benchmark


@pytest.mark.parametrize('mode', ['binary', 'target_exposure'])
def test_frozen_event_scenarios(tmp_path, monkeypatch, mode):
    history_path, _ = _history(tmp_path)
    calibration = CostCalibration.from_history(history_path, cutoff_ns=CUTOFF)
    inputs, policy, conditions, benchmark = _recorded_candidate(mode)
    second_policy = replace(policy, slip_bps=7.0)
    first = CostCandidate(inputs, policy, 'recorded-test:first', conditions, benchmark)
    second = CostCandidate(inputs, second_policy, 'recorded-test:second', conditions, benchmark)
    calls = []

    def execute(actual_inputs, actual_policy):
        calls.append(actual_policy)
        return trade_execution(actual_inputs, actual_policy)

    monkeypatch.setattr(_cost_comparison, 'trade_execution', execute)
    comparison = calibration.compare({'first': first, 'second': second})
    assert len(calls) == 4
    report = comparison.report()
    assert list(report['candidates']) == ['first', 'second']
    assert report['ranks']['original'] is None
    assert report['ranks']['calibrated'] == {'first': 1, 'second': 1}
    assert report['ranks']['delta'] is None
    for name, original_policy in [('first', policy), ('second', second_policy)]:
        original = comparison.ledger(name, 'original')
        calibrated = comparison.ledger(name, 'calibrated')
        expected_policy = replace(original_policy, slip_bps=float(Decimal(calibration.report()['estimates']['slip_bps'])))
        for actual, expected in [(original, trade_execution(inputs, original_policy)), (calibrated, trade_execution(inputs, expected_policy))]:
            assert actual.fills.height > 0 and actual.metrics['completed_trades'] > 0
            assert actual.metrics == expected.metrics
            for field in ('states', 'intents', 'fills', 'episodes', 'funding'):
                assert_frame_equal(getattr(actual, field), getattr(expected, field))
        assert original.metrics != calibrated.metrics
        scenario = report['candidates'][name]['scenarios']['calibrated']
        assert scenario['contract']['trade_contract']['policy']['fee_bps'] == policy.fee_bps
        assert scenario['contract']['execution_cost']['application_digest'] == report['candidates'][name]['application_digest']
        assert 'time_extrapolation' in scenario['support']['flags']
        assert scenario['disposition'] == 'retrospective_low_effective_support'
    artifact = tmp_path / 'comparison.json'
    assert comparison.write(artifact) == comparison.write(artifact) == comparison.digest
    # Caller aliases are discarded; public projections cannot mutate snapshots.
    inputs.signals.replace_column(inputs.signals.get_column_index('target'), pl.Series('target', [0.0] * inputs.signals.height))
    conditions['venue'] = 'metadata-fault'
    unchanged = calibration.compare({'first': first}).report()
    assert unchanged['candidates']['first'] == report['candidates']['first']
    changed_projection = comparison.ledger('first', 'calibrated')
    changed_projection.states.replace_column(changed_projection.states.get_column_index('equity'), pl.Series('equity', [0.0] * changed_projection.states.height))
    assert comparison.ledger('first', 'calibrated').states['equity'].min() > 0
    damaged = copy.deepcopy(first._snapshot)
    damaged['research_id'] = 'fault:changed-internal'
    object.__setattr__(first, '_snapshot', damaged)
    with pytest.raises(ValueError, match='Frozen candidate'):
        calibration.compare({'first': first})

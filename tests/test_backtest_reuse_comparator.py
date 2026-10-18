import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from limen.backtest.trade_contract import contract_digest, export_trade_contract
from limen.backtest.trade_execution import trade_execution
from limen.features.price_lines import price_lines
from limen.sfd.rule_based.common_strategies import strategy_signals
from limen.targets.trade_outcome import TradeOutcomeTarget, TradeTargetContext
from tests.backtest_reuse.worker import _trade_case
from tests.benchmark_backtest_reuse import _assert_campaign_contract, _compare

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'docs/benchmarks/backtest-reuse/report.json'


@pytest.fixture(scope='module')
def recorded_outputs():
    bars = pl.read_parquet(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet',
                           n_rows=64)
    inputs, policy = _trade_case(bars, strategy_signals(bars, 'buy_and_hold'), timer=False)
    ledger = trade_execution(inputs, policy)
    context = TradeTargetContext(policy, inputs, inputs.partition_start_ns,
                                 inputs.partition_end_ns,
                                 contract_digest(export_trade_contract(policy, inputs)))
    target = TradeOutcomeTarget(bars, 'outcome', trade_context=context)
    labels = target.transform(bars, trade_context=context)
    names = tuple(ledger.metrics)
    arrays = {'metric_names': np.asarray(names),
              'metrics': np.asarray([ledger.metrics[name] for name in names]),
              'clock': ledger.states['time_ns'].to_numpy(),
              'binary_labels': labels['outcome'].drop_nulls().to_numpy(),
              'contract_digest': np.asarray(ledger.contract_digest)}
    frames = {name: getattr(ledger, name) for name in
              ('states', 'intents', 'fills', 'episodes', 'funding')}
    features = price_lines(bars, 12, 0.001, include_research_only=False)
    frames['features'] = features.select('datetime', *[
        name for name in features.columns if name not in bars.columns])
    frames['labels'] = target.outcomes.rows
    frames['primary_label'] = labels.select('datetime', 'outcome')
    return arrays, frames


def _saved_pair(directory, recorded):
    arrays, frames = recorded
    paths = tuple(directory / name for name in ('baseline', 'optimized'))
    for path in paths:
        path.mkdir()
        np.savez(path / 'arrays.npz', **arrays)
        for name, frame in frames.items():
            frame.write_parquet(path / f'{name}.parquet')
    return paths


def _change_array(path, name, change):
    with np.load(path / 'arrays.npz', allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    values = arrays[name].copy()
    index = () if values.ndim == 0 else 0
    values[index] = change(values[index])
    arrays[name] = values
    np.savez(path / 'arrays.npz', **arrays)


def _change_frame(path, name, change):
    destination = path / f'{name}.parquet'
    frame = change(pl.read_parquet(destination))
    frame.write_parquet(destination)


def _first_row(column, expression):
    return (pl.when(pl.int_range(pl.len()) == 0).then(expression)
            .otherwise(pl.col(column)).alias(column))


def _corrupt(path, corruption):
    if corruption == 'excessive_float_array':
        _change_array(path, 'metrics', lambda value: value + 1e-7)
    elif corruption == 'clock_array_1ns':
        _change_array(path, 'clock', lambda value: value + 1)
    elif corruption == 'binary_label':
        _change_array(path, 'binary_labels', lambda value: 1 - value)
    elif corruption == 'contract_digest':
        _change_array(path, 'contract_digest', lambda _value: '0' * 64)
    elif corruption == 'excessive_float_ledger':
        _change_frame(path, 'states', lambda frame: frame.with_columns(
            _first_row('cash', pl.col('cash') + 1e-7)))
    elif corruption == 'excessive_float_feature':
        _change_frame(path, 'features', lambda frame: frame.with_columns(
            _first_row('trending_score', pl.col('trending_score') + 1e-7)))
    elif corruption == 'clock_ledger_1ns':
        _change_frame(path, 'states', lambda frame: frame.with_columns(
            _first_row('time_ns', pl.col('time_ns') + 1)))
    elif corruption == 'row_drop':
        _change_frame(path, 'states', lambda frame: frame.head(frame.height - 1))
    elif corruption == 'row_order':
        _change_frame(path, 'features', lambda frame: frame.reverse())
    elif corruption == 'null_mask':
        _change_frame(path, 'states', lambda frame: frame.with_columns(
            _first_row('cash', pl.lit(None))))
    elif corruption == 'nan_mask':
        _change_frame(path, 'states', lambda frame: frame.with_columns(
            _first_row('cash', pl.lit(float('nan')))))
    elif corruption == 'label_null_mask':
        _change_frame(path, 'primary_label', lambda frame: frame.with_columns(
            _first_row('outcome', pl.lit(None))))
    elif corruption == 'label_availability':
        _change_frame(path, 'labels', lambda frame: frame.with_columns(
            _first_row('long_available', ~pl.col('long_available'))))
    elif corruption == 'column_order':
        _change_frame(path, 'states', lambda frame: frame.select(frame.columns[::-1]))
    elif corruption == 'dtype':
        _change_frame(path, 'states', lambda frame: frame.with_columns(
            pl.col('time_ns').cast(pl.UInt64)))
    else:
        raise ValueError(corruption)


def test_comparator_accepts_complete_recorded_outputs(tmp_path, recorded_outputs):
    baseline, optimized = _saved_pair(tmp_path, recorded_outputs)
    result = _compare(baseline, optimized)
    assert result['max_absolute_error'] == 0
    assert result['floating_values_compared'] > 0
    assert len(result['tables_compared']) == len(recorded_outputs[1])


@pytest.mark.parametrize('corruption', (
    'excessive_float_array', 'excessive_float_ledger', 'excessive_float_feature',
    'clock_array_1ns', 'clock_ledger_1ns', 'row_drop', 'row_order', 'binary_label',
    'contract_digest', 'null_mask', 'nan_mask', 'label_null_mask',
    'label_availability', 'column_order', 'dtype',
))
def test_comparator_rejects_recorded_output_corruption(
        tmp_path, recorded_outputs, corruption):
    baseline, optimized = _saved_pair(tmp_path, recorded_outputs)
    _corrupt(optimized, corruption)
    with pytest.raises(AssertionError):
        _compare(baseline, optimized)


def test_recorded_benchmark_report_meets_acceptance_contract():
    report = json.loads(REPORT.read_text())
    _assert_campaign_contract(report)


@pytest.mark.parametrize('corruption', ('equivalence', 'error', 'equal_speed',
                                       'slower', 'missing_workload'))
def test_benchmark_contract_rejects_recorded_acceptance_regressions(corruption):
    report = json.loads(REPORT.read_text())
    if corruption == 'equivalence':
        report['scenarios'][0]['equivalent'] = False
    elif corruption == 'error':
        report['scenarios'][0]['max_absolute_error'] = 1e-7
    elif corruption == 'equal_speed':
        report['scenarios'][0]['warm_speedup'] = 1.0
    elif corruption == 'slower':
        report['scenarios'][0]['warm_speedup'] = 0.5
    else:
        report['scenarios'].pop()
    with pytest.raises(AssertionError):
        _assert_campaign_contract(report)

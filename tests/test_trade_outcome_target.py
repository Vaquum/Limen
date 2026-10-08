import math
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from limen.backtest.trade_contract import contract_digest, export_trade_contract, source_binding
from limen.targets import TradeOutcomeTarget, TradeTargetContext
from tests.test_target_exposure import FIXTURE, _case


def target_case(count=8, **params):
    inputs, policy = _case([1.0] * count, max_holding_seconds=900, **params)
    frame = pl.read_parquet(FIXTURE).head(count)
    inputs = replace(inputs, signals=inputs.signals.with_columns(frame['datetime']))
    context = TradeTargetContext(policy, inputs, inputs.partition_start_ns, inputs.partition_end_ns, contract_digest(export_trade_contract(policy, inputs)))
    return frame, context


def test_forward_trade_matches_shared_net_ledger():
    frame, context = target_case(fee_bps=10, slip_bps=5)
    target = TradeOutcomeTarget(frame, 'outcome', trade_context=context, output='net_return')
    target.transform(frame, trade_context=context)
    prices = context.inputs.observations
    for side, sign in [('long', 1), ('short', -1)]:
        row = target.outcomes.rows.row(0, named=True)
        assert row[f'{side}_available']
        first = prices['close'][0]
        last = prices.filter(pl.col('available_at_ns') == row[f'{side}_exit_ns'])['close'][0]
        entry, exit_price = first * (1 + sign * .0005), last * (1 - sign * .0005)
        quantity = math.floor(math.nextafter(10000 / (first * (1 + .0005 + (1 + sign * .0005) * .001)) / 1e-12, math.inf)) * 1e-12
        expected = (sign * quantity * (exit_price - entry) - quantity * (entry + exit_price) * .001) / 10000
        assert row[f'{side}_return'] == pytest.approx(expected)
        assert row[f'{side}_exit_reason'] == 'time_stop'


def test_binary_and_return_outputs():
    frame, context = target_case()
    for side in ('long', 'short'):
        binary = TradeOutcomeTarget(frame, 'outcome', trade_context=context, side=side)
        continuous = TradeOutcomeTarget(frame, 'outcome', trade_context=context, side=side, output='net_return')
        classified = binary.transform(frame, trade_context=context)['outcome']
        returns = continuous.transform(frame, trade_context=context)['outcome']
        assert classified.equals((returns > 0).cast(pl.Int8))
        assert returns[-1] is None
    with pytest.raises(ValueError, match='Unknown'):
        binary.transform(frame, trade_context=context, ignored=True)


def test_split_tail_and_future_isolation():
    frame, context = target_case(count=20)
    target = TradeOutcomeTarget(frame, 'outcome', trade_context=context)
    original = target.transform(frame, trade_context=context)
    assert original['outcome'][-1] is None
    assert target.outcomes.rows['long_available'].sum() > 0
    # Only real recorded observations after the true partition are added.
    future, _ = _case([1.0] * 25)
    observations = future.observations
    binding = source_binding(observations, str(FIXTURE), context.partition_start_ns, context.partition_end_ns, 1, 'recorded out-of-partition observations')
    extended = replace(context, inputs=replace(context.inputs, observations=observations, sources=(*context.inputs.sources, binding)))
    assert target.transform(frame, trade_context=extended).equals(original)
    with pytest.raises(ValueError, match='bounds'):
        target.transform(frame, trade_context=replace(context, partition_end_ns=context.partition_end_ns + 1))


def test_auxiliary_targets_do_not_enter_features():
    from tests.test_direction_sizing import native_manifest, recorded_source

    source = recorded_source()
    manifest = native_manifest()
    data = manifest.prepare_data(source, {})
    assert data['x_test'].height == data['_trade_context'].partitions[2].signals.height
    assert data['x_test'].height > data['_trade_labels'].rows.filter(pl.col('row_id').str.starts_with('partition:2:') & pl.col('long_available') & pl.col('short_available')).height
    assert not any(name.startswith(('long_', 'short_')) or name.endswith('_ns') or name == 'outcome' for name in data['x_train'].columns)
    sensor_data, _ = manifest.sensor_input_prep(source, data['_fitted_params'], {})
    assert not any(name.startswith(('long_', 'short_')) or name == 'outcome' for name in sensor_data.columns)
    assert '__trade_available_at_ns__' in sensor_data.columns


def test_zero_is_unprofitable_and_missing_context_fails():
    from limen.experiment._prepare_trade_context import normalize_observations
    from limen.backtest.trade_contract import ProductSpec, TradeInputs, TradePolicy

    recorded = pl.read_parquet(Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet')
    price = recorded.group_by('open').len().sort('len', descending=True)['open'][0]
    selected = recorded.filter(pl.col('open') == price).head(2)
    times = selected['datetime'].dt.epoch('ns')
    observations = normalize_observations(selected.with_columns(start_ns=times, end_ns=times, open_available_at_ns=times, available_at_ns=times, high=pl.col('open'), low=pl.col('open'), close=pl.col('open')))
    # A point projection uses only the recorded open, with its recorded time.
    signals = pl.DataFrame({'row_id':['a','b'], 'available_at_ns':times, 'target':[None,None], 'datetime':selected['datetime']})
    binding = source_binding(observations, 'recorded equal open points', int(times[0]), int(times[-1]), 1, 'recorded open projection')
    inputs = TradeInputs(10000, int(times[0]), int(times[-1]), signals, observations, None, (binding,))
    policy = TradePolicy(ProductSpec('linear_perpetual','BTCUSDT','BTC','USDT',1e-12,0), max_holding_seconds=1)
    context = TradeTargetContext(policy, inputs, inputs.partition_start_ns, inputs.partition_end_ns, contract_digest(export_trade_contract(policy, inputs)))
    result = TradeOutcomeTarget(selected, 'outcome', trade_context=context).transform(selected, trade_context=context)
    assert result['outcome'][0] == 0 and result['outcome'][1] is None
    manifest = native_manifest_without_economics()
    with pytest.raises(ValueError, match='execution economics'):
        manifest.prepare_data(recorded.head(40), {})


def native_manifest_without_economics():
    from limen.experiment.manifest_core import MLManifest

    manifest = MLManifest().set_split_config(6,2,2)
    return manifest.with_target_label('outcome', TradeOutcomeTarget)

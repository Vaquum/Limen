import math
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from limen.backtest.trade_contract import ProductSpec, TradeInputs, TradePolicy, source_binding
from limen.backtest.trade_execution import trade_execution

FIXTURE = Path(__file__).parent / 'fixtures/dollar_bar_crash_reversal_15m.parquet'


def _case(targets, *, spot=False, **options):
    # Strategy targets are test configuration; every timestamp/price is taken
    # unchanged from the attributed repository market fixture.
    market = pl.read_parquet(FIXTURE).head(len(targets))
    times = market['datetime'].dt.epoch('ns')
    points = pl.DataFrame({'row_id': [f'price:{i}' for i in range(len(targets))],
                          'start_ns': times, 'end_ns': times, 'open_available_at_ns': times,
                          'available_at_ns': times, **{col: market['close'] for col in ('open', 'high', 'low', 'close')}})
    signals = pl.DataFrame({'row_id': [f'signal:{i}' for i in range(len(targets))],
                           'available_at_ns': times, 'target': targets})
    start, end = int(times[0]), int(times[-1])
    binding = source_binding(points, str(FIXTURE), start, end, 900000000000, 'recorded close points')
    policy = TradePolicy(ProductSpec('cash_spot' if spot else 'linear_perpetual', 'BTCUSDT', 'BTC', 'USDT', 1e-12, 0.0), **options)
    return TradeInputs(10000.0, start, end, signals, points, None, (binding,)), policy


@pytest.mark.parametrize('side', (1, -1))
def test_signed_cash_and_fill_costs(side):
    inputs, policy = _case([float(side), 0.0], fee_bps=10.0, slip_bps=5.0)
    result = trade_execution(inputs, policy)
    first, last = inputs.observations['close'].to_list()
    f, s = 0.001, 0.0005
    quantity = math.floor(math.nextafter(10000 / (first * (1 + s + (1 + side * s) * f)) / 1e-12, math.inf)) * 1e-12
    entry, exit_price = first * (1 + side * s), last * (1 - side * s)
    expected = 10000 + side * quantity * (exit_price - entry) - quantity * (entry + exit_price) * f
    assert result.fills['quantity_delta'][0] == pytest.approx(side * quantity)
    assert result.metrics['ending_equity'] == pytest.approx(expected)
    assert result.metrics['fees'] == pytest.approx(quantity * (entry + exit_price) * f)
    assert result.metrics['slippage'] == pytest.approx(quantity * (first + last) * s)
    assert result.metrics['completed_trades'] == 1
    assert result.states['quantity'][-1] == 0


def test_full_allocation_reserves_cash():
    inputs, policy = _case([1.0, 1.0], spot=True, fee_bps=10.0, slip_bps=5.0)
    result = trade_execution(inputs, policy)
    price = inputs.observations['close'][0]
    expected = 10000 / (price * 1.0005 * 1.001)
    assert result.fills.height == 1
    assert result.fills['quantity_delta'][0] == pytest.approx(expected)
    assert result.states['cash'][0] >= -1e-9
    assert abs(result.states['cash'][0]) < 1e-6
    assert result.metrics['open_trades'] == 1


def test_signal_change_resize_preserves_episode_and_anchors():
    inputs, policy = _case([0.5, 0.5, 0.75, 0.0], fee_bps=10.0, slip_bps=5.0)
    result = trade_execution(inputs, policy)
    p0, _, p2, _ = inputs.observations['close'].to_list()
    cost0 = p0 * (0.0005 + 1.0005 * 0.001)
    q0 = math.floor(math.nextafter(0.5 * 10000 / (p0 + 0.5 * cost0) / 1e-12, math.inf)) * 1e-12
    equity2 = 10000 - q0 * cost0 + q0 * (p2 - p0)
    cost2 = p2 * (0.0005 + 1.0005 * 0.001)
    q2 = math.floor(math.nextafter(0.75 * (equity2 + cost2 * q0) / (p2 + 0.75 * cost2) / 1e-12, math.inf)) * 1e-12
    assert result.states['quantity'][0] == result.states['quantity'][1]
    assert result.states['quantity'][2] == pytest.approx(q2)
    assert result.fills.height == 3
    assert result.fills['episode_id'].n_unique() == 1
    assert result.episodes['first_fill_ns'][0] == inputs.partition_start_ns
    assert result.episodes['anchor_price'][0] == pytest.approx(p0 * 1.0005)


def test_reversal_closes_before_opposite_episode():
    inputs, policy = _case([0.5, -0.5, 0.0])
    result = trade_execution(inputs, policy)
    assert result.fills.height == 4
    assert result.episodes['side'].to_list() == [1, -1]
    assert result.episodes['closed_at_ns'][0] == result.episodes['first_fill_ns'][1]
    assert result.fills['episode_id'][1] != result.fills['episode_id'][2]


def test_disabled_path_and_mode_validation():
    inputs, policy = _case([0.25, 0.0])
    with pytest.raises(ValueError, match='mode/bounds'):
        trade_execution(inputs, replace(policy, prediction_mode='binary'))
    with pytest.raises(ValueError, match='borrowed shorts'):
        trade_execution(replace(inputs, signals=inputs.signals.with_columns(pl.lit(-0.5).alias('target'))),
                        replace(policy, product=replace(policy.product, kind='cash_spot')))
    with pytest.raises(ValueError, match='finite'):
        trade_execution(replace(inputs, signals=inputs.signals.with_columns(pl.lit(float('nan')).alias('target'))), policy)
    with pytest.raises(ValueError, match='maximum exposure'):
        trade_execution(inputs, replace(policy, max_exposure=0.1))

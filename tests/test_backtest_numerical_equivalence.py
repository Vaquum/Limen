from pathlib import Path

import numpy as np
import polars as pl
import pytest

import limen.backtest._snapshot_ledger as ledger_module
from limen.backtest._snapshot_execution import snapshot_with_execution
from limen.backtest._snapshot_ledger import snapshot_ledger
from limen.backtest.backtest_snapshot import BACKTEST_SNAPSHOT_COLUMNS
from limen.features.dollar_bar_crash_reversal import dollar_bar_crash_reversal
from limen.sfd.rule_based.common_strategies import STRATEGY_NAMES, strategy_signals
from tests.backtest_reuse.snapshot_reference import snapshot_ledger as reference_ledger

FIXTURES = Path(__file__).parent / 'fixtures'


@pytest.fixture(scope='module')
def recorded():
    return pl.read_parquet(FIXTURES / 'spot_1h_20240101_20241231.parquet')


def _columns(bars, predictions):
    columns = {name: bars[name].to_numpy() for name in ('open', 'high', 'low', 'close')}
    columns['price_change'] = columns['close'] - columns['open']
    columns['predictions'] = predictions
    return columns


def _assert_metrics(actual, expected):
    assert list(actual) == list(expected) == BACKTEST_SNAPSHOT_COLUMNS
    np.testing.assert_allclose(list(actual.values()), list(expected.values()),
                               atol=1e-12, rtol=0, equal_nan=True)


@pytest.mark.parametrize('strategy', STRATEGY_NAMES)
@pytest.mark.parametrize('notional_rate', (0.1, 0.5, 1.0))
def test_all_snapshot_metrics_match_prechange_on_recorded_strategies(
        recorded, strategy, notional_rate):
    columns = _columns(recorded, strategy_signals(recorded, strategy))
    metrics, result = snapshot_with_execution(columns, notional_rate=notional_rate,
                                              fee_bps=10.0, slip_bps=3.0)
    _assert_metrics(metrics, reference_ledger(result, notional_rate))


@pytest.mark.parametrize('length', (1, 19, 20, 21, 39, 40, 41))
def test_short_recorded_windows_preserve_flat_and_cvar_boundaries(recorded, length):
    bars = recorded.head(length)
    columns = _columns(bars, strategy_signals(bars, 'sma_cross'))
    metrics, result = snapshot_with_execution(columns)
    assert not result.pos.any()
    _assert_metrics(metrics, reference_ledger(result, 1.0))
    assert np.isnan(metrics['avg_win_bps']) and np.isnan(metrics['avg_loss_bps'])
    assert np.isnan(metrics['cvar_95_pnl_bps']) == (length < 20)


@pytest.mark.parametrize('options', (
    {'take_profit_bps': 150.49}, {'stop_loss_bps': 30.0},
    {'take_profit_bps': 10.0, 'stop_loss_bps': 10.0},
    {'stop_loss_bps': 0.8}, {'take_profit_bps': 1e9},
))
def test_barrier_snapshot_matches_prechange_on_recorded_crash(options):
    bars = pl.read_parquet(FIXTURES / 'dollar_bar_crash_reversal_15m.parquet')
    featured = dollar_bar_crash_reversal(
        bars, momentum_threshold_bps=-525.0, flow_z_threshold=-0.5, hold_minutes=30)
    window = featured.slice(2897, 583)
    columns = _columns(window, window['dollar_bar_crash_reversal_position'].to_numpy())
    metrics, result = snapshot_with_execution(columns, notional_rate=0.5,
                                              fee_bps=10.0, slip_bps=5.0, **options)
    assert result.pos.any()
    _assert_metrics(metrics, reference_ledger(result, 0.5))


def test_partitioned_tail_preserves_full_sort_summation_order(recorded):
    columns = _columns(recorded, strategy_signals(recorded, 'buy_and_hold'))
    _, result = snapshot_with_execution(columns)
    for length in (20, 21, 99, 100, 101, len(result.pos)):
        truncated = type(result)(*(values[:length] for values in result))
        actual = snapshot_ledger(truncated, 1.0)
        expected = reference_ledger(truncated, 1.0)
        _assert_metrics(actual, expected)
        assert actual['cvar_95_pnl_bps'] == expected['cvar_95_pnl_bps']


def test_snapshot_groups_quantiles_and_sorts_only_cvar_tail(recorded, monkeypatch):
    columns = _columns(recorded, strategy_signals(recorded, 'sma_cross'))
    _, result = snapshot_with_execution(columns)
    quantile_calls, partition_calls, sort_sizes = [], [], []

    class ObservedNumpy:
        def __getattr__(self, name):
            return getattr(np, name)

        def quantile(self, values, quantiles):
            quantile_calls.append(tuple(quantiles))
            return np.quantile(values, quantiles)

        def partition(self, values, kth):
            partition_calls.append((values.size, kth))
            return np.partition(values, kth)

        def sort(self, values):
            sort_sizes.append(values.size)
            return np.sort(values)

    with monkeypatch.context() as observed:
        observed.setattr(ledger_module, 'np', ObservedNumpy())
        actual = snapshot_ledger(result, 1.0)
    tail_count = int(np.floor(0.05 * result.net.size))
    assert quantile_calls == [(0.05, 0.50, 0.95)] * 4
    assert partition_calls == [(result.net.size, tail_count - 1)]
    assert sort_sizes == [tail_count]
    _assert_metrics(actual, reference_ledger(result, 1.0))

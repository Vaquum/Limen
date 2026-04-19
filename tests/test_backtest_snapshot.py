import math

import pandas as pd
import pytest

from limen.backtest.backtest_snapshot import backtest_snapshot


def _make_snapshot_input() -> pd.DataFrame:

    return pd.DataFrame({
        'predictions': [1, 1, 0],
        'open': [100.0, 120.0, 108.0],
        'close': [120.0, 108.0, 108.0],
        'price_change': [20.0, -12.0, 0.0],
    })


def test_snapshot_defaults_to_next_bar_execution() -> None:

    data = pd.DataFrame({
        'predictions': [1, 0],
        'open': [100.0, 200.0],
        'close': [150.0, 210.0],
        'price_change': [50.0, 10.0],
    })

    result = backtest_snapshot(data, fee_bps=0.0, slip_bps=0.0).iloc[0]

    assert result['execution_lag_bars'] == 1
    assert result['bars_total'] == 1
    assert result['bars_in_market_count'] == 1
    assert result['bars_in_market_pct'] == 100.0
    assert result['trade_runs_count'] == 1
    assert result['total_return_net_pct'] == 5.0
    assert math.isnan(result['sharpe_per_bar'])


def test_snapshot_can_reproduce_legacy_same_row_execution() -> None:

    data = pd.DataFrame({
        'predictions': [1, 0],
        'open': [100.0, 200.0],
        'close': [150.0, 210.0],
        'price_change': [50.0, 10.0],
    })

    result = backtest_snapshot(
        data,
        fee_bps=0.0,
        slip_bps=0.0,
        execution_lag_bars=0,
    ).iloc[0]

    assert result['execution_lag_bars'] == 0
    assert result['total_return_net_pct'] == 50.0


def test_snapshot_trade_metrics_are_run_level_by_default() -> None:

    result = backtest_snapshot(
        _make_snapshot_input(),
        fee_bps=0.0,
        slip_bps=0.0,
        execution_lag_bars=0,
    ).iloc[0]

    assert result['trades_count'] == 1
    assert result['trade_runs_count'] == 1
    assert result['bars_in_market_count'] == 2
    assert result['trade_win_rate_pct'] == 100.0
    assert result['trade_expectancy_pct'] == 8.0
    assert result['trade_return_mean_win_pct'] == 8.0
    assert math.isnan(result['trade_return_mean_loss_pct'])
    assert result['bar_win_rate_pct'] == 50.0
    assert result['bar_expectancy_pct'] == 5.0
    assert result['bar_return_mean_win_pct'] == 20.0
    assert result['bar_return_mean_loss_pct'] == -10.0


def test_snapshot_bar_mode_preserves_legacy_trade_metrics() -> None:

    result = backtest_snapshot(
        _make_snapshot_input(),
        fee_bps=0.0,
        slip_bps=0.0,
        execution_lag_bars=0,
        trades_count_mode='bars',
    ).iloc[0]

    assert result['trades_count'] == 2
    assert result['trade_runs_count'] == 1
    assert result['trade_win_rate_pct'] == 50.0
    assert result['trade_expectancy_pct'] == 5.0
    assert result['trade_return_mean_win_pct'] == 20.0
    assert result['trade_return_mean_loss_pct'] == -10.0

def test_snapshot_mean_kelly_pct_uses_trade_runs_by_default() -> None:

    data = pd.DataFrame({
        'predictions': [1, 0, 1, 0],
        'open': [100.0, 100.0, 100.0, 100.0],
        'close': [120.0, 100.0, 90.0, 100.0],
        'price_change': [20.0, 0.0, -10.0, 0.0],
    })

    result = backtest_snapshot(
        data,
        fee_bps=0.0,
        slip_bps=0.0,
        execution_lag_bars=0,
    ).iloc[0]

    assert result['mean_kelly_pct'] == 25.0


def test_snapshot_mean_kelly_pct_keeps_breakevens_in_denominator() -> None:

    data = pd.DataFrame({
        'predictions': [1, 0, 1, 0, 1],
        'open': [100.0, 100.0, 100.0, 100.0, 100.0],
        'close': [120.0, 100.0, 90.0, 100.0, 100.0],
        'price_change': [20.0, 0.0, -10.0, 0.0, 0.0],
    })

    result = backtest_snapshot(
        data,
        fee_bps=0.0,
        slip_bps=0.0,
        execution_lag_bars=0,
        trades_count_mode='bars',
    ).iloc[0]

    assert result['mean_kelly_pct'] == pytest.approx(16.667, abs=1e-3)


def test_snapshot_handles_empty_input() -> None:

    data = pd.DataFrame({
        'predictions': [],
        'open': [],
        'close': [],
        'price_change': [],
    })

    result = backtest_snapshot(data).iloc[0]

    assert result['bars_total'] == 0
    assert result['bars_in_market_count'] == 0
    assert result['trade_runs_count'] == 0
    assert math.isnan(result['total_return_net_pct'])
    assert math.isnan(result['max_drawdown_pct'])
    assert math.isnan(result['sharpe_per_bar'])


def test_snapshot_validates_mode_and_execution_lag() -> None:

    with pytest.raises(ValueError, match='trades_count_mode'):
        backtest_snapshot(_make_snapshot_input(), trades_count_mode='invalid')

    with pytest.raises(ValueError, match='execution_lag_bars'):
        backtest_snapshot(_make_snapshot_input(), execution_lag_bars=-1)

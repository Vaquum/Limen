import numpy as np

from limen.metrics.rule_based_metrics import rule_based_metrics


def _make_positions(train, val, test):
    return {
        'train': np.array(train),
        'val': np.array(val),
        'test': np.array(test),
    }


def _make_backtest(drawdown_train, drawdown_val, drawdown_test):
    return {
        'train': {'drawdown_bps_p5': drawdown_train},
        'val':   {'drawdown_bps_p5': drawdown_val},
        'test':  {'drawdown_bps_p5': drawdown_test},
    }


def test_num_trades_counts_entries_not_bars() -> None:
    positions = _make_positions([0, 1, 1, 0, 1], [1, 1, 0, 1, 0], [0, 0, 1, 0, 0])
    result = rule_based_metrics(positions, {})
    assert result['num_trades_train'] == 2
    assert result['num_trades_val'] == 2
    assert result['num_trades_test'] == 1


def test_position_rate() -> None:
    positions = _make_positions([1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0])
    result = rule_based_metrics(positions, {})
    assert result['position_rate_train'] == 0.5
    assert result['position_rate_val'] == 0.25
    assert result['position_rate_test'] == 0.0


def test_backtest_metrics_flattened_with_split_suffix() -> None:
    positions = _make_positions([1], [1], [1])
    bt = _make_backtest(-5.0, -6.0, -8.0)
    result = rule_based_metrics(positions, bt)
    assert result['drawdown_bps_p5_test'] == -8.0


def test_drawdown_std_bps() -> None:
    positions = _make_positions([1], [1], [1])
    bt = _make_backtest(-5.0, -5.0, -5.0)
    result = rule_based_metrics(positions, bt)
    assert result['drawdown_std_bps'] == 0.0


def test_is_stable_false_without_legacy_sharpe_surface() -> None:
    positions = _make_positions([1], [1], [1])
    bt = _make_backtest(-5.0, -5.0, -5.0)
    result = rule_based_metrics(positions, bt)
    assert result['is_stable'] is False


def test_missing_backtest_results_degrade_gracefully() -> None:
    positions = _make_positions([1], [1], [1])
    result = rule_based_metrics(positions, {})
    assert result['drawdown_std_bps'] is None
    assert result['is_stable'] is False

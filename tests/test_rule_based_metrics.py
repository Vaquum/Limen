import numpy as np

from limen.metrics.rule_based_metrics import rule_based_metrics


def _make_positions(train, val, test):
    return {
        'train': np.array(train),
        'val': np.array(val),
        'test': np.array(test),
    }


def _make_backtest(sharpe_train, sharpe_val, sharpe_test,
                   drawdown_train, drawdown_val, drawdown_test):
    return {
        'train': {'sharpe_per_bar': sharpe_train, 'max_drawdown_bps': drawdown_train},
        'val':   {'sharpe_per_bar': sharpe_val,   'max_drawdown_bps': drawdown_val},
        'test':  {'sharpe_per_bar': sharpe_test,  'max_drawdown_bps': drawdown_test},
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
    bt = _make_backtest(1.0, 0.8, 0.6, -5.0, -6.0, -8.0)
    result = rule_based_metrics(positions, bt)
    assert 'sharpe_per_bar_train' in result
    assert 'sharpe_per_bar_val' in result
    assert 'sharpe_per_bar_test' in result
    assert result['sharpe_per_bar_train'] == 1.0
    assert result['max_drawdown_bps_test'] == -8.0


def test_sharpe_std_and_drawdown_std() -> None:
    positions = _make_positions([1], [1], [1])
    bt = _make_backtest(1.0, 1.0, 1.0, -5.0, -5.0, -5.0)
    result = rule_based_metrics(positions, bt)
    assert result['sharpe_std'] == 0.0
    assert result['drawdown_std'] == 0.0


def test_sharpe_degradation() -> None:
    positions = _make_positions([1], [1], [1])
    bt = _make_backtest(1.0, 0.8, 0.5, -5.0, -5.0, -5.0)
    result = rule_based_metrics(positions, bt)
    assert result['sharpe_degradation'] == 0.5


def test_is_stable_true_when_within_thresholds() -> None:
    positions = _make_positions([1], [1], [1])
    bt = _make_backtest(1.0, 1.0, 1.0, -5.0, -5.0, -5.0)
    result = rule_based_metrics(positions, bt, sharpe_std_threshold=0.5, sharpe_degradation_threshold=0.3)
    assert result['is_stable'] is True


def test_is_stable_false_when_sharpe_std_exceeds_threshold() -> None:
    positions = _make_positions([1], [1], [1])
    bt = _make_backtest(2.0, 0.0, -1.0, -5.0, -5.0, -5.0)
    result = rule_based_metrics(positions, bt, sharpe_std_threshold=0.5, sharpe_degradation_threshold=0.3)
    assert result['is_stable'] is False


def test_missing_backtest_results_degrade_gracefully() -> None:
    positions = _make_positions([1], [1], [1])
    result = rule_based_metrics(positions, {})
    assert result['sharpe_std'] is None
    assert result['drawdown_std'] is None
    assert result['sharpe_degradation'] is None
    assert result['is_stable'] is False

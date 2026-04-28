import numpy as np
import pandas as pd
import polars as pl

from limen.sfd.reference_architecture import LogRegBinary
from limen.sfd.reference_architecture import RandomBinary
from limen.sfd.reference_architecture import RuleBasedStrategy
from limen.sfd.reference_architecture import TabPFNBinary
from limen.sfd.reference_architecture import XGBoostRegressor
from limen.sfd.reference_architecture.rule_based import rule_based


def _make_data(n=200, binary=False, with_val=True, with_price=True):

    np.random.seed(42)
    n_features = 5
    x = np.random.randn(n, n_features)

    if binary:
        y = (x[:, 0] > 0).astype(int)
    else:
        y = x[:, 0] * 0.5 + np.random.randn(n) * 0.1

    split_train = int(n * 0.6)
    split_val = int(n * 0.8)

    data = {
        'x_train': x[:split_train],
        'y_train': y[:split_train],
        'x_test': x[split_val:],
        'y_test': y[split_val:],
    }

    if with_val:
        data['x_val'] = x[split_train:split_val]
        data['y_val'] = y[split_train:split_val]

    if with_price:
        n_test = n - split_val
        data['price_data_for_backtest'] = pl.DataFrame({
            'open': np.random.uniform(100, 200, n_test),
            'high': np.random.uniform(200, 300, n_test),
            'low': np.random.uniform(50, 100, n_test),
            'close': np.random.uniform(100, 200, n_test),
            'datetime': pd.date_range('2025-01-01', periods=n_test, freq='h'),
        })

    return data


def test_xgboost_train_returns_fitted_model():

    data = _make_data()
    model = XGBoostRegressor()
    result = model.train(data, learning_rate=0.1, n_estimators=10, random_state=42)

    assert result is model
    assert model.model is not None


def test_xgboost_evaluate_returns_all_metric_types():

    data = _make_data(with_price=True)
    model = XGBoostRegressor().train(data, learning_rate=0.1, n_estimators=10, random_state=42)
    results = model.evaluate(data)

    for key in ['bias', 'mae', 'rmse', 'r2', 'mape']:
        assert key in results, f"Missing results key: {key}"

    for key in ['confusion_tp', 'confusion_fp', 'confusion_tn', 'confusion_fn',
                'confusion_precision', 'confusion_recall',
                'confusion_tp_mean_return_pct', 'confusion_fp_mean_return_pct',
                'confusion_tn_mean_return_pct', 'confusion_fn_mean_return_pct']:
        assert key in results, f"Missing confusion key: {key}"

    for key in ['backtest_trade_win_rate_pct', 'backtest_max_drawdown_pct',
                'backtest_total_return_net_pct', 'backtest_sharpe_per_bar',
                'backtest_mean_kelly_pct']:
        assert key in results, f"Missing backtest key: {key}"

    assert '_preds' in results


def test_reference_backtest_uses_snapshot_cost_config():
    data = {
        'price_data_for_backtest': pl.DataFrame({
            'open': [100.0, 100.0],
            'high': [100.0, 100.0],
            'low': [100.0, 100.0],
            'close': [100.0, 100.0],
            'datetime': pd.date_range('2025-01-01', periods=2, freq='h'),
        }),
        '_snapshot_fee_bps': 50.0,
        '_snapshot_slip_bps': 50.0,
    }

    result = LogRegBinary()._compute_backtest(np.array([1, 0]), data)

    assert result['backtest_cost_round_trip_bps'] == 198
    assert result['backtest_trade_expectancy_pct'] == -1.983
    assert result['backtest_total_return_net_pct'] == -2.0


def test_logreg_train_evaluate_end_to_end():

    data = _make_data(binary=True, with_price=True)
    model = LogRegBinary().train(data, solver='lbfgs', max_iter=200)
    results = model.evaluate(data)

    for key in ['recall', 'precision', 'fpr', 'auc', 'accuracy']:
        assert key in results, f"Missing results key: {key}"

    for key in ['confusion_tp', 'confusion_fp', 'confusion_tn', 'confusion_fn']:
        assert key in results, f"Missing confusion key: {key}"

    assert results['confusion_tp'] + results['confusion_fn'] == int((data['y_test'] == 1).sum())
    assert results['confusion_fp'] + results['confusion_tn'] == int((data['y_test'] == 0).sum())

    assert '_preds' in results


def test_random_binary_train_evaluate_end_to_end():

    data = _make_data(binary=True, with_price=True)
    model = RandomBinary().train(data, random_weights=0.5)
    results = model.evaluate(data)

    for key in ['recall', 'precision', 'fpr', 'auc', 'accuracy']:
        assert key in results, f"Missing results key: {key}"

    for key in ['confusion_tp', 'confusion_fp', 'confusion_tn', 'confusion_fn']:
        assert key in results, f"Missing confusion key: {key}"

    assert '_preds' in results


def test_tabpfn_train_evaluate_end_to_end():

    if TabPFNBinary is None:
        return

    data = _make_data(binary=True, with_price=True)
    model = TabPFNBinary().train(data, n_ensemble_configurations=2, device='cpu')
    results = model.evaluate(data)

    for key in ['recall', 'precision', 'fpr', 'auc', 'accuracy']:
        assert key in results, f"Missing results key: {key}"

    for key in ['confusion_tp', 'confusion_fp', 'confusion_tn', 'confusion_fn']:
        assert key in results, f"Missing confusion key: {key}"

    assert 'optimal_threshold' in results
    assert 'val_score' in results
    assert '_preds' in results


def test_train_with_validation_data():

    data = _make_data(with_val=True)
    model = XGBoostRegressor().train(data, learning_rate=0.1, n_estimators=10, random_state=42)
    assert model.model is not None

    results = model.evaluate(data)
    assert 'rmse' in results


def test_train_without_validation_data():

    data = _make_data(with_val=False)
    model = XGBoostRegressor().train(data, learning_rate=0.1, n_estimators=10, random_state=42)
    assert model.model is not None

    results = model.evaluate(data)
    assert 'rmse' in results


_CONDITIONS_RB = [
    {'id': 'macd_cross', 'type': 'threshold', 'column': 'macd_cross', 'operator': '>', 'value': 0},
    {'id': 'above_ema',  'type': 'threshold', 'column': 'above_ema',  'operator': '>', 'value': 0},
    {'id': 'entry', 'operator': 'and', 'operands': ['macd_cross', 'above_ema']},
]


def _make_rule_based_data(n: int = 20) -> dict:
    pattern = [True, True, False, False] * (n // 4)
    df = pl.DataFrame({
        'macd_cross': pattern,
        'above_ema': ([True, True, True, False] * (n // 4)),
        'open':  [100.0 + i for i in range(n)],
        'close': [101.0 + i for i in range(n)],
    })
    split = n // 5
    return {
        'train': df[:split * 3],
        'val':   df[split * 3: split * 4],
        'test':  df[split * 4:],
        'strategy': {'conditions': _CONDITIONS_RB, 'entry': 'entry'},
        '_alignment': {},
    }


def test_rule_based_train_is_noop():
    model = RuleBasedStrategy()
    data = _make_rule_based_data()
    result = model.train(data, some_param=1)
    assert result is model
    assert model.model is None


def test_rule_based_evaluate_returns_expected_metrics():
    assert RuleBasedStrategy.deterministic is True
    data = _make_rule_based_data()
    results = RuleBasedStrategy().train(data).evaluate(data)

    for split in ('train', 'val', 'test'):
        assert isinstance(results[f'num_trades_{split}'], int)
        rate = results[f'position_rate_{split}']
        assert 0.0 <= rate <= 1.0

    assert isinstance(results['sharpe_std'], float)
    assert isinstance(results['drawdown_std'], float)
    assert isinstance(results['sharpe_degradation'], float)
    assert isinstance(results['is_stable'], bool)

    assert isinstance(results['_preds'], np.ndarray)
    assert '_probs' not in results


def test_rule_based_is_stable_respects_thresholds():
    data = _make_rule_based_data()
    results_tight = RuleBasedStrategy(sharpe_std_threshold=0.0).train(data).evaluate(data)
    results_loose = RuleBasedStrategy(sharpe_std_threshold=999.0,
                                      sharpe_degradation_threshold=999.0).train(data).evaluate(data)
    assert results_tight['is_stable'] is False
    assert results_loose['is_stable'] is True


def test_rule_based_function_returns_flat_dict():
    data = _make_rule_based_data()
    results = rule_based(data)
    assert isinstance(results['num_trades_test'], int)
    assert isinstance(results['is_stable'], bool)
    assert isinstance(results['_preds'], np.ndarray)


def test_rule_based_or_compound_condition():
    data = _make_rule_based_data()
    or_strategy = {
        'conditions': [
            {'id': 'macd_cross', 'type': 'threshold', 'column': 'macd_cross', 'operator': '>', 'value': 0},
            {'id': 'above_ema',  'type': 'threshold', 'column': 'above_ema',  'operator': '>', 'value': 0},
            {'id': 'entry', 'operator': 'or', 'operands': ['macd_cross', 'above_ema']},
        ],
        'entry': 'entry',
    }
    data['strategy'] = or_strategy
    results = RuleBasedStrategy().evaluate(data)
    assert 'num_trades_test' in results
    assert '_preds' in results


def test_rule_based_not_operator():
    data = _make_rule_based_data()
    not_strategy = {
        'conditions': [
            {'id': 'above_ema', 'type': 'threshold', 'column': 'above_ema', 'operator': '>', 'value': 0},
            {'id': 'entry', 'operator': 'not', 'operands': ['above_ema']},
        ],
        'entry': 'entry',
    }
    data['strategy'] = not_strategy
    results = RuleBasedStrategy().evaluate(data)
    assert '_preds' in results


def test_rule_based_empty_operands_raises():
    data = _make_rule_based_data()
    bad_strategy = {
        'conditions': [
            {'id': 'entry', 'operator': 'and', 'operands': []},
        ],
        'entry': 'entry',
    }
    data['strategy'] = bad_strategy
    try:
        RuleBasedStrategy().evaluate(data)
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'no operands' in str(e)


def test_rule_based_unknown_operator_raises():
    data = _make_rule_based_data()
    bad_strategy = {
        'conditions': [
            {'id': 'sig', 'type': 'threshold', 'column': 'macd_cross', 'operator': '>', 'value': 0},
            {'id': 'entry', 'operator': 'xor', 'operands': ['sig']},
        ],
        'entry': 'entry',
    }
    data['strategy'] = bad_strategy
    try:
        RuleBasedStrategy().evaluate(data)
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'Unknown logical operator' in str(e)

import numpy as np
import pandas as pd
import polars as pl

from limen.sfd.reference_architecture import XGBoostRegressor
from limen.sfd.reference_architecture import LogRegBinary
from limen.sfd.reference_architecture import RandomBinary
from limen.sfd.reference_architecture import TabPFNBinary


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

import inspect

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression

from limen.backtest.backtest_snapshot import BACKTEST_SNAPSHOT_COLUMNS
from limen.calibration import grid_threshold_optimizer
from limen.calibration import sklearn_probability_calibrator
from limen.experiment import CalibrationConfig
from limen.experiment import MLManifest
from limen.sfd.foundational_sfd import lightgbm_binary as foundational_lightgbm_binary
from limen.sfd.foundational_sfd import logreg_binary as foundational_logreg_binary
from limen.sfd.reference_architecture import LightGBMBinary
from limen.sfd.reference_architecture import LogRegBinary
from limen.sfd.reference_architecture import RandomBinary
from limen.sfd.reference_architecture import RuleBasedStrategy
from limen.sfd.reference_architecture import TabPFNBinary
from limen.sfd.reference_architecture import XGBoostRegressor
from limen.sfd.reference_architecture import lightgbm_binary
from limen.sfd.reference_architecture import logreg_binary
from limen.sfd.reference_architecture import random_binary
from limen.sfd.reference_architecture import tabpfn_binary
from limen.sfd.reference_architecture import xgboost_regressor
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

    for metric_col in BACKTEST_SNAPSHOT_COLUMNS:
        key = f'backtest_{metric_col}'
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


def test_logreg_wrapper_exposes_sklearn_constructor_params():

    sklearn_params = set(inspect.signature(LogisticRegression).parameters)
    wrapper_params = set(inspect.signature(logreg_binary).parameters) - {
        'data',
        'prediction_calibration_config',
    }

    assert sklearn_params <= wrapper_params


def test_logreg_foundational_params_cover_wrapper_model_surface():

    model_params = set(inspect.signature(logreg_binary).parameters) - {
        'data',
        'prediction_calibration_config',
    }

    assert model_params <= set(foundational_logreg_binary.params())


def test_logreg_numeric_class_weight_preserves_legacy_shorthand():

    data = _make_data(binary=True, with_price=False)
    model = LogRegBinary().train(data, solver='lbfgs', max_iter=200, class_weight=0.45)

    assert model.model.class_weight == {0: 0.45, 1: 1}


def test_logreg_class_weight_accepts_sklearn_native_values():

    data = _make_data(binary=True, with_price=False)
    model = LogRegBinary().train(data, solver='lbfgs', max_iter=200, class_weight='balanced')

    assert model.model.class_weight == 'balanced'


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

    config = CalibrationConfig(
        calibration_func=sklearn_probability_calibrator,
        calibration_params={'method': 'isotonic'},
        threshold_func=grid_threshold_optimizer,
    )
    data = _make_data(binary=True, with_price=True)
    model = TabPFNBinary(prediction_calibration_config=config).train(data, n_ensemble_configurations=2, device='cpu')
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

    assert results['drawdown_std_bps'] == 0.0
    assert isinstance(results['is_stable'], bool)

    assert isinstance(results['_preds'], np.ndarray)
    assert '_probs' not in results


def test_rule_based_is_stable_respects_thresholds():
    data = _make_rule_based_data()
    results_tight = RuleBasedStrategy(sharpe_std_threshold=0.0).train(data).evaluate(data)
    results_loose = RuleBasedStrategy(sharpe_std_threshold=999.0,
                                      sharpe_degradation_threshold=999.0).train(data).evaluate(data)
    assert results_tight['is_stable'] is False
    assert results_loose['is_stable'] is False


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


def test_compute_backtest_honours_injected_cost():

    base = _make_data(binary=True, with_price=True)
    model = LogRegBinary().train(_make_data(binary=True, with_price=True), solver='lbfgs', max_iter=200)

    zero = model.evaluate({**base, 'backtest_fee_bps': 0.0, 'backtest_slip_bps': 0.0})
    high = model.evaluate({**base, 'backtest_fee_bps': 20.0, 'backtest_slip_bps': 20.0})

    assert zero['backtest_cost_per_bar_bps'] == 0.0
    assert high['backtest_cost_per_bar_bps'] > zero['backtest_cost_per_bar_bps']
    assert zero['backtest_pnl_per_bar_bps'] > high['backtest_pnl_per_bar_bps']


def test_manifest_backtest_config_resolves_and_sweeps():

    swept = {}
    MLManifest().set_backtest_config(fee_bps='fee', slip_bps=3.0)._apply_backtest_cost(swept, {'fee': 7.0})
    assert swept['backtest_fee_bps'] == 7.0
    assert swept['backtest_slip_bps'] == 3.0

    literal = {}
    MLManifest().set_backtest_config(fee_bps=2.0, slip_bps=4.0)._apply_backtest_cost(literal, {})
    assert literal['backtest_fee_bps'] == 2.0
    assert literal['backtest_slip_bps'] == 4.0
    assert literal['backtest_notional_rate'] == 1.0

    sized = {}
    MLManifest().set_backtest_config(notional_rate='size')._apply_backtest_cost(sized, {'size': 0.1})
    assert sized['backtest_notional_rate'] == 0.1
    assert sized['backtest_fee_bps'] == 5.0

    unset = {}
    MLManifest()._apply_backtest_cost(unset, {})
    assert unset == {}


def test_architectures_do_not_expose_cost_parameters():

    architectures = [logreg_binary, random_binary, xgboost_regressor, rule_based]
    if tabpfn_binary is not None:
        architectures.append(tabpfn_binary)

    for architecture in architectures:
        params = inspect.signature(architecture).parameters
        assert 'fee_bps' not in params, f"{architecture.__name__} should not expose fee_bps"
        assert 'slip_bps' not in params, f"{architecture.__name__} should not expose slip_bps"
        assert 'notional_rate' not in params, f"{architecture.__name__} should not expose notional_rate"


def test_invalid_backtest_config_is_rejected():

    bad = (
        ({'fee_bps': -1.0}, {}),
        ({'slip_bps': -1.0}, {}),
        ({'fee_bps': 'fee'}, {'fee': float('inf')}),
        ({'slip_bps': 'slip'}, {'slip': float('nan')}),
        ({'fee_bps': 'fee'}, {'fee': 'oops'}),
    )
    for config_kwargs, round_params in bad:
        try:
            MLManifest().set_backtest_config(**config_kwargs)._apply_backtest_cost({}, round_params)
            assert False, 'Expected ValueError for invalid backtest cost'
        except ValueError as e:
            assert 'bps' in str(e)
            assert 'got ' in str(e)
            assert 'unknown search-param' not in str(e)

    try:
        MLManifest().set_backtest_config(fee_bps='missing_param')._apply_backtest_cost({}, {})
        assert False, 'Expected ValueError for unresolved param reference'
    except ValueError as e:
        assert 'unknown search-param' in str(e)


def test_invalid_notional_rate_config_is_rejected():

    for value in (1.5, 0.0, -0.1, float('inf')):
        try:
            MLManifest().set_backtest_config(notional_rate=value)._apply_backtest_cost({}, {})
            assert False, 'Expected ValueError for invalid notional_rate'
        except ValueError as e:
            assert 'notional_rate must be in (0, 1]' in str(e)
            assert 'got ' in str(e)


def test_lightgbm_binary_function_returns_metrics_and_model():

    data = _make_data(binary=True, with_price=True)
    results = lightgbm_binary(data, n_estimators=60, num_leaves=15)

    for key in ['recall', 'precision', 'fpr', 'auc', 'accuracy']:
        assert key in results, f"Missing results key: {key}"

    for key in ['confusion_tp', 'confusion_fp', 'confusion_tn', 'confusion_fn']:
        assert key in results, f"Missing confusion key: {key}"

    assert any(key.startswith('backtest_') for key in results)
    assert '_preds' in results
    assert isinstance(results['_model'], LightGBMBinary)

    repeat = lightgbm_binary(data, n_estimators=60, num_leaves=15)
    assert np.array_equal(results['_preds'], repeat['_preds'])


def test_lightgbm_binary_with_calibration_config():

    config = CalibrationConfig(
        calibration_func=sklearn_probability_calibrator,
        calibration_params={'method': 'isotonic'},
        threshold_func=grid_threshold_optimizer,
    )
    data = _make_data(binary=True, with_price=True)
    results = lightgbm_binary(data, n_estimators=60, num_leaves=15,
                              prediction_calibration_config=config)

    assert results['optimal_threshold'] is not None
    assert results['val_score'] is not None
    assert '_preds' in results


def test_lightgbm_binary_early_stops_on_validation_split():

    data = _make_data(binary=True, with_price=False)
    model = LightGBMBinary().train(data, n_estimators=500, early_stopping_rounds=5, verbosity=-1,
                                   deterministic=True, force_row_wise=True, random_state=42)

    assert model.model.best_iteration_ is not None
    assert model.model.best_iteration_ < 500


def test_lightgbm_binary_trains_without_validation_split():

    data = _make_data(binary=True, with_val=False, with_price=False)
    model = LightGBMBinary().train(data, n_estimators=20, verbosity=-1,
                                   deterministic=True, force_row_wise=True, random_state=42)

    assert model.model is not None
    preds = model.predict(data)['_preds']
    assert len(preds) == len(data['y_test'])


def test_lightgbm_binary_numeric_class_weight_preserves_legacy_shorthand():

    data = _make_data(binary=True, with_price=False)
    model = LightGBMBinary().train(data, n_estimators=20, class_weight=0.45, verbosity=-1,
                                   deterministic=True, force_row_wise=True, random_state=42)

    assert model.model.class_weight == {0: 0.45, 1: 1}

    balanced = LightGBMBinary().train(data, n_estimators=20, class_weight='balanced', verbosity=-1,
                                      deterministic=True, force_row_wise=True, random_state=42)
    assert balanced.model.class_weight == 'balanced'


def test_lightgbm_wrapper_exposes_lgbm_constructor_params():

    import lightgbm

    lgbm_params = set(inspect.signature(lightgbm.LGBMClassifier).parameters) - {
        'kwargs',
    }
    wrapper_params = set(inspect.signature(lightgbm_binary).parameters) - {
        'data',
        'prediction_calibration_config',
    }

    assert lgbm_params <= wrapper_params


def test_lightgbm_foundational_params_cover_wrapper_model_surface():

    model_params = set(inspect.signature(lightgbm_binary).parameters) - {
        'data',
        'prediction_calibration_config',
    }

    assert model_params <= set(foundational_lightgbm_binary.params())

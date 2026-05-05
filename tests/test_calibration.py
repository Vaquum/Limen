import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from limen.calibration import grid_threshold_optimizer
from limen.calibration import sklearn_probability_calibrator
from limen.experiment.manifest_core import CalibrationBuilder
from limen.experiment.manifest_core import CalibrationConfig
from limen.experiment.manifest_core import Manifest
from limen.metrics.balanced_metric import balanced_metric
from limen.sfd.reference_architecture.logreg_binary import LogRegBinary


def test_sklearn_probability_calibrator_no_future_warning() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4))
    y = (X[:, 0] > 0).astype(int)
    clf = LogisticRegression(max_iter=200).fit(X, y)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        calibrated = sklearn_probability_calibrator(clf, X, y)
        future = [w for w in caught if issubclass(w.category, FutureWarning)]
        assert not future, f'Unexpected FutureWarning: {future}'

    proba = calibrated.predict_proba(X)
    assert proba.shape == (100, 2)


def test_grid_threshold_optimizer_picks_best_balanced_threshold() -> None:
    y_val = np.asarray([0, 1, 1, 0], dtype=np.int8)
    y_val_proba = np.asarray([0.2, 0.8, 0.52, 0.45])

    threshold, score = grid_threshold_optimizer(
        y_val,
        y_val_proba,
        threshold_min=0.2,
        threshold_max=0.7,
        threshold_step=0.1,
    )

    assert threshold == pytest.approx(0.5)
    assert score == pytest.approx(np.sqrt(0.5))


def test_grid_threshold_optimizer_returns_default_when_all_thresholds_predict_zero() -> None:
    threshold, score = grid_threshold_optimizer(
        np.asarray([0, 1, 1, 0], dtype=np.int8),
        np.asarray([0.01, 0.02, 0.03, 0.04]),
        threshold_min=0.2,
        threshold_max=0.3,
        threshold_step=0.1,
        default_threshold=0.35,
    )

    assert threshold == 0.35
    assert score == 0.0


def test_grid_threshold_optimizer_returns_bounded_threshold() -> None:
    rng = np.random.default_rng(1)
    y_val = rng.integers(0, 2, 50).astype(np.int8)
    val_proba = rng.uniform(0.0, 1.0, 50)

    threshold, _ = grid_threshold_optimizer(
        y_val, val_proba, threshold_min=0.20, threshold_max=0.70
    )

    assert 0.20 <= threshold <= 0.70


def test_calibration_config_resolve_substitutes_string_params() -> None:
    config = CalibrationConfig(
        calibration_func=sklearn_probability_calibrator,
        threshold_func=grid_threshold_optimizer,
        threshold_params={'threshold_min': 'cal_min', 'threshold_max': 0.70},
    )

    resolved = config.resolve({'cal_min': 0.15})

    assert resolved.threshold_params['threshold_min'] == 0.15
    assert resolved.threshold_params['threshold_max'] == 0.70


def test_calibration_builder_done_raises_without_any_func() -> None:
    builder = CalibrationBuilder(Manifest())
    try:
        builder.done()
        assert False, 'expected ValueError'
    except ValueError as e:
        assert 'probability_calibration' in str(e) or 'threshold_function' in str(e)


def test_run_model_raises_when_arch_cannot_accept_calibration_config() -> None:
    def mock_arch(data: dict) -> dict:
        return {}

    m = (Manifest()
         .with_reference_architecture(mock_arch)
         .with_calibration()
         .threshold_function(func=grid_threshold_optimizer)
         .done())

    try:
        m.run_model({}, {})
        assert False, 'expected ValueError'
    except ValueError as e:
        assert 'prediction_calibration_config' in str(e)


def test_run_model_injects_resolved_prediction_calibration_config() -> None:
    received = {}

    def mock_arch(data: dict, prediction_calibration_config=None) -> dict:
        received['config'] = prediction_calibration_config
        return {}

    m = (Manifest()
         .with_reference_architecture(mock_arch)
         .with_calibration()
         .probability_calibration(func=sklearn_probability_calibrator, method='isotonic')
         .threshold_function(func=grid_threshold_optimizer, threshold_min='cal_min')
         .done())

    m.run_model({}, {'cal_min': 0.15})

    assert received['config'] is not None
    assert isinstance(received['config'], CalibrationConfig)
    assert received['config'].threshold_params['threshold_min'] == 0.15


def test_run_model_no_injection_when_calibration_not_set() -> None:
    received = {}

    def mock_arch(data: dict, **kwargs) -> dict:
        received['kwargs'] = kwargs
        return {}

    m = Manifest().with_reference_architecture(mock_arch)
    m.run_model({}, {})

    assert 'prediction_calibration_config' not in received['kwargs']


def test_resolve_model_kwargs_forwards_round_params_to_var_keyword_arch() -> None:
    received = {}

    def mock_arch(data: dict, **kwargs) -> dict:
        received['kwargs'] = kwargs
        return {}

    m = Manifest().with_reference_architecture(mock_arch)
    m.run_model({}, {'n_estimators': 100, 'device': 'cpu', '_id': 0})

    assert received['kwargs']['n_estimators'] == 100
    assert received['kwargs']['device'] == 'cpu'
    assert '_id' not in received['kwargs']


def test_run_model_no_injection_when_both_flags_false() -> None:
    received = {}

    def mock_arch(data: dict, prediction_calibration_config=None) -> dict:
        received['config'] = prediction_calibration_config
        return {}

    m = (Manifest()
         .with_reference_architecture(mock_arch)
         .with_calibration()
         .probability_calibration(func=sklearn_probability_calibrator)
         .threshold_function(func=grid_threshold_optimizer)
         .done())

    m.run_model({}, {'use_calibration': False, 'use_threshold': False})

    assert received['config'] is None


def test_run_model_masks_calibration_func_when_use_calibration_false() -> None:
    received = {}

    def mock_arch(data: dict, prediction_calibration_config=None) -> dict:
        received['config'] = prediction_calibration_config
        return {}

    m = (Manifest()
         .with_reference_architecture(mock_arch)
         .with_calibration()
         .probability_calibration(func=sklearn_probability_calibrator)
         .threshold_function(func=grid_threshold_optimizer)
         .done())

    m.run_model({}, {'use_calibration': False, 'use_threshold': True})

    assert received['config'] is not None
    assert received['config'].calibration_func is None
    assert received['config'].threshold_func is grid_threshold_optimizer


def test_run_model_masks_threshold_func_when_use_threshold_false() -> None:
    received = {}

    def mock_arch(data: dict, prediction_calibration_config=None) -> dict:
        received['config'] = prediction_calibration_config
        return {}

    m = (Manifest()
         .with_reference_architecture(mock_arch)
         .with_calibration()
         .probability_calibration(func=sklearn_probability_calibrator)
         .threshold_function(func=grid_threshold_optimizer)
         .done())

    m.run_model({}, {'use_calibration': True, 'use_threshold': False})

    assert received['config'] is not None
    assert received['config'].calibration_func is sklearn_probability_calibrator
    assert received['config'].threshold_func is None


def _make_logreg_data() -> dict:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 4))
    y = (X[:, 0] > 0).astype(np.int8)
    split = 120
    return {
        'x_train': X[:split], 'y_train': y[:split],
        'x_val': X[split:160], 'y_val': y[split:160],
        'x_test': X[160:], 'y_test': y[160:],
    }


def test_logreg_predict_without_calibration_unchanged() -> None:
    data = _make_logreg_data()
    model = LogRegBinary().train(data)
    result = model.predict(data)

    assert '_preds' in result
    assert '_probs' in result
    assert 'optimal_threshold' not in result
    assert 'val_score' not in result


def test_logreg_predict_threshold_only_returns_threshold_keys() -> None:
    data = _make_logreg_data()
    config = CalibrationConfig(
        threshold_func=grid_threshold_optimizer,
        threshold_params={'metric': balanced_metric},
    )
    model = LogRegBinary(prediction_calibration_config=config).train(data)
    result = model.predict(data)

    assert '_preds' in result
    assert '_probs' in result
    assert 'optimal_threshold' in result
    assert 'val_score' in result
    assert 0.0 <= result['optimal_threshold'] <= 1.0

from typing import Any, Protocol

import numpy as np


class CalibratorProtocol(Protocol):

    '''Protocol for probability calibration functions.'''

    def __call__(self,
                 clf: Any,
                 x_val: np.ndarray,
                 y_val: np.ndarray,
                 **params: Any) -> Any:
        ...


class ThresholdOptimizerProtocol(Protocol):

    '''Protocol for threshold optimisation functions.'''

    def __call__(self,
                 y_val: np.ndarray,
                 val_proba: np.ndarray,
                 **params: Any) -> tuple[float, float]:
        ...


class CalibrationConfigProtocol(Protocol):

    '''Structural type for a resolved calibration configuration.'''

    calibration_func: CalibratorProtocol | None
    calibration_params: dict[str, Any]
    threshold_func: ThresholdOptimizerProtocol | None
    threshold_params: dict[str, Any]


def fit_calibrator(model: Any,
                   config: CalibrationConfigProtocol,
                   x_val: np.ndarray,
                   y_val: np.ndarray) -> tuple[Any, float, float | None]:

    '''
    Fit calibrator on validation data.

    Args:
        model (Any): Fitted classifier with predict_proba method
        config (CalibrationConfigProtocol): Resolved calibration configuration
        x_val (np.ndarray): Validation features
        y_val (np.ndarray): Validation labels

    Returns:
        tuple: (fitted_calibrator, optimal_threshold, val_score)
            val_score is None when no threshold_func is configured
    '''

    fitted = (config.calibration_func(model, x_val, y_val, **config.calibration_params)
              if config.calibration_func is not None else model)
    val_proba = fitted.predict_proba(x_val)[:, 1]

    if config.threshold_func is not None:
        threshold, score = config.threshold_func(y_val, val_proba, **config.threshold_params)
    else:
        threshold, score = 0.5, None

    return fitted, threshold, score


def apply_calibrated_predict(model: Any,
                              config: CalibrationConfigProtocol,
                              data: dict[str, Any]) -> dict:

    '''
    Apply calibration and threshold optimisation to a fitted model's predictions.

    Args:
        model: Fitted classifier with predict_proba method
        config (CalibrationConfigProtocol): Resolved calibration configuration
        data (dict): Data dictionary with x_val, y_val, x_test keys

    Returns:
        dict: Results with '_preds', '_probs', 'optimal_threshold' and 'val_score'
            (val_score is None when no threshold_func is configured)
    '''

    fitted, threshold, score = fit_calibrator(model, config, data['x_val'], data['y_val'])
    test_proba = fitted.predict_proba(data['x_test'])[:, 1]
    preds = (test_proba >= threshold).astype(np.int8)
    return {'_preds': preds, '_probs': test_proba, 'optimal_threshold': threshold, 'val_score': score}

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV


class CalibratorProtocol(Protocol):

    '''Protocol for probability calibration functions.'''

    def __call__(self,
                 clf: Any,
                 x_val: npt.NDArray[Any] | pl.DataFrame,
                 y_val: npt.NDArray[Any] | pl.Series,
                 **params: Any) -> Any:
        ...


class ThresholdOptimizerProtocol(Protocol):

    '''Protocol for threshold optimisation functions.'''

    def __call__(self,
                 y_val: npt.NDArray[Any] | pl.Series,
                 val_proba: npt.NDArray[np.floating[Any]],
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
                   x_val: npt.NDArray[Any] | pl.DataFrame,
                   y_val: npt.NDArray[Any] | pl.Series) -> tuple[Any, float, float | None]:

    '''
    Fit calibrator on validation data.

    Args:
        model (Any): Fitted classifier with predict_proba method
        config (CalibrationConfigProtocol): Resolved calibration configuration
        x_val (np.ndarray or pl.DataFrame): Validation features
        y_val (np.ndarray or pl.Series): Validation labels

    Returns:
        tuple: (fitted_calibrator, optimal_threshold, val_score)
            val_score is None when no threshold_func is configured
    '''

    fitted = (config.calibration_func(model, x_val, y_val, **config.calibration_params)
              if config.calibration_func is not None else model)
    predict_proba = fitted.predict_proba
    val_input = (np.asarray(x_val) if isinstance(fitted, CalibratedClassifierCV)
                 or (fitted is model and isinstance(model, BaseEstimator)) else x_val)
    val_proba = predict_proba(val_input)[:, 1]

    threshold, score = (config.threshold_func(y_val, val_proba, **config.threshold_params)
                        if config.threshold_func is not None else (0.5, None))

    return fitted, threshold, score


def apply_calibrated_predict(model: Any,
                              config: CalibrationConfigProtocol,
                              data: dict[str, Any]) -> dict[str, Any]:

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
    predict_proba = fitted.predict_proba
    test_input = (np.asarray(data['x_test']) if isinstance(fitted, CalibratedClassifierCV)
                  or (fitted is model and isinstance(model, BaseEstimator)) else data['x_test'])
    test_proba = predict_proba(test_input)[:, 1]
    return {'_preds': (test_proba >= threshold).astype(np.int8), '_probs': test_proba, 'optimal_threshold': threshold, 'val_score': score}

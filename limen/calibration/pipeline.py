from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from limen.experiment.manifest_core import CalibrationConfig


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


def apply_calibrated_predict(model: Any,
                              config: 'CalibrationConfig',
                              data: dict[str, Any]) -> dict:

    '''
    Apply calibration and threshold optimisation to a fitted model's predictions.

    Args:
        model: Fitted classifier with predict_proba method
        config (CalibrationConfig): Resolved calibration configuration
        data (dict): Data dictionary with x_val, y_val, x_test keys

    Returns:
        dict: Results with '_preds', '_probs', 'optimal_threshold' and 'val_score'
            (val_score is None when no threshold_func is configured)
    '''

    fitted = (config.calibration_func(model, data['x_val'], data['y_val'], **config.calibration_params)
              if config.calibration_func is not None else model)
    val_proba = fitted.predict_proba(data['x_val'])[:, 1]
    test_proba = fitted.predict_proba(data['x_test'])[:, 1]

    if config.threshold_func is not None:
        threshold, score = config.threshold_func(data['y_val'], val_proba, **config.threshold_params)
    else:
        threshold, score = 0.5, None

    preds = (test_proba >= threshold).astype(np.int8)
    return {'_preds': preds, '_probs': test_proba, 'optimal_threshold': threshold, 'val_score': score}

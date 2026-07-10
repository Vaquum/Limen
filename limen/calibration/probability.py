from typing import Any

import numpy.typing as npt
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


def sklearn_probability_calibrator(clf: Any,
                                    x_val: npt.NDArray[Any] | pl.DataFrame,
                                    y_val: npt.NDArray[Any] | pl.Series,
                                    method: str = 'isotonic') -> Any:

    '''
    Fit isotonic or sigmoid calibration on a pre-fitted classifier.

    Args:
        clf: Pre-fitted classifier with predict_proba method
        x_val (np.ndarray or pl.DataFrame): Validation features for calibration fitting
        y_val (np.ndarray or pl.Series): Validation labels for calibration fitting
        method (str): Calibration method ('isotonic' or 'sigmoid')

    Returns:
        Any: Fitted calibrated classifier with predict_proba method
    '''

    return CalibratedClassifierCV(FrozenEstimator(clf), method=method).fit(x_val, y_val)

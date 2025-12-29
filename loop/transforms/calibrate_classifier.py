import numpy as np
from sklearn.calibration import CalibratedClassifierCV


def calibrate_classifier(clf,
                         x_val: np.ndarray,
                         y_val: np.ndarray,
                         x_sets: list,
                         method: str = 'isotonic') -> tuple:

    '''
    Apply isotonic or sigmoid calibration to a fitted classifier.

    Args:
        clf: Fitted classifier with predict_proba method
        x_val: Validation features for calibration fitting
        y_val: Validation labels for calibration fitting
        x_sets: List of feature arrays to get calibrated probabilities for
        method: Calibration method ('isotonic' or 'sigmoid')

    Returns:
        tuple: Calibrated probabilities for each array in x_sets (positive class)
    '''

    calibrator = CalibratedClassifierCV(clf, method=method, cv='prefit')
    calibrator.fit(x_val, y_val)

    return tuple(calibrator.predict_proba(x)[:, 1] for x in x_sets)

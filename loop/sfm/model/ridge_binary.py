import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV

from loop.metrics.binary_metrics import binary_metrics


def ridge_binary(data,
                 alpha=1.0,
                 tol=0.0001,
                 class_weight=None,
                 max_iter=None,
                 random_state=None,
                 fit_intercept=True,
                 solver='auto',
                 use_calibration=True,
                 calibration_method='sigmoid',
                 calibration_cv=3,
                 n_jobs=-1,
                 pred_threshold=0.5,
                 **kwargs):

    '''
    Execute Ridge binary classification with optional calibration, training and evaluation.

    Args:
        data (dict): Data dictionary with x_train, y_train, x_val, y_val, x_test, y_test
        alpha (float): Regularization strength
        tol (float): Tolerance for stopping criteria
        class_weight (str or dict): Class weights
        max_iter (int): Maximum iterations
        random_state (int): Random seed
        fit_intercept (bool): Whether to fit intercept
        solver (str): Solver algorithm
        use_calibration (bool): Whether to apply probability calibration
        calibration_method (str): Calibration method ('sigmoid' or 'isotonic')
        calibration_cv (int): CV folds for calibration
        pred_threshold (float): Threshold for binary predictions
        **kwargs: Additional parameters (ignored)

    Returns:
        dict: Results with binary metrics and predictions
    '''

    clf = RidgeClassifier(
        alpha=alpha,
        tol=tol,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        fit_intercept=fit_intercept,
        solver=solver
    )

    clf.fit(data['x_train'], data['y_train'])

    if use_calibration:
        calibrator = CalibratedClassifierCV(
            clf,
            method=calibration_method,
            cv=calibration_cv,
            n_jobs=n_jobs
        )
        calibrator.fit(data['x_val'], data['y_val'])
        y_proba = calibrator.predict_proba(data['x_test'])[:, 1]
    else:
        y_proba = clf.decision_function(data['x_test'])
        y_proba = 1 / (1 + np.exp(-y_proba))

    y_pred = (y_proba >= pred_threshold).astype(np.int8)

    round_results = binary_metrics(data, y_pred, y_proba)
    round_results['_preds'] = y_pred

    return round_results

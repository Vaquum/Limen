#!/usr/bin/env python3
"""
loop/sfm/model/ridge.py

Reusable Ridge model function that can be used with both manifest and legacy approaches.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from loop.metrics.binary_metrics import binary_metrics


def model(data,
          solver='auto',
          penalty='l2',
          dual=False,
          tol=0.0001,
          C='C',
          fit_intercept=True,
          intercept_scaling=1,
          class_weight=None,
          random_state=None,
          max_iter=None, 
          verbose=0,
          warm_start=False,
          n_jobs=1,
          **kwargs):
    """
    Train Ridge classifier with optional calibration.

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
        dict: Dictionary with trained model and predictions
            - 'model': Trained model
            - 'y_pred': Binary predictions
            - 'y_proba': Probability predictions
    """
    # Extract data
    X_train = data['x_train']
    y_train = data['y_train']
    X_test = data['x_test']

    # Initialize Ridge classifier
    clf = LogisticRegression(
        solver=solver,
        penalty=penalty,
        dual=dual,
        tol=tol,
        C=C,
        fit_intercept=fit_intercept,
        intercept_scaling=intercept_scaling,
        class_weight={0: class_weight, 1: 1},
        random_state=random_state,
        max_iter=max_iter,
        verbose=verbose,
        warm_start=warm_start,
        n_jobs=n_jobs,
    )

    # Train base model
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    round_results = binary_metrics(data, preds, probs)
    round_results['_preds'] = preds

    return round_results

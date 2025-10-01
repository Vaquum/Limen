#!/usr/bin/env python3
"""
loop/sfm/model/ridge.py

Reusable Ridge model function that can be used with both manifest and legacy approaches.
"""

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV


def model(data, 
          alpha=1.0,
          tol=0.0001,
          class_weight=None,
          max_iter=None,
          random_state=None,
          fit_intercept=True,
          solver='auto',
          use_calibration=False,
          calibration_method='sigmoid',
          calibration_cv=3,
          n_jobs=1,
          pred_threshold=0.5,
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
    X_val = data['x_val']
    y_val = data['y_val']
    X_test = data['x_test']
    
    # Initialize Ridge classifier
    clf = RidgeClassifier(
        alpha=alpha,
        tol=tol,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        fit_intercept=fit_intercept,
        solver=solver
    )
    
    # Train base model
    clf.fit(X_train, y_train)
    
    # Apply calibration if requested
    if use_calibration:
        calibrator = CalibratedClassifierCV(
            clf,
            method=calibration_method,
            cv=calibration_cv,
            n_jobs=n_jobs
        )
        calibrator.fit(X_val, y_val)
        model = calibrator
    else:
        model = clf
    
    # Generate predictions
    if hasattr(model, 'predict_proba'):
        y_proba_raw = model.predict_proba(X_test)
        # Extract positive class probabilities
        if len(y_proba_raw.shape) == 2 and y_proba_raw.shape[1] == 2:
            y_proba = y_proba_raw[:, 1]
        else:
            y_proba = y_proba_raw
    else:
        # Fallback for models without predict_proba
        y_proba = model.decision_function(X_test)
    
    # Apply threshold
    y_pred = (y_proba >= pred_threshold).astype(np.int8)
    
    return {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba
    }
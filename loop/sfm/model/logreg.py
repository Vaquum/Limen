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
        solver (str): Solver algorithm
        penalty (str): Regularization penalty
        dual (bool): Dual or primal formulation
        tol (float): Tolerance for stopping criteria
        C (float): Inverse of regularization strength
        fit_intercept (bool): Whether to fit intercept
        intercept_scaling (float): Intercept scaling
        class_weight (str or dict): Class weights
        random_state (int): Random seed
        max_iter (int): Maximum iterations
        verbose (int): Verbosity level
        warm_start (bool): Whether to reuse previous solution
        n_jobs (int): Number of parallel jobs
        **kwargs: Additional parameters (ignored)

    Returns:
        dict: Dictionary containing:
            - All metrics from binary_metrics
            - '_preds' (np.ndarray): Binary predictions on test set
    """
    X_train = data['x_train']
    y_train = data['y_train']
    X_test = data['x_test']

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

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    round_results = binary_metrics(data, preds, probs)
    round_results['_preds'] = preds

    return round_results

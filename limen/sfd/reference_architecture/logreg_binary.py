from sklearn.linear_model import LogisticRegression

from limen.metrics.binary_metrics import binary_metrics


def logreg_binary(data: dict,
                  solver: str = 'lbfgs',
                  penalty: str = 'l2',
                  dual: bool = False,
                  tol: float = 0.0001,
                  C: float = 1.0,
                  fit_intercept: bool = True,
                  intercept_scaling: float = 1,
                  class_weight: str | dict | None = None,
                  random_state: int | None = None,
                  max_iter: int = 100,
                  verbose: int = 0,
                  warm_start: bool = False,
                  n_jobs: int = -1) -> dict:

    '''
    Compute logistic regression binary predictions and evaluation metrics.

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

    Returns:
        dict: Results with binary metrics and predictions
    '''

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

    clf.fit(data['x_train'], data['y_train'])

    preds = clf.predict(data['x_test'])
    probs = clf.predict_proba(data['x_test'])[:, 1]

    round_results = binary_metrics(data, preds, probs)
    round_results['_preds'] = preds

    return round_results

from sklearn.linear_model import Ridge

from loop.metrics.continuous_metrics import continuous_metrics


def ridge_regression(data: dict,
                     alpha: float = 1.0,
                     solver: str = 'auto',
                     max_iter: int | None = None,
                     tol: float = 0.0001,
                     fit_intercept: bool = True,
                     random_state: int | None = None,
                     **kwargs) -> dict:

    '''
    Compute Ridge regression predictions and evaluation metrics.

    Args:
        data (dict): Data dictionary with x_train, y_train, x_val, y_val, x_test, y_test
        alpha (float): Regularization strength
        solver (str): Solver algorithm
        max_iter (int): Maximum iterations
        tol (float): Tolerance for stopping criteria
        fit_intercept (bool): Whether to fit intercept
        random_state (int): Random seed
        **kwargs: Additional parameters (ignored)

    Returns:
        dict: Results with continuous metrics and predictions
    '''

    ridge_params = {
        'alpha': alpha,
        'solver': solver,
        'max_iter': max_iter,
        'tol': tol,
        'fit_intercept': fit_intercept,
        'random_state': random_state,
    }

    if solver == 'lbfgs':
        ridge_params['positive'] = True

    ridge = Ridge(**ridge_params)

    ridge.fit(data['x_train'], data['y_train'])

    preds = ridge.predict(data['x_test'])

    round_results = continuous_metrics(data, preds)
    round_results['_preds'] = preds

    return round_results

from sklearn.linear_model import LogisticRegression

from loop.metrics.multiclass_metrics import multiclass_metrics


def logreg_multiclass(data,
                      penalty='l2',
                      C=1.0,
                      solver='lbfgs',
                      max_iter=100,
                      tol=0.0001,
                      class_weight=None,
                      l1_ratio=None,
                      fit_intercept=True,
                      random_state=None,
                      confidence_threshold=0.40,
                      n_jobs=-1,
                      **kwargs):

    '''
    Execute logistic regression multiclass classification with training and evaluation.

    Args:
        data (dict): Data dictionary with x_train, y_train, x_val, y_val, x_test, y_test
        penalty (str): Regularization penalty ('l1', 'l2', 'elasticnet')
        C (float): Inverse of regularization strength
        solver (str): Solver algorithm
        max_iter (int): Maximum iterations
        tol (float): Tolerance for stopping criteria
        class_weight (str or dict): Class weights
        l1_ratio (float): ElasticNet mixing parameter
        fit_intercept (bool): Whether to fit intercept
        random_state (int): Random seed
        confidence_threshold (float): Minimum probability threshold for non-zero predictions
        n_jobs (int): Number of parallel jobs
        **kwargs: Additional parameters (ignored)

    Returns:
        dict: Results with multiclass metrics and predictions
    '''

    if class_weight == 'None':
        class_weight = None

    params = {
        'penalty': penalty,
        'C': C,
        'solver': solver,
        'max_iter': max_iter,
        'tol': tol,
        'class_weight': class_weight,
        'fit_intercept': fit_intercept,
        'random_state': random_state,
        'verbose': 0,
        'n_jobs': n_jobs
    }

    if solver == 'liblinear':
        if penalty == 'elasticnet':
            params['penalty'] = 'l2'

    if penalty == 'elasticnet' and solver not in ['saga']:
        params['solver'] = 'saga'

    if penalty == 'l1' and solver not in ['liblinear', 'saga']:
        params['solver'] = 'saga'

    if params['penalty'] == 'elasticnet':
        params['l1_ratio'] = l1_ratio

    clf = LogisticRegression(**params)

    clf.fit(data['x_train'], data['y_train'])

    prediction_probs = clf.predict_proba(data['x_test'])

    preds = prediction_probs.argmax(axis=1)
    probs = prediction_probs.max(axis=1)

    preds[probs < confidence_threshold] = 0

    round_results = multiclass_metrics(data, preds, prediction_probs)
    round_results['_preds'] = preds

    return round_results

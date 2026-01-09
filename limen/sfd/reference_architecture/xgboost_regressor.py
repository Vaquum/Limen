import xgboost as xgb

from limen.metrics.continuous_metrics import continuous_metrics


def xgboost_regressor(data: dict,
                  learning_rate: float = 0.01,
                  max_depth: int = 3,
                  n_estimators: int = 500,
                  min_child_weight: int = 10,
                  subsample: float = 0.6,
                  colsample_bytree: float = 0.6,
                  gamma: float = 0.5,
                  reg_alpha: float = 0.5,
                  reg_lambda: float = 5.0,
                  objective: str = 'reg:squarederror',
                  booster: str = 'gbtree',
                  early_stopping_rounds: int | None = 50,
                  random_state: int = 42) -> dict:

    '''
    Compute XGBoost regression predictions and evaluation metrics.

    Args:
        data (dict): Data dictionary with x_train, y_train, x_val, y_val, x_test, y_test
        learning_rate (float): Boosting learning rate
        max_depth (int): Maximum tree depth
        n_estimators (int): Number of boosting rounds
        min_child_weight (int): Minimum sum of instance weight in a child
        subsample (float): Subsample ratio of training instances
        colsample_bytree (float): Subsample ratio of columns when constructing each tree
        gamma (float): Minimum loss reduction required to make split
        reg_alpha (float): L1 regularization term on weights
        reg_lambda (float): L2 regularization term on weights
        objective (str): Learning objective
        booster (str): Which booster to use
        early_stopping_rounds (int): Early stopping rounds
        random_state (int): Random seed

    Returns:
        dict: Results with continuous metrics and predictions
    '''

    model = xgb.XGBRegressor(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective=objective,
        booster=booster,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds if early_stopping_rounds is not None else None,
        eval_metric=['rmse']
    )

    model.fit(
        data['x_train'],
        data['y_train'],
        eval_set=[(data['x_val'], data['y_val'])],
        verbose=False
    )

    preds = model.predict(data['x_test'])

    round_results = continuous_metrics(data, preds)
    round_results['_preds'] = preds

    return round_results

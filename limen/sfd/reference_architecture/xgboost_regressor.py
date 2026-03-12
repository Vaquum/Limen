from typing import Any

import numpy as np
import xgboost as xgb

from limen.metrics.continuous_metrics import continuous_metrics
from limen.sfd.reference_architecture.base import ReferenceModel


class XGBoostRegressor(ReferenceModel):

    '''XGBoost regression model with train/evaluate interface.'''

    def train(self, data: dict, **params: Any) -> 'XGBoostRegressor':

        '''
        Train XGBoost regressor on provided data.

        Args:
            data (dict): Data dictionary with x_train, y_train, and optionally x_val, y_val
            **params: XGBoost hyperparameters

        Returns:
            XGBoostRegressor: Self with fitted model stored
        '''

        early_stopping_rounds = params.pop('early_stopping_rounds', 50)

        self.model = xgb.XGBRegressor(
            early_stopping_rounds=early_stopping_rounds if early_stopping_rounds is not None else None,
            eval_metric=['rmse'],
            **params,
        )

        has_val = 'x_val' in data and 'y_val' in data

        fit_kwargs = {'verbose': False}
        if has_val:
            fit_kwargs['eval_set'] = [(data['x_val'], data['y_val'])]
        else:
            self.model.set_params(early_stopping_rounds=None)

        self.model.fit(data['x_train'], data['y_train'], **fit_kwargs)

        return self

    def evaluate(self, data: dict, inline_metrics: bool = True) -> dict:

        '''
        Evaluate trained model on test data.

        Args:
            data (dict): Data dictionary with x_test, y_test, and optionally price_data_for_backtest
            inline_metrics (bool): Whether to include confusion_* and backtest_* keys

        Returns:
            dict: Metrics dict, optionally with flattened confusion_* and backtest_* keys
        '''

        preds = self.model.predict(data['x_test'])

        results = continuous_metrics(data, preds)
        results['_preds'] = preds

        if inline_metrics:
            y_test = np.asarray(data['y_test'])
            pred_direction = (preds > 0).astype(int)
            actual_direction = (y_test > 0).astype(int)
            results.update(self._compute_confusion(pred_direction, actual_direction))
            results.update(self._compute_backtest(pred_direction, data))

        return results


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

    model = XGBoostRegressor().train(
        data,
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
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
    )

    return model.evaluate(data, inline_metrics=False)

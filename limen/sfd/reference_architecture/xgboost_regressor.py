from typing import Any

import numpy as np
from typing_extensions import override

from limen._optional import require_optional
from limen.metrics.continuous_metrics import continuous_metrics
from limen.sfd.reference_architecture.base import ReferenceModel

__all__ = ['XGBoostRegressor', 'xgboost_regressor']


class XGBoostRegressor(ReferenceModel):

    '''XGBoost regression model with train/evaluate interface.'''

    deterministic = True

    @override
    def train(self, data: dict[str, Any], **params: Any) -> 'XGBoostRegressor':

        """Fit the training split; early stopping uses only nonempty validation data."""

        early_stopping_rounds = params.pop('early_stopping_rounds', 50)
        params = {key: value for key, value in params.items() if value is not None}
        params.setdefault('eval_metric', ['rmse'])

        xgb = require_optional('xgboost', 'XGBoost', 'boosting')

        self.model = xgb.XGBRegressor(early_stopping_rounds=early_stopping_rounds, **params)

        has_val = 'x_val' in data and 'y_val' in data and len(data['x_val']) > 0

        fit_kwargs: dict[str, Any] = {'verbose': False}
        if has_val:
            fit_kwargs['eval_set'] = [(data['x_val'], data['y_val'])]
        else:
            self.model.set_params(early_stopping_rounds=None)

        self.model.fit(data['x_train'], data['y_train'], **fit_kwargs)

        return self

    @override
    def predict(self, data: dict[str, Any]) -> dict[str, Any]:

        """Predict continuous returns for the test features."""

        preds = self.model.predict(data['x_test'])

        return {'_preds': preds}


    @override
    def evaluate(self, data: dict[str, Any], inline_metrics: bool = True) -> dict[str, Any]:
        '''Evaluate test predictions and preserve the configured boosting count.'''

        preds = self.predict(data)['_preds']

        results = continuous_metrics(data, preds)
        results['_preds'] = preds
        if data.get('_record_model_outputs'):
            booster = self.model.get_booster()
            best = booster.attr('best_iteration')
            results['best_iteration'] = (
                int(best) + 1 if best is not None and self.model.booster != 'gblinear'
                else booster.num_boosted_rounds()
            )

        if inline_metrics:
            y_test = np.asarray(data['y_test'])
            pred_direction = (preds > 0).astype(int)
            actual_direction = (y_test > 0).astype(int)
            results.update(self._compute_confusion(pred_direction, actual_direction, data.get('price_data_for_backtest')))
            results.update(self._compute_backtest(pred_direction, data))

        return results


def xgboost_regressor(data: dict[str, Any],
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
                  random_state: int = 42,
                  *,
                  eval_metric: str | list[str] | None = None,
                  max_leaves: int | None = None,
                  max_bin: int | None = None,
                  grow_policy: str | None = None,
                  sampling_method: str | None = None,
                  colsample_bylevel: float | None = None,
                  colsample_bynode: float | None = None,
                  max_delta_step: float | None = None,
                  tree_method: str | None = None,
                  num_parallel_tree: int | None = None,
                  base_score: float | None = None,
                  scale_pos_weight: float | None = None,
                  n_jobs: int | None = None,
                  verbosity: int | None = None,
                  device: str | None = None,
                  enable_categorical: bool | None = None,
                  max_cat_to_onehot: int | None = None,
                  max_cat_threshold: int | None = None,
                  importance_type: str | None = None,
                  interaction_constraints: str | list[list[int | str]] | None = None,
                  monotone_constraints: str | tuple[int, ...] | dict[str, int] | None = None,
                  multi_strategy: str | None = None,
                  validate_parameters: bool | None = None,
                  ) -> dict[str, Any]:

    """Train and evaluate a regressor, forwarding optional XGBoost controls."""

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
        eval_metric=eval_metric,
        max_leaves=max_leaves,
        max_bin=max_bin,
        grow_policy=grow_policy,
        sampling_method=sampling_method,
        colsample_bylevel=colsample_bylevel,
        colsample_bynode=colsample_bynode,
        max_delta_step=max_delta_step,
        tree_method=tree_method,
        num_parallel_tree=num_parallel_tree,
        base_score=base_score,
        scale_pos_weight=scale_pos_weight,
        n_jobs=n_jobs,
        verbosity=verbosity,
        device=device,
        enable_categorical=enable_categorical,
        max_cat_to_onehot=max_cat_to_onehot,
        max_cat_threshold=max_cat_threshold,
        importance_type=importance_type,
        interaction_constraints=interaction_constraints,
        monotone_constraints=monotone_constraints,
        multi_strategy=multi_strategy,
        validate_parameters=validate_parameters,
    )

    result = model.evaluate(data, inline_metrics=True)
    result['_model'] = model
    return result

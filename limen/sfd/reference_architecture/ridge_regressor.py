from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import Ridge
from typing_extensions import override

from limen.metrics.continuous_metrics import continuous_metrics
from limen.sfd.reference_architecture.base import ReferenceModel

__all__ = ['RidgeRegressor', 'ridge_regressor']

_Predictions = npt.NDArray[np.float32 | np.float64]


class _RidgeEstimator(Protocol):
    def fit(self, X: object, y: object) -> object: ...

    def predict(self, X: object) -> _Predictions: ...


def _ridge(params: dict[str, object]) -> _RidgeEstimator:
    return Ridge(**params)


class RidgeRegressor(ReferenceModel):
    """Sklearn ridge regression with Limen's train/predict/evaluate contract.

    Each train call fits a fresh estimator on the training split only.
    Solver behavior, validation, and warnings belong to sklearn. The
    deterministic flag is conservative because stochastic solvers are allowed.
    """

    deterministic = False

    def __init__(self) -> None:
        super().__init__()
        self.model = _ridge({})

    @override
    def train(self, data: dict[str, object], **params: object) -> 'RidgeRegressor':
        """Fit sklearn Ridge using native constructor parameters."""
        self.model = _ridge(params)
        _ = self.model.fit(data['x_train'], data['y_train'])
        return self

    @override
    def predict(self, data: dict[str, object]) -> dict[str, _Predictions]:
        """Return continuous predictions; only x_test is required."""
        return {'_preds': self.model.predict(data['x_test'])}

    @override
    def evaluate(self, data: dict[str, object], inline_metrics: bool = True) -> dict[str, object]:
        """Score continuous predictions, with optional standard directional metrics."""
        preds = self.predict(data)['_preds']
        results = continuous_metrics(data, preds)
        results['_preds'] = preds

        if inline_metrics:
            y_test = np.asarray(cast(npt.ArrayLike, data['y_test']))
            pred_direction = (preds > 0).astype(int)
            actual_direction = (y_test > 0).astype(int)
            results.update(self._compute_confusion(
                pred_direction, actual_direction, data.get('price_data_for_backtest'),
            ))
            results.update(self._compute_backtest(pred_direction, data))

        return results


def ridge_regressor(data: dict[str, object],
                    alpha: float = 1.0,
                    fit_intercept: bool = True,
                    copy_X: bool = True,
                    max_iter: int | None = None,
                    tol: float = 1e-4,
                    solver: str = 'auto',
                    positive: bool = False,
                    random_state: int | None = 42) -> dict[str, object]:
    """Fit and evaluate ridge with sklearn constructor semantics.

    alpha controls L2 regularization; fit_intercept controls the intercept.
    copy_X controls training-input copying. solver selects sklearn's solver;
    tol and max_iter apply to iterative solvers, not Cholesky or SVD.
    positive constrains coefficients and requires a compatible solver.
    random_state seeds stochastic solvers. Preprocessing belongs to the
    manifest, not this wrapper. Return standard metrics, _preds, and _model.
    """
    model = RidgeRegressor().train(
        data, alpha=alpha, fit_intercept=fit_intercept, copy_X=copy_X,
        max_iter=max_iter, tol=tol, solver=solver, positive=positive,
        random_state=random_state,
    )
    result = model.evaluate(data, inline_metrics=True)
    result['_model'] = model
    return result

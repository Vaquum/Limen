import inspect
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from typing_extensions import override

from limen.calibration import fit_calibrator
from limen.metrics.binary_metrics import binary_metrics
from limen.sfd.reference_architecture.base import ReferenceModel

if TYPE_CHECKING:
    from limen.experiment.manifest_core import CalibrationConfig


def _resolve_class_weight(class_weight: Any) -> Any:

    '''Preserve legacy numeric shorthand while allowing sklearn-native values.'''

    if isinstance(class_weight, (int, float)) and not isinstance(class_weight, bool):
        return {0: class_weight, 1: 1}
    return class_weight


def _drop_removed_sklearn_params(params: dict[str, Any]) -> dict[str, Any]:

    '''Drop compatibility params removed by the installed sklearn version.'''

    supported = set(inspect.signature(LogisticRegression).parameters)
    cleaned = dict(params)
    for name in {'multi_class'} - supported:
        cleaned.pop(name, None)
    return cleaned


class LogRegBinary(ReferenceModel):

    '''
    Logistic regression binary classifier with train/evaluate interface.

    Declared non-deterministic because sklearn solver refits are not
    bit-reproducible across BLAS builds and thread counts even with a fixed
    random_state, so reconstruction validation must use relative tolerance.
    '''

    deterministic = False

    def __init__(self, prediction_calibration_config: 'CalibrationConfig | None' = None) -> None:

        super().__init__()
        self.prediction_calibration_config = prediction_calibration_config
        self._fitted_calibrator: Any = None
        self._calibration_threshold: float = 0.5
        self._val_score: float | None = None

    @override
    def train(self, data: dict[str, Any], **params: Any) -> 'LogRegBinary':

        '''
        Train logistic regression classifier on provided data.

        Args:
            data (dict): Data dictionary with x_train, y_train
            **params: LogisticRegression hyperparameters

        Returns:
            LogRegBinary: Self with fitted model stored
        '''

        prediction_calibration_config = params.pop('prediction_calibration_config', None)
        if prediction_calibration_config is not None:
            self.prediction_calibration_config = prediction_calibration_config

        if 'class_weight' in params:
            params['class_weight'] = _resolve_class_weight(params['class_weight'])

        params = _drop_removed_sklearn_params(params)
        self.model = LogisticRegression(**params)
        self.model.fit(data['x_train'], data['y_train'])
        self._fitted_calibrator = None
        self._calibration_threshold = 0.5
        self._val_score = None

        return self


    @override
    def predict(self, data: dict[str, Any]) -> dict[str, Any]:

        '''
        Compute binary predictions from feature data.

        On the first call with calibration configured, fits the calibrator on
        validation data and stores it for reuse. Subsequent calls (including
        inference without val data) use the stored calibrator directly.

        Args:
            data (dict): Data dictionary. Training call requires x_val, y_val, x_test;
                inference call requires only x_test

        Returns:
            dict: Prediction results with '_preds' and '_probs' keys; calibrated path
                also includes 'optimal_threshold' and 'val_score'
        '''

        if self.prediction_calibration_config is not None:
            if self._fitted_calibrator is None:
                self._fitted_calibrator, self._calibration_threshold, self._val_score = fit_calibrator(
                    self.model, self.prediction_calibration_config, data['x_val'], data['y_val']
                )
            test_proba = self._fitted_calibrator.predict_proba(data['x_test'])[:, 1]
            preds = (test_proba >= self._calibration_threshold).astype(np.int8)
            return {
                '_preds': preds,
                '_probs': test_proba,
                'optimal_threshold': self._calibration_threshold,
                'val_score': self._val_score,
            }

        preds = self.model.predict(data['x_test'])
        probs = self.model.predict_proba(data['x_test'])[:, 1]
        return {'_preds': preds, '_probs': probs}


    @override
    def evaluate(self, data: dict[str, Any], inline_metrics: bool = True) -> dict[str, Any]:

        '''
        Evaluate trained model on test data.

        Args:
            data (dict): Data dictionary with x_test, y_test, and optionally price_data_for_backtest
            inline_metrics (bool): Whether to include confusion_* and backtest_* keys

        Returns:
            dict: Metrics dict, optionally with flattened confusion_* and backtest_* keys
        '''

        pred_result = self.predict(data)
        preds = pred_result['_preds']
        probs = pred_result['_probs']

        results = binary_metrics(data, preds, probs)
        results['_preds'] = preds

        results['optimal_threshold'] = pred_result.get('optimal_threshold')
        results['val_score'] = pred_result.get('val_score')

        if inline_metrics:
            results.update(self._compute_confusion(preds, data['y_test'], data.get('price_data_for_backtest')))
            results.update(self._compute_backtest(preds, data))

        return results


def logreg_binary(data: dict[str, Any],
                  solver: str = 'lbfgs',
                  penalty: str = 'l2',
                  dual: bool = False,
                  tol: float = 0.0001,
                  C: float = 1.0,
                  fit_intercept: bool = True,
                  intercept_scaling: float = 1,
                  class_weight: float | str | dict[int, float] | None = None,
                  random_state: int = 42,
                  max_iter: int = 100,
                  multi_class: str = 'deprecated',
                  verbose: int = 0,
                  warm_start: bool = False,
                  n_jobs: int | None = None,
                  l1_ratio: float | None = None,
                  prediction_calibration_config: 'CalibrationConfig | None' = None) -> dict[str, Any]:

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
        class_weight (float, str, or dict): Class weights. Numeric values keep the
            legacy `{0: value, 1: 1}` shorthand; strings/dicts pass through to sklearn.
        random_state (int): Random seed
        max_iter (int): Maximum iterations
        multi_class (str): Multiclass handling passthrough
        verbose (int): Verbosity level
        warm_start (bool): Whether to reuse previous solution
        n_jobs (int | None): Number of parallel jobs. Binary classification fits a
            single class, so parallel dispatch only adds joblib overhead; None keeps
            the fit in-process
        l1_ratio (float | None): Elastic-net mixing parameter
        prediction_calibration_config (CalibrationConfig | None): Optional calibration config

    Returns:
        dict: Results with binary metrics, predictions, inline confusion metrics,
            and backtest metrics when price_data_for_backtest is in data
    '''

    model = LogRegBinary(prediction_calibration_config=prediction_calibration_config).train(
        data,
        solver=solver,
        penalty=penalty,
        dual=dual,
        tol=tol,
        C=C,
        fit_intercept=fit_intercept,
        intercept_scaling=intercept_scaling,
        class_weight=class_weight,
        random_state=random_state,
        max_iter=max_iter,
        multi_class=multi_class,
        verbose=verbose,
        warm_start=warm_start,
        n_jobs=n_jobs,
        l1_ratio=l1_ratio,
    )

    result = model.evaluate(data, inline_metrics=True)
    result['_model'] = model
    return result

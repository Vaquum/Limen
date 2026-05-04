from typing import TYPE_CHECKING, Any

from sklearn.linear_model import LogisticRegression

from limen.calibration import apply_calibrated_predict
from limen.metrics.binary_metrics import binary_metrics
from limen.sfd.reference_architecture.base import ReferenceModel

if TYPE_CHECKING:
    from limen.experiment.manifest_core import CalibrationConfig


class LogRegBinary(ReferenceModel):

    '''Logistic regression binary classifier with train/evaluate interface.'''

    deterministic = True

    def __init__(self, prediction_calibration_config: 'CalibrationConfig | None' = None) -> None:

        super().__init__()
        self.prediction_calibration_config = prediction_calibration_config

    def train(self, data: dict, **params: Any) -> 'LogRegBinary':

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

        class_weight = params.pop('class_weight', None)
        if class_weight is not None:
            params['class_weight'] = {0: class_weight, 1: 1}

        self.model = LogisticRegression(**params)
        self.model.fit(data['x_train'], data['y_train'])

        return self

    def predict(self, data: dict) -> dict:

        '''
        Compute binary predictions from feature data.

        Args:
            data (dict): Data dictionary with x_val, y_val, x_test

        Returns:
            dict: Prediction results with '_preds' and '_probs' keys; calibrated path
                also includes 'optimal_threshold' and 'val_score'
        '''

        if self.prediction_calibration_config is not None:
            return apply_calibrated_predict(self.model, self.prediction_calibration_config, data)

        preds = self.model.predict(data['x_test'])
        probs = self.model.predict_proba(data['x_test'])[:, 1]
        return {'_preds': preds, '_probs': probs}


    def evaluate(self, data: dict, inline_metrics: bool = True) -> dict:

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

        if 'optimal_threshold' in pred_result:
            results['optimal_threshold'] = pred_result['optimal_threshold']
            results['val_score'] = pred_result['val_score']

        if inline_metrics:
            results.update(self._compute_confusion(preds, data['y_test'], data.get('price_data_for_backtest')))
            results.update(self._compute_backtest(preds, data))

        return results


def logreg_binary(data: dict,
                  solver: str = 'lbfgs',
                  penalty: str = 'l2',
                  dual: bool = False,
                  tol: float = 0.0001,
                  C: float = 1.0,
                  fit_intercept: bool = True,
                  intercept_scaling: float = 1,
                  class_weight: str | dict | None = None,
                  random_state: int = 42,
                  max_iter: int = 100,
                  verbose: int = 0,
                  warm_start: bool = False,
                  n_jobs: int = -1,
                  prediction_calibration_config: 'CalibrationConfig | None' = None) -> dict:

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
        verbose=verbose,
        warm_start=warm_start,
        n_jobs=n_jobs,
    )

    return model.evaluate(data, inline_metrics=True)

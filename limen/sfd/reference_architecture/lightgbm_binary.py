from typing import TYPE_CHECKING, Any

import lightgbm
import numpy as np

from limen.calibration import fit_calibrator
from limen.metrics.binary_metrics import binary_metrics
from limen.sfd.reference_architecture.base import ReferenceModel

if TYPE_CHECKING:
    from limen.experiment.manifest_core import CalibrationConfig

DEFAULT_EARLY_STOPPING_ROUNDS = 50


def _resolve_class_weight(class_weight: Any) -> Any:

    '''Preserve the legacy numeric shorthand while allowing sklearn-native values.'''

    if isinstance(class_weight, (int, float)) and not isinstance(class_weight, bool):
        return {0: class_weight, 1: 1}
    return class_weight


class LightGBMBinary(ReferenceModel):

    '''LightGBM binary classifier with train/evaluate interface.'''

    deterministic = True

    def __init__(self, prediction_calibration_config: 'CalibrationConfig | None' = None) -> None:

        super().__init__()
        self.prediction_calibration_config = prediction_calibration_config
        self._fitted_calibrator: Any = None
        self._calibration_threshold: float = 0.5
        self._val_score: float | None = None

    def train(self, data: dict, **params: Any) -> 'LightGBMBinary':

        '''
        Train LightGBM binary classifier on provided data.

        Fits with early stopping against the validation split when one is
        present and early_stopping_rounds is set; otherwise fits plainly on
        the training split.

        Args:
            data (dict): Data dictionary with x_train, y_train, and optionally x_val, y_val
            **params: LGBMClassifier hyperparameters plus early_stopping_rounds

        Returns:
            LightGBMBinary: Self with fitted model stored
        '''

        prediction_calibration_config = params.pop('prediction_calibration_config', None)
        if prediction_calibration_config is not None:
            self.prediction_calibration_config = prediction_calibration_config

        early_stopping_rounds = params.pop('early_stopping_rounds', DEFAULT_EARLY_STOPPING_ROUNDS)

        if 'class_weight' in params:
            params['class_weight'] = _resolve_class_weight(params['class_weight'])

        self.model = lightgbm.LGBMClassifier(**params)

        has_val = 'x_val' in data and 'y_val' in data and len(data['x_val']) > 0

        fit_kwargs: dict[str, Any] = {}
        if has_val and early_stopping_rounds:
            fit_kwargs['eval_set'] = [(np.asarray(data['x_val']), np.asarray(data['y_val']))]
            fit_kwargs['callbacks'] = [
                lightgbm.early_stopping(early_stopping_rounds, verbose=False),
            ]

        self.model.fit(np.asarray(data['x_train']), np.asarray(data['y_train']), **fit_kwargs)
        self._fitted_calibrator = None
        self._calibration_threshold = 0.5
        self._val_score = None

        return self

    def predict(self, data: dict) -> dict:

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
                    self.model, self.prediction_calibration_config,
                    np.asarray(data['x_val']), np.asarray(data['y_val'])
                )
            test_proba = self._fitted_calibrator.predict_proba(np.asarray(data['x_test']))[:, 1]
            preds = (test_proba >= self._calibration_threshold).astype(np.int8)
            return {
                '_preds': preds,
                '_probs': test_proba,
                'optimal_threshold': self._calibration_threshold,
                'val_score': self._val_score,
            }

        preds = self.model.predict(np.asarray(data['x_test']))
        probs = self.model.predict_proba(np.asarray(data['x_test']))[:, 1]
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

        results['optimal_threshold'] = pred_result.get('optimal_threshold')
        results['val_score'] = pred_result.get('val_score')

        if inline_metrics:
            results.update(self._compute_confusion(preds, data['y_test'], data.get('price_data_for_backtest')))
            results.update(self._compute_backtest(preds, data))

        return results


def lightgbm_binary(data: dict,
                    objective: str | None = None,
                    boosting_type: str = 'gbdt',
                    num_leaves: int = 31,
                    max_depth: int = -1,
                    learning_rate: float = 0.1,
                    n_estimators: int = 100,
                    subsample_for_bin: int = 200000,
                    min_split_gain: float = 0.0,
                    min_child_weight: float = 0.001,
                    min_child_samples: int = 20,
                    subsample: float = 1.0,
                    subsample_freq: int = 0,
                    colsample_bytree: float = 1.0,
                    reg_alpha: float = 0.0,
                    reg_lambda: float = 0.0,
                    class_weight: float | str | dict | None = None,
                    random_state: int = 42,
                    n_jobs: int | None = None,
                    importance_type: str = 'split',
                    early_stopping_rounds: int | None = DEFAULT_EARLY_STOPPING_ROUNDS,
                    deterministic: bool = True,
                    force_row_wise: bool = True,
                    verbosity: int = -1,
                    prediction_calibration_config: 'CalibrationConfig | None' = None) -> dict:

    '''
    Compute LightGBM binary predictions and evaluation metrics.

    Args:
        data (dict): Data dictionary with x_train, y_train, x_val, y_val, x_test, y_test
        objective (str | None): LightGBM objective; None lets LightGBM select the
            binary default. Must be a binary objective ('binary', 'cross_entropy')
        boosting_type (str): Boosting algorithm ('gbdt', 'dart', 'rf')
        num_leaves (int): Maximum tree leaves per booster
        max_depth (int): Maximum tree depth, -1 for no limit
        learning_rate (float): Boosting learning rate
        n_estimators (int): Number of boosting rounds (upper bound under early stopping)
        subsample_for_bin (int): Number of samples for histogram bin construction
        min_split_gain (float): Minimum loss reduction to split a leaf
        min_child_weight (float): Minimum sum of instance Hessian in a leaf
        min_child_samples (int): Minimum samples in a leaf
        subsample (float): Row subsampling ratio per boosting round
        subsample_freq (int): Subsampling frequency, 0 disables
        colsample_bytree (float): Feature subsampling ratio per tree
        reg_alpha (float): L1 regularization weight
        reg_lambda (float): L2 regularization weight
        class_weight (float, str, or dict): Class weights. Numeric values keep the
            legacy `{0: value, 1: 1}` shorthand; 'balanced'/dicts pass through to LightGBM
        random_state (int): Random seed
        n_jobs (int | None): Number of parallel threads, None for LightGBM default
        importance_type (str): Feature importance type ('split' or 'gain')
        early_stopping_rounds (int | None): Early-stopping patience against the
            validation split; None or 0 disables early stopping
        deterministic (bool): Enforce reproducible LightGBM training; pair with
            force_row_wise and a fixed random_state
        force_row_wise (bool): Force row-wise histogram building, required for
            deterministic training
        verbosity (int): LightGBM verbosity, -1 silences
        prediction_calibration_config (CalibrationConfig | None): Optional calibration config

    Returns:
        dict: Results with binary metrics, predictions, inline confusion metrics,
            and backtest metrics when price_data_for_backtest is in data
    '''

    model = LightGBMBinary(prediction_calibration_config=prediction_calibration_config).train(
        data,
        objective=objective,
        boosting_type=boosting_type,
        num_leaves=num_leaves,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample_for_bin=subsample_for_bin,
        min_split_gain=min_split_gain,
        min_child_weight=min_child_weight,
        min_child_samples=min_child_samples,
        subsample=subsample,
        subsample_freq=subsample_freq,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        importance_type=importance_type,
        early_stopping_rounds=early_stopping_rounds,
        deterministic=deterministic,
        force_row_wise=force_row_wise,
        verbosity=verbosity,
    )

    result = model.evaluate(data, inline_metrics=True)
    result['_model'] = model
    return result

from typing import TYPE_CHECKING, Any

import numpy as np

from tabpfn import TabPFNClassifier

from limen.calibration import fit_calibrator
from limen.metrics.binary_metrics import binary_metrics
from limen.sfd.reference_architecture.base import ReferenceModel
from limen.utils.data_dict_to_numpy import data_dict_to_numpy

if TYPE_CHECKING:
    from limen.experiment.manifest_core import CalibrationConfig


TABPFN_MODEL_PATH = 'tabpfn-v2-classifier-v2_default.ckpt'


class TabPFNBinary(ReferenceModel):

    '''TabPFN binary classifier with train/evaluate interface.'''

    deterministic = False

    def __init__(self, prediction_calibration_config: 'CalibrationConfig | None' = None) -> None:

        super().__init__()
        self.prediction_calibration_config = prediction_calibration_config
        self._fitted_calibrator: Any = None
        self._calibration_threshold: float = 0.5
        self._val_score: float | None = None

    def train(self, data: dict, **params: Any) -> 'TabPFNBinary':

        '''
        Train TabPFN classifier on provided data.

        Args:
            data (dict): Data dictionary with x_train, y_train
            **params: TabPFN hyperparameters including n_ensemble_configurations, device

        Returns:
            TabPFNBinary: Self with fitted model stored
        '''

        prediction_calibration_config = params.pop('prediction_calibration_config', None)
        if prediction_calibration_config is not None:
            self.prediction_calibration_config = prediction_calibration_config

        n_ensemble_configurations = params.pop('n_ensemble_configurations', 4)
        device = params.pop('device', 'cpu')

        arrays = data_dict_to_numpy(data, ['x_train', 'y_train'])

        self.model = TabPFNClassifier(
            device=device,
            n_estimators=n_ensemble_configurations,
            model_path=TABPFN_MODEL_PATH,
            ignore_pretraining_limits=True,
            **params,
        )

        self.model.fit(arrays['x_train'], arrays['y_train'])

        return self

    def predict(self, data: dict) -> dict:

        '''
        Compute binary predictions with optional calibration and threshold tuning.

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

        x_test = data_dict_to_numpy(data, ['x_test'])['x_test']

        if self.prediction_calibration_config is not None:
            if self._fitted_calibrator is None:
                arrays = data_dict_to_numpy(data, ['x_val', 'y_val'])
                self._fitted_calibrator, self._calibration_threshold, self._val_score = fit_calibrator(
                    self.model, self.prediction_calibration_config, arrays['x_val'], arrays['y_val']
                )
            test_proba = self._fitted_calibrator.predict_proba(x_test)[:, 1]
            preds = (test_proba >= self._calibration_threshold).astype(np.int8)
            return {
                '_preds': preds,
                '_probs': test_proba,
                'optimal_threshold': self._calibration_threshold,
                'val_score': self._val_score,
            }

        y_test_proba = self.model.predict_proba(x_test)[:, 1]
        y_pred = self.model.predict(x_test).astype(np.int8)
        return {'_preds': y_pred, '_probs': y_test_proba}


    def evaluate(self, data: dict, inline_metrics: bool = True) -> dict:

        '''
        Evaluate trained model on test data.

        Args:
            data (dict): Data dictionary with x_val, y_val, x_test, y_test,
                         and optionally price_data_for_backtest
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


def tabpfn_binary(data: dict,
                  n_ensemble_configurations: int = 4,
                  device: str = 'cpu',
                  prediction_calibration_config: 'CalibrationConfig | None' = None) -> dict:

    '''
    Compute TabPFN binary classification with optional calibration and threshold tuning.

    Args:
        data (dict): Data dictionary with x_train, y_train, x_val, y_val, x_test, y_test
        n_ensemble_configurations (int): Number of ensemble configurations for TabPFN
        device (str): Device to run on ('cpu' or 'cuda')
        prediction_calibration_config (CalibrationConfig | None): Optional calibration config

    Returns:
        dict: Results with binary metrics, predictions, inline confusion metrics,
            backtest metrics when price_data_for_backtest is in data,
            and optionally 'optimal_threshold', 'val_score' when calibration is configured
    '''

    model = TabPFNBinary(prediction_calibration_config=prediction_calibration_config).train(
        data,
        n_ensemble_configurations=n_ensemble_configurations,
        device=device,
    )

    result = model.evaluate(data, inline_metrics=True)
    result['_model'] = model
    return result

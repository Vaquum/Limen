from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from typing_extensions import override

from limen._optional import require_optional
from limen.calibration import fit_calibrator
from limen.metrics.binary_metrics import binary_metrics
from limen.sfd.reference_architecture.base import ReferenceModel
from limen.utils.data_dict_to_numpy import data_dict_to_numpy

if TYPE_CHECKING:
    from limen.experiment.manifest_core import CalibrationConfig


TABPFN_MODEL_PATH = 'tabpfn-v2-classifier-v2_default.ckpt'


class TabPFNBinary(ReferenceModel):


    '''TabPFN binary classifier with train/evaluate interface.'''
    inference_output = ('binary', True)

    deterministic = False

    def __init__(self, prediction_calibration_config: 'CalibrationConfig | None' = None) -> None:

        super().__init__()
        self.prediction_calibration_config = prediction_calibration_config
        self._fitted_calibrator: Any = None
        self._calibration_threshold: float = 0.5
        self._val_score: float | None = None

    @override
    def train(self, data: dict[str, Any], **params: Any) -> 'TabPFNBinary':

        '''Fit the upstream classifier with the supplied TabPFN controls.'''

        prediction_calibration_config = params.pop('prediction_calibration_config', None)
        if prediction_calibration_config is not None:
            self.prediction_calibration_config = prediction_calibration_config

        n_ensemble_configurations = params.pop('n_ensemble_configurations', 4)
        device = params.pop('device', 'cpu')
        model_path = params.pop('model_path', TABPFN_MODEL_PATH)
        ignore_pretraining_limits = params.pop('ignore_pretraining_limits', True)

        arrays = data_dict_to_numpy(data, ['x_train', 'y_train'])

        tabpfn = require_optional('tabpfn', 'TabPFN', 'tabpfn')

        self.model = tabpfn.TabPFNClassifier(
            device=device,
            n_estimators=n_ensemble_configurations,
            model_path=model_path,
            ignore_pretraining_limits=ignore_pretraining_limits,
            **params,
        )

        self.model.fit(arrays['x_train'], arrays['y_train'])
        self._fitted_calibrator = None
        self._calibration_threshold = 0.5
        self._val_score = None

        return self


    @override
    def predict(self, data: dict[str, Any]) -> dict[str, Any]:

        '''Predict binary outputs, fitting configured calibration once on validation.

        The first calibrated call needs x_val/y_val; later calls need only x_test.
        Return _preds/_probs and calibrated optimal_threshold/val_score when configured.
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


    @override
    def evaluate(self, data: dict[str, Any], inline_metrics: bool = True) -> dict[str, Any]:

        '''Return binary metrics and optional inline confusion/backtest metrics.'''

        pred_result = self.predict(data)
        preds = pred_result['_preds']
        probs = pred_result['_probs']
        self._record_probabilities(data, pred_result)

        results = binary_metrics(data, preds, probs)
        results['_preds'] = preds

        results['optimal_threshold'] = pred_result.get('optimal_threshold')
        results['val_score'] = pred_result.get('val_score')

        if inline_metrics:
            results.update(self._compute_confusion(preds, data['y_test'], data.get('price_data_for_backtest')))
            results.update(self._compute_backtest(preds, data))

        return results


def tabpfn_binary(data: dict[str, Any],
                  n_ensemble_configurations: int = 4,
                  device: object = 'cpu',
                  prediction_calibration_config: 'CalibrationConfig | None' = None, *,
                  categorical_features_indices: Sequence[int] | None = None,
                  softmax_temperature: float = 0.9,
                  balance_probabilities: bool = False,
                  average_before_softmax: bool = False,
                  model_path: object = TABPFN_MODEL_PATH,
                  ignore_pretraining_limits: bool = True,
                  inference_precision: object = 'auto',
                  fit_mode: Literal['low_memory', 'fit_preprocessors', 'fit_with_cache', 'batched'] = 'fit_preprocessors',
                  memory_saving_mode: str | bool | float = 'auto',
                  random_state: int | np.random.RandomState | np.random.Generator | None = 0,
                  n_preprocessing_jobs: int = 1,
                  inference_config: object = None,
                  differentiable_input: bool = False,
                  eval_metric: object = None,
                  tuning_config: object = None) -> dict[str, Any]:

    '''Fit TabPFN with Limen calibration; n_ensemble_configurations names n_estimators.'''

    model = TabPFNBinary(prediction_calibration_config=prediction_calibration_config).train(
        data,
        n_ensemble_configurations=n_ensemble_configurations,
        device=device,
        categorical_features_indices=categorical_features_indices,
        softmax_temperature=softmax_temperature,
        balance_probabilities=balance_probabilities,
        average_before_softmax=average_before_softmax,
        model_path=model_path,
        ignore_pretraining_limits=ignore_pretraining_limits,
        inference_precision=inference_precision,
        fit_mode=fit_mode,
        memory_saving_mode=memory_saving_mode,
        random_state=random_state,
        n_preprocessing_jobs=n_preprocessing_jobs,
        inference_config=inference_config,
        differentiable_input=differentiable_input,
        eval_metric=eval_metric,
        tuning_config=tuning_config,
    )

    result = model.evaluate(data, inline_metrics=True)
    result['_model'] = model
    return result

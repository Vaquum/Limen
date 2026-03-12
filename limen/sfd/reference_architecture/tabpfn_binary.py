from typing import Any

import numpy as np

from tabpfn import TabPFNClassifier

from limen.metrics.binary_metrics import binary_metrics
from limen.sfd.reference_architecture.base import ReferenceModel
from limen.transforms.calibrate_classifier import calibrate_classifier
from limen.transforms.optimize_binary_threshold import optimize_binary_threshold
from limen.utils.data_dict_to_numpy import data_dict_to_numpy


# TabPFN model checkpoint - auto-downloaded by tabpfn library on first use
TABPFN_MODEL_PATH = 'tabpfn-v2-classifier-v2_default.ckpt'

THRESHOLD_MIN = 0.20
THRESHOLD_MAX = 0.70
THRESHOLD_STEP = 0.05
DEFAULT_THRESHOLD = 0.35


class TabPFNBinary(ReferenceModel):

    '''TabPFN binary classifier with train/evaluate interface.'''

    def __init__(self) -> None:

        super().__init__()
        self._use_calibration = True
        self._threshold_metric = 'balanced'

    def train(self, data: dict, **params: Any) -> 'TabPFNBinary':

        '''
        Train TabPFN classifier on provided data.

        Args:
            data (dict): Data dictionary with x_train, y_train
            **params: TabPFN hyperparameters including n_ensemble_configurations, device

        Returns:
            TabPFNBinary: Self with fitted model stored
        '''

        self._use_calibration = params.pop('use_calibration', True)
        self._threshold_metric = params.pop('threshold_metric', 'balanced')

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

    def evaluate(self, data: dict, inline_metrics: bool = True) -> dict:

        '''
        Evaluate trained model on test data with threshold tuning.

        Args:
            data (dict): Data dictionary with x_val, y_val, x_test, y_test,
                         and optionally price_data_for_backtest
            inline_metrics (bool): Whether to include confusion_* and backtest_* keys

        Returns:
            dict: Metrics dict, optionally with flattened confusion_* and backtest_* keys
        '''

        arrays = data_dict_to_numpy(data, ['x_val', 'y_val', 'x_test'])
        x_val = arrays['x_val']
        y_val = arrays['y_val']
        x_test = arrays['x_test']

        if self._use_calibration:
            y_val_proba, y_test_proba = calibrate_classifier(
                self.model, x_val, y_val, [x_val, x_test], method='isotonic'
            )
        else:
            y_val_proba = self.model.predict_proba(x_val)[:, 1]
            y_test_proba = self.model.predict_proba(x_test)[:, 1]

        best_threshold, best_score = optimize_binary_threshold(
            y_val,
            y_val_proba,
            threshold_min=THRESHOLD_MIN,
            threshold_max=THRESHOLD_MAX,
            threshold_step=THRESHOLD_STEP,
            default_threshold=DEFAULT_THRESHOLD,
            metric=self._threshold_metric,
        )

        y_pred = (y_test_proba >= best_threshold).astype(np.int8)

        results = binary_metrics(data, y_pred, y_test_proba)
        results['optimal_threshold'] = best_threshold
        results['val_score'] = best_score
        results['_preds'] = y_pred

        if inline_metrics:
            results.update(self._compute_confusion(y_pred, data['y_test']))
            results.update(self._compute_backtest(y_pred, data))

        return results


def tabpfn_binary(data: dict,
                  n_ensemble_configurations: int = 4,
                  device: str = 'cpu',
                  use_calibration: bool = True,
                  threshold_metric: str = 'balanced') -> dict:

    '''
    Compute TabPFN binary classification with dynamic threshold tuning on validation set.

    Args:
        data (dict): Data dictionary with x_train, y_train, x_val, y_val, x_test, y_test
        n_ensemble_configurations (int): Number of ensemble configurations for TabPFN
        device (str): Device to run on ('cpu' or 'cuda')
        use_calibration (bool): Whether to apply isotonic calibration on validation set
        threshold_metric (str): Metric to optimize ('f1', 'precision', 'accuracy', 'balanced')

    Returns:
        dict: Results from binary_metrics with '_preds', 'optimal_threshold', 'val_score' added
    '''

    model = TabPFNBinary().train(
        data,
        n_ensemble_configurations=n_ensemble_configurations,
        device=device,
        use_calibration=use_calibration,
        threshold_metric=threshold_metric,
    )

    return model.evaluate(data, inline_metrics=False)

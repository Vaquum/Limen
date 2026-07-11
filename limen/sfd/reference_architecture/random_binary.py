from typing import Any

import numpy as np
from typing_extensions import override

from limen.metrics.binary_metrics import binary_metrics
from limen.sfd.reference_architecture.base import ReferenceModel


class RandomBinary(ReferenceModel):

    '''Random binary classifier with train/evaluate interface.'''

    deterministic = False

    def __init__(self) -> None:

        super().__init__()
        self._random_weights = 0.5

    @override
    def train(self, data: dict[str, Any], **params: Any) -> 'RandomBinary':

        '''
        Store random weights configuration. No actual training.

        Args:
            data (dict): Data dictionary (unused for random classifier)
            **params: Must include 'random_weights' if overriding default

        Returns:
            RandomBinary: Self with configuration stored
        '''

        _ = data
        self._random_weights = params.get('random_weights', 0.5)

        return self

    @override
    def predict(self, data: dict[str, Any]) -> dict[str, Any]:

        '''
        Compute random binary predictions.

        Args:
            data (dict): Data dictionary with x_test

        Returns:
            dict: Prediction results with '_preds' and '_probs' keys
        '''

        weights = [1 - self._random_weights, self._random_weights]

        preds = np.random.choice([0, 1], size=len(data['x_test']), p=weights)
        probs = np.where(preds == 1, 0.9, 0.1)

        return {'_preds': preds, '_probs': probs}


    @override
    def evaluate(self, data: dict[str, Any], inline_metrics: bool = True) -> dict[str, Any]:

        '''
        Evaluate random classifier on test data.

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

        if inline_metrics:
            results.update(self._compute_confusion(preds, data['y_test'], data.get('price_data_for_backtest')))
            results.update(self._compute_backtest(preds, data))

        return results


def random_binary(data: dict[str, Any],
                  random_weights: float = 0.5) -> dict[str, Any]:

    '''
    Random binary classifier for testing and demonstration purposes.

    Args:
        data (dict): Data dictionary with x_test, y_test
        random_weights (float): Probability weight for class 1 (0.0 to 1.0)

    Returns:
        dict: Results with binary metrics, predictions, inline confusion metrics,
            and backtest metrics when price_data_for_backtest is in data
    '''

    model = RandomBinary().train(data, random_weights=random_weights)
    result = model.evaluate(data, inline_metrics=True)
    result['_model'] = model
    return result

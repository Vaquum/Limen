from typing import Any

import numpy as np

from limen.metrics.binary_metrics import binary_metrics
from limen.sfd.reference_architecture.base import ReferenceModel


class RandomBinary(ReferenceModel):

    '''Random binary classifier with train/evaluate interface.'''

    def __init__(self) -> None:

        super().__init__()
        self._random_weights = 0.5

    def train(self, data: dict, **params: Any) -> 'RandomBinary':

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

    def evaluate(self, data: dict, inline_metrics: bool = True) -> dict:

        '''
        Evaluate random classifier on test data.

        Args:
            data (dict): Data dictionary with x_test, y_test, and optionally price_data_for_backtest
            inline_metrics (bool): Whether to include confusion_* and backtest_* keys

        Returns:
            dict: Metrics dict, optionally with flattened confusion_* and backtest_* keys
        '''

        weights = [self._random_weights, 1 - self._random_weights]

        preds = np.random.choice([0, 1], size=len(data['x_test']), p=weights)
        probs = np.random.choice([0.1, 0.9], size=len(data['x_test']), p=weights)

        results = binary_metrics(data, preds, probs)
        results['_preds'] = preds

        if inline_metrics:
            results.update(self._compute_confusion(preds, data['y_test']))
            results.update(self._compute_backtest(preds, data))

        return results


def random_binary(data: dict,
                  random_weights: float = 0.5) -> dict:

    '''
    Random binary classifier for testing and demonstration purposes.

    Args:
        data (dict): Data dictionary with x_test, y_test
        random_weights (float): Probability weight for class 1 (0.0 to 1.0)

    Returns:
        dict: Results with binary metrics and predictions
    '''

    return RandomBinary().train(data, random_weights=random_weights).evaluate(data, inline_metrics=False)

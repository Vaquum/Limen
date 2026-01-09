import numpy as np

from limen.metrics.binary_metrics import binary_metrics


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

    weights = [random_weights, 1 - random_weights]

    preds = np.random.choice([0, 1], size=len(data['x_test']), p=weights)
    probs = np.random.choice([0.1, 0.9], size=len(data['x_test']), p=weights)

    round_results = binary_metrics(data, preds, probs)
    round_results['_preds'] = preds

    return round_results

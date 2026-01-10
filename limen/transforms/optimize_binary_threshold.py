import numpy as np
from sklearn.metrics import f1_score, precision_score, accuracy_score

from limen.metrics.balanced_metric import balanced_metric


def optimize_binary_threshold(y_val: np.ndarray,
                               y_val_proba: np.ndarray,
                               threshold_min: float = 0.20,
                               threshold_max: float = 0.70,
                               threshold_step: float = 0.05,
                               default_threshold: float = 0.35,
                               metric: str = 'balanced') -> tuple:

    '''
    Find optimal binary classification threshold by sweeping over validation set.

    Args:
        y_val: Ground truth validation labels
        y_val_proba: Predicted probabilities for positive class
        threshold_min: Minimum threshold to test
        threshold_max: Maximum threshold to test
        threshold_step: Step size for threshold sweep
        default_threshold: Fallback threshold if no valid threshold found
        metric: Metric to optimize ('f1', 'precision', 'accuracy', 'balanced')

    Returns:
        tuple: (best_threshold, best_score)
    '''

    metric_fn = {
        'f1': lambda y, p: f1_score(y, p, zero_division=0),
        'precision': lambda y, p: precision_score(y, p, zero_division=0),
        'accuracy': accuracy_score,
        'balanced': balanced_metric,
    }.get(metric, balanced_metric)

    thresholds = np.arange(threshold_min, threshold_max + threshold_step, threshold_step)
    best_threshold = default_threshold
    best_score = -1.0

    for thresh in thresholds:
        y_val_pred = (y_val_proba >= thresh).astype(np.int8)
        if y_val_pred.sum() == 0:
            continue
        score = metric_fn(y_val, y_val_pred)
        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score

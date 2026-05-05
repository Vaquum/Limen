from collections.abc import Callable

import numpy as np

from limen.metrics.balanced_metric import balanced_metric


def grid_threshold_optimizer(y_val: np.ndarray,
                              val_proba: np.ndarray,
                              threshold_min: float = 0.20,
                              threshold_max: float = 0.70,
                              threshold_step: float = 0.05,
                              default_threshold: float = 0.35,
                              metric: Callable = balanced_metric) -> tuple[float, float]:

    '''
    Find optimal binary classification threshold by sweeping over a bounded range.

    Args:
        y_val (np.ndarray): Ground truth validation labels
        val_proba (np.ndarray): Predicted probabilities for positive class on validation set
        threshold_min (float): Minimum threshold to test
        threshold_max (float): Maximum threshold to test
        threshold_step (float): Step size for threshold sweep
        default_threshold (float): Fallback threshold if no valid threshold found
        metric (Callable): Scoring function with signature (y_true, y_pred) -> float

    Returns:
        tuple[float, float]: (best_threshold, best_score)
    '''

    thresholds = np.arange(threshold_min, threshold_max + threshold_step, threshold_step)
    preds_matrix = (val_proba[:, None] >= thresholds).astype(np.int8)
    valid_mask = preds_matrix.sum(axis=0) > 0
    if not valid_mask.any():
        return default_threshold, 0.0

    valid_thresholds = thresholds[valid_mask]
    valid_preds = preds_matrix[:, valid_mask]
    scores = np.array([metric(y_val, valid_preds[:, i]) for i in range(len(valid_thresholds))])
    best_idx = int(np.argmax(scores))
    return float(valid_thresholds[best_idx]), float(scores[best_idx])

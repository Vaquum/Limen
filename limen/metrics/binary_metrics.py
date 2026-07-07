from typing import Any, cast

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix

_BINARY_CLASS_COUNT = 2


def binary_metrics(data: dict, preds: list, probs: list) -> dict:

    '''
    Compute binary classification metrics from predictions and probabilities.

    Degenerate one-class outcomes never crash: fpr is NaN when y_test has no
    negatives, and auc is NaN when y_test contains a single class, mirroring
    the safe_ovr_auc convention.

    Args:
        data (dict): Data dictionary with 'y_test' key containing true binary labels
        preds (list): Predicted binary class labels
        probs (list): Predicted class probabilities

    Returns:
        dict: Dictionary containing recall, precision, fpr, auc, and accuracy metrics
    '''

    y_test = np.asarray(data['y_test'])
    negatives = int((y_test == 0).sum())
    false_positives = int(confusion_matrix(y_test, preds, labels=[0, 1])[0, 1])
    fpr = float('nan') if negatives == 0 else round(false_positives / negatives, 3)
    auc = (float('nan') if np.unique(y_test).size < _BINARY_CLASS_COUNT
           else round(roc_auc_score(y_test, probs), 3))

    round_results = {'recall': round(recall_score(y_test, preds, zero_division=cast(Any, 0.0)), 3),
                     'precision': round(precision_score(y_test, preds, zero_division=cast(Any, 0.0)), 3),
                     'fpr': fpr,
                     'auc': auc,
                     'accuracy': round(accuracy_score(y_test, preds), 3)}

    return round_results

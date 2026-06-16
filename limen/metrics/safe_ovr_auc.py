import numpy as np
from sklearn.metrics import roc_auc_score


def _probability_column_index(label: object, present: np.ndarray, n_columns: int) -> int | None:
    if n_columns == len(present):
        return int(np.where(present == label)[0][0])
    if isinstance(label, (np.integer, int)) and not isinstance(label, (np.bool_, bool)):
        index = int(label)
        if 0 <= index < n_columns:
            return index
    return None


def safe_ovr_auc(y_true: np.ndarray, probs: np.ndarray) -> float:

    '''
    Compute one-vs-rest AUC safely handling missing classes.

    Args:
        y_true (np.ndarray): True class labels, shape (n_samples,)
        probs (np.ndarray): Predicted probabilities, shape (n_samples, n_classes)

    Returns:
        float: Mean AUC across all valid class comparisons, or NaN if no valid AUC calculations can be made
    '''

    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    present = np.unique(y_true)
    aucs = []
    for c in present:
        pos = (y_true == c)
        neg = ~pos
        column_index = _probability_column_index(c, present, probs.shape[1])
        if pos.any() and neg.any() and column_index is not None:
            aucs.append(roc_auc_score(pos, probs[:, column_index]))
    return float('nan') if not aucs else np.mean(aucs)

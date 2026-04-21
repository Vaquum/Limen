import numpy as np
from sklearn.metrics import roc_auc_score

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
    class_to_index = {label: idx for idx, label in enumerate(present)}  # map class labels to column indices
    aucs = []
    for c in present:
        pos = (y_true == c)
        neg = ~pos
        if pos.any() and neg.any():  # need both to draw an ROC curve
            aucs.append(
                roc_auc_score(pos, probs[:, class_to_index[c]])  # use mapped index for class c
            )
    return float('nan') if not aucs else np.mean(aucs)

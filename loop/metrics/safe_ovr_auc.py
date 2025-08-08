import numpy as np
from sklearn.metrics import roc_auc_score

def safe_ovr_auc(y_true: list, proba: list) -> float:
    
    '''
    Compute one-vs-rest AUC safely handling missing classes.
    
    Args:
        y_true (list): True class labels
        proba (list): Predicted probabilities with shape (n_samples, n_classes) where each column corresponds to the probability of that class
        
    Returns:
        float: Mean AUC across all valid class comparisons, or NaN if no valid AUC calculations can be made
    '''
    
    present = np.unique(y_true)  # classes that exist in this fold
    class_to_index = {label: idx for idx, label in enumerate(present)}  # map class labels to column indices
    aucs = []
    for c in present:
        pos = (y_true == c)
        neg = ~pos
        if pos.any() and neg.any():  # need both to draw an ROC curve
            aucs.append(
                roc_auc_score(pos, proba[:, class_to_index[c]])  # use mapped index for class c
            )
    return float('nan') if not aucs else np.mean(aucs)

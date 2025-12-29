import numpy as np
from sklearn.metrics import precision_score


def balanced_metric(y_true, y_pred) -> float:

    '''
    Compute balanced precision metric that accounts for trade rate.

    Calculates precision * sqrt(trade_rate) to balance signal quality
    with trade frequency. Higher scores indicate better balance between
    accurate predictions and sufficient trading activity.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels

    Returns:
        float: Balanced metric score (0.0 if no positive predictions)
    '''

    if np.sum(y_pred) == 0:
        return 0.0

    prec = precision_score(y_true, y_pred, zero_division=0)
    trade_rate = np.sum(y_pred) / len(y_pred)

    return prec * np.sqrt(trade_rate)

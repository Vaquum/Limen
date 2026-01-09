from sklearn.metrics import accuracy_score, precision_score, recall_score

from limen.metrics.safe_ovr_auc import safe_ovr_auc


def multiclass_metrics(data: dict, preds: list, probs: list, average: str = 'macro') -> dict:
    
    '''
    Compute multiclass classification metrics from predictions and probabilities.
    
    Args:
        data (dict): Data dictionary with 'y_test' key containing true class labels
        preds (list): Predicted class labels
        probs (list): Predicted class probabilities
        average (str): Averaging strategy for precision and recall
        
    Returns:
        dict: Dictionary containing precision, recall, auc, and accuracy metrics
    '''

    round_results = {'precision': round(precision_score(data['y_test'], preds, average=average), 3),
                     'recall': round(recall_score(data['y_test'], preds, average=average), 3),
                     'auc': round(safe_ovr_auc(data['y_test'], probs), 3),
                     'accuracy': round(accuracy_score(data['y_test'], preds), 3)}

    return round_results 
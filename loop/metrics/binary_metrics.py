from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix


def binary_metrics(data: dict, preds: list, probs: list) -> dict:
    
    '''
    Compute binary classification metrics from predictions and probabilities.
    
    Args:
        data (dict): Data dictionary with 'y_test' key containing true binary labels
        preds (list): Predicted binary class labels
        probs (list): Predicted class probabilities
        
    Returns:
        dict: Dictionary containing recall, precision, fpr, auc, and accuracy metrics
    '''

    round_results = {'recall': round(recall_score(data['y_test'], preds), 3),
                     'precision': round(precision_score(data['y_test'], preds), 3),
                     'fpr': round(confusion_matrix(data['y_test'], preds)[0, 1] / (data['y_test'] == 0).sum(), 3),
                     'auc': round(roc_auc_score(data['y_test'], probs), 3),
                     'accuracy': round(accuracy_score(data['y_test'], preds), 3)}

    return round_results 
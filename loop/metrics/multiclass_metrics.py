from sklearn.metrics import accuracy_score, precision_score, recall_score

from loop.metrics.safe_ovr_auc import safe_ovr_auc


def multiclass_metrics(data, preds, probs, average='macro'):

    '''
    Takes in data dictionary and predicted values and returns a dictionary of metrics.

    Returns:
        dict: dictionary of metrics

    Args:
        data (dict): data dictionary
        preds (array): predicted binary values
        probs (array): predicted probabilities
    '''

    round_results = {'precision': round(precision_score(data['test_y'], preds, average=average), 3),
                     'recall': round(recall_score(data['test_y'], preds, average=average), 3),
                     'auc': round(safe_ovr_auc(data['test_y'], probs), 3),
                     'accuracy': round(accuracy_score(data['test_y'], preds), 3)}

    return round_results 
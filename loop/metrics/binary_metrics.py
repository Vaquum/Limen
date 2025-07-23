from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix


def binary_metrics(data, preds, probs):

    '''
    Takes in data dictionary and predicted values and returns a dictionary of metrics.

    Returns:
        dict: dictionary of metrics

    Args:
        data (dict): data dictionary
        preds (array): predicted binary values
        probs (array): predicted probabilities
    '''

    round_results = {'recall': round(recall_score(data['y_test'], preds), 3),
                     'precision': round(precision_score(data['y_test'], preds), 3),
                     'fpr': round(confusion_matrix(data['y_test'], preds)[0, 1] / (data['y_test'] == 0).sum(), 3),
                     'auc': round(roc_auc_score(data['y_test'], probs), 3),
                     'accuracy': round(accuracy_score(data['y_test'], preds), 3)}

    return round_results 
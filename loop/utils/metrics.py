import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix

from loop.utils.safe_ovr_auc import safe_ovr_auc


def continuous_metrics(data, preds):

    '''
    WARNING: This is still experimental.

    Takes in data dictionary and predicted values and returns a dictionary of metrics.

    Returns:
        dict: dictionary of metrics

    Args:
        data (dict): data dictionary
        preds (array): predicted continuous values
    '''

    bias = np.mean(preds - data['y_test'])
    
    round_results = {'mae': round(mean_absolute_error(data['y_test'], preds), 3),
                     'rmse': round(np.sqrt(mean_squared_error(data['y_test'], preds, squared=False)), 3),
                     'r2': round(r2_score(data['y_test'], preds), 3),
                     'bias': round(bias, 3),
                     'mape': round(np.mean(np.abs((data['y_test'] - preds) / data['y_test'])) * 100, 3)}

    return round_results


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
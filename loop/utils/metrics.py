from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np

# - [ ] Rename metrics_for_classification to binary_metrics
# - [ ] Rename metrics_for_regression to regression_metrics
# - [ ] Reflect the changes all around the codebase
# - [ ] Add multiclass_metrics function
# - [ ] Update CHANGELOG.md
# - [ ] Update the documentation


def continuous_metrics(data, pred_cont):

    '''
    WARNING: This is still experimental.

    Takes in data dictionary and predicted values and returns a dictionary of metrics.

    Returns:
        dict: dictionary of metrics

    Args:
        data (dict): data dictionary
        pred_cont (array): predicted continuous values
    '''

    bias = np.mean(pred_cont - data['y_test'])
    
    round_results = {'mae': round(mean_absolute_error(data['y_test'], pred_cont), 4),
                     'rmse': round(np.sqrt(mean_squared_error(data['y_test'], pred_cont, squared=False)), 4),
                     'r2': round(r2_score(data['y_test'], pred_cont), 4),
                     'bias': round(bias, 4),
                     'mape': round(np.mean(np.abs((data['y_test'] - pred_cont) / data['y_test'])) * 100, 4)}

    return round_results


def binary_metrics(data, pred_bin):

    '''
    Takes in data dictionary and predicted values and returns a dictionary of metrics.

    Returns:
        dict: dictionary of metrics

    Args:
        data (dict): data dictionary
        pred_bin (array): predicted binary values
    '''

    round_results = {'recall': round(recall_score(data['y_test'], pred_bin), 4),
                     'precision': round(precision_score(data['y_test'], pred_bin), 4),
                     'fpr': round(confusion_matrix(data['y_test'], pred_bin)[0, 1] / (data['y_test'] == 0).sum(), 4),
                     'auc': round(roc_auc_score(data['y_test'], pred_bin), 4),
                     'accuracy': round(accuracy_score(data['y_test'], pred_bin), 4)}

    return round_results


def multiclass_metrics(data, pred_bin):

    '''
    Takes in data dictionary and predicted values and returns a dictionary of metrics.

    Returns:
        dict: dictionary of metrics

    Args:
        data (dict): data dictionary
        pred_bin (array): predicted binary values
    '''

    round_results = {}

    return round_results
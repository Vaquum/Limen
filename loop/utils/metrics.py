from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def metrics_for_regression(data, pred):

    bias = np.mean(pred - data['y_test'])
    
    round_results = {'mae': round(mean_absolute_error(data['y_test'], pred), 4),
                     'rmse': round(np.sqrt(mean_squared_error(data['y_test'], pred)), 4),
                     'r2': round(r2_score(data['y_test'], pred), 4),
                     'bias': round(bias, 4),
                     'mape': round(np.mean(np.abs((data['y_test'] - pred) / data['y_test'])) * 100, 4)}

    return round_results


def metrics_for_classification(data, pred_bin):

    round_results = {'recall': round(recall_score(data['y_test'], pred_bin), 2),
                     'precision': round(precision_score(data['y_test'], pred_bin), 2),
                     'f1score': round(f1_score(data['y_test'], pred_bin), 2),
                     'auc': round(roc_auc_score(data['y_test'], pred_bin), 2),
                     'accuracy': round(accuracy_score(data['y_test'], pred_bin), 2)}

    return round_results

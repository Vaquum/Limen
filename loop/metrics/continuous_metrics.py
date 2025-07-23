import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
import numpy as np

from sklearn.metrics import mean_absolute_error, root_mean_squared_error 
from sklearn.metrics import mean_absolute_percentage_error, r2_score


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

    y_test = np.asarray(data['y_test'])
    preds = np.asarray(preds)

    bias = np.mean(preds - y_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds) * 100

    return {
        'bias': round(bias, 3),
        'mae': round(mae, 3),
        'rmse': round(rmse, 3),
        'r2': round(r2, 3),
        'mape': round(mape, 3),
    }
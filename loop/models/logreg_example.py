import polars as pl

from sklearn.linear_model import LogisticRegression
from loop.utils.metrics import binary_metrics
from loop.indicators import quantile_flag, wilder_rsi, atr, ppo, vwap, kline_imbalance, roc
from loop.utils.splits import split_sequential, split_data_to_prep_output
from loop.utils.generators import generate_parameter_range
from loop.transforms.logreg_transform import LogRegTransform
from loop.utils.scale_data_dict import scale_data_dict


def params(): 
    
    return {
        # these are for data prep
        'shift': [-1],
        'q': [0.38, 0.39, 0.40, 0.41],
        'train_size': [70, 80, 90],
        'val_size': [20, 10, 1],
        'test_size': [30, 20, 10],
        'roc_period': [3, 6, 12, 24],
        'penalty': ['l2'],
        # these are for the classifier
        'class_weight': [0.5, 0.6, 0.7, 0.8, 0.9],
        'C': [0.1, 0.5, 1, 2.5, 5],
        'max_iter': generate_parameter_range(30, 150, 20, 3),
        'solver': ['lbfgs', 'liblinear', 'sag'],
        'tol': [0.005, 0.01, 0.05, 0.1],
        
        # this is for doing feature testing
        'feature_to_drop': ['high',
                            'low',
                            'close',
                            'volume',
                            'maker_ratio',
                            'no_of_trades',
                            'atr',
                            'ppo',
                            'wilder_rsi',
                            'vwap',
                            'imbalance'],
    }


def prep(data, round_params):
    
    # Calculate ROC and filter NaN values
    data = roc(data, period=12).filter(~pl.col("roc").is_nan())
    
    # Calculate other technical indicators
    data = atr(data)
    data = ppo(data)[1:]
    data = wilder_rsi(data).filter(~pl.col("wilder_rsi").is_nan())[1:]
    data = vwap(data)
    data = kline_imbalance(data)
    
    # Split data BEFORE calculating quantile flags to prevent data leakage
    if 'train_size' not in round_params:
        split_data = split_sequential(data, (8, 1, 2))
    else:
        split_data = split_sequential(data, (round_params['train_size'],
                                             round_params['val_size'],
                                             round_params['test_size']))
    
    # Calculate quantile flag on training data and get the cutoff
    split_data[0], train_cutoff = quantile_flag(
        data=split_data[0],
        col='roc',
        q=round_params['q'],
        return_cutoff=True
    )
    
    # Apply the same cutoff to validation and test sets
    for i in range(1, len(split_data)):
        split_data[i] = quantile_flag(
            data=split_data[i],
            col='roc',
            q=round_params['q'],
            cutoff=train_cutoff
        )
    
    # Shift the quantile flag to create the target (predicting future quantile)
    for i in range(len(split_data)):
        split_data[i] = split_data[i].with_columns(
            pl.col("quantile_flag")
              .shift(round_params['shift'])
              .alias("quantile_flag")).drop_nulls("quantile_flag")
    
    # Define columns for the model
    cols = ['high',
            'low',
            'open',
            'close',
            'volume',
            'maker_ratio',
            'no_of_trades',
            'atr',
            'ppo',
            'wilder_rsi',
            'vwap',
            'imbalance',
            'quantile_flag']

    if 'feature_to_drop' in round_params:
    # TODO: Make every nth round skip this and keep all colls
        cols = [col for col in cols if col != round_params['feature_to_drop']]

    # Create data dictionary from splits
    data_dict = split_data_to_prep_output(split_data, cols)
    
    # Scale features using training data statistics
    scaler = LogRegTransform(data_dict['x_train'])

    for col in data_dict.keys():
        if col.startswith('x_'):
            data_dict[col] = scaler.transform(data_dict[col])

    data_dict['_scaler'] = scaler

    return data_dict


def model(data: dict, round_params):

    if round_params['solver'] == 'sag':
        if round_params['tol'] < 0.05:
            round_params['tol'] = 0.05
    
    clf = LogisticRegression(
        solver=round_params['solver'],
        penalty=round_params['penalty'],
        dual=False,
        tol=round_params['tol'],
        C=round_params['C'],
        fit_intercept=True,
        intercept_scaling=1,
        class_weight={0: round_params['class_weight'], 1: 1},
        random_state=None,
        max_iter=round_params['max_iter'],
        verbose=0,
        warm_start=False,
        n_jobs=None,
    )
    
    clf.fit(data['x_train'], data['y_train'])

    preds = clf.predict(data['x_test'])

    round_results = binary_metrics(data, preds)
    round_results['_preds'] = preds

    return round_results

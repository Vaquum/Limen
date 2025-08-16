import polars as pl

from sklearn.linear_model import LogisticRegression

from loop.metrics.binary_metrics import binary_metrics
from loop.features import quantile_flag, kline_imbalance, vwap
from loop.indicators import wilder_rsi, atr, ppo, roc
from loop.utils.splits import split_sequential, split_data_to_prep_output
from loop.transforms.logreg_transform import LogRegTransform


def params(): 
    
    return {
        # these are for data prep
        'shift': [-1, -2, -3, -4, -5],
        'q': [0.35, 0.38, 0.41, 0.44, 0.47, 0.50, 0.53],
        'roc_period': [1, 4, 12, 24, 144],
        'penalty': ['l2'],
        # these are for the classifier
        'class_weight': [0.45, 0.55, 0.65, 0.75, 0.85],
        'C': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'max_iter': [30, 60, 90, 120, 180, 240],
        'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'newton-cholesky'],
        'tol': [0.001, 0.01, 0.03, 0.1, 0.3],
    }


import polars as pl

from sklearn.linear_model import LogisticRegression

from loop.metrics.binary_metrics import binary_metrics
from loop.features import quantile_flag, kline_imbalance, vwap
from loop.indicators import wilder_rsi, atr, ppo, roc
from loop.utils.splits import split_sequential
from loop.transforms.logreg_transform import LogRegTransform

from loop.utils.splits import split_data_to_prep_output


def prep(data, round_params):

    all_datetimes = data['datetime'].to_list()
    
    # Calculate ROC and filter NaN values
    roc_col = f"roc_{round_params['roc_period']}"
    data = roc(data, period=round_params['roc_period']).filter(~pl.col(roc_col).is_nan())
    
    # Calculate other technical indicators
    atr_col = 'atr_14'
    data = atr(data)

    ppo_col = 'ppo_12_26'
    data = ppo(data)[1:]
    
    wilder_rsi_col = 'wilder_rsi_14'
    data = wilder_rsi(data).filter(~pl.col(wilder_rsi_col).is_nan())[1:]
    
    data = vwap(data)
    data = kline_imbalance(data)
    
    split_data = split_sequential(data, (8, 1, 2))
    
    # Calculate quantile flag on training data and get the cutoff
    split_data[0], train_cutoff = quantile_flag(
        data=split_data[0],
        col=roc_col,
        q=round_params['q'],
        return_cutoff=True
    )
    
    # Apply the same cutoff to validation and test sets
    for i in range(1, len(split_data)):
        split_data[i] = quantile_flag(
            data=split_data[i],
            col=roc_col,
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
    cols = ['datetime',
            'high',
            'low',
            'open',
            'close',
            'mean',
            'std',
            'median',
            'iqr',
            'volume',
            'maker_ratio',
            'no_of_trades',
            atr_col,
            ppo_col,
            wilder_rsi_col,
            'vwap',
            'imbalance',
            'quantile_flag']

    # Create data dictionary from splits
    data_dict = split_data_to_prep_output(split_data, cols, all_datetimes)
    
    # Scale features using training data statistics
    scaler = LogRegTransform(data_dict['x_train'])

    for col in data_dict.keys():
        if col.startswith('x_'):
            data_dict[col] = scaler.transform(data_dict[col])

    data_dict['_scaler'] = scaler

    return data_dict


def model(data: dict, round_params):

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
    probs = clf.predict_proba(data['x_test'])[:, 1]

    round_results = binary_metrics(data, preds, probs)
    round_results['_preds'] = preds

    return round_results

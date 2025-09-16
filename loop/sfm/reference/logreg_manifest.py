import polars as pl

from sklearn.linear_model import LogisticRegression

from loop.metrics.binary_metrics import binary_metrics
from loop.features import quantile_flag, compute_quantile_cutoff, kline_imbalance, vwap
from loop.indicators import wilder_rsi, atr, ppo, roc
from loop.utils.splits import split_sequential, split_data_to_prep_output
from loop.transforms.logreg_transform import LogRegTransform
from loop.manifest import Manifest, process_manifest, process_bars

def adaptive_bar_formation(data, **kwargs):
    return data


def manifest():
    return Manifest(
        required_bar_columns=[
            'datetime',
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
            'no_of_trades'
        ],

        bar_formation=(adaptive_bar_formation, {
            'bar_type': lambda p: p['bar_type'],
            'time_freq': lambda p: p['time_freq'],
            'volume_threshold': lambda p: p['volume_threshold'],
            'liquidity_threshold': lambda p: p['liquidity_threshold']
        }),

        ordered_transformations=[
            (roc, {'period': lambda p: p['roc_period']}),
            (atr, {'period': 14}),
            (ppo, {}),
            (wilder_rsi, {}),
            (vwap, {}),
            (kline_imbalance, {}),
            ([
                ('quantile_cutoff', compute_quantile_cutoff, {
                 'col': lambda p: f"roc_{p['roc_period']}",
                 'q': lambda p: p['q']
                })
             ],
             quantile_flag, {
                'col': lambda p: f"roc_{p['roc_period']}",
                'cutoff': lambda p: p['quantile_cutoff']
            }),
            (lambda data, shift: data.with_columns(
                pl.col("quantile_flag").shift(shift).alias("quantile_flag")
            ), {'shift': lambda p: p['shift']})
        ],

        target_column='quantile_flag',
        split_config=(8,1,2),
    )

def params():

    return {
        # data prep parameters
        'shift': [-1, -2, -3, -4, -5],
        'q': [0.35, 0.38, 0.41, 0.44, 0.47, 0.50, 0.53],
        'roc_period': [1, 4, 12, 24, 144],
        'penalty': ['l2'],
        # bar formation parameters
        'bar_type': ['base', 'time', 'volume', 'liquidity'],
        'time_freq': ['5m', '15m', '30m', '1h'],
        'volume_threshold': [500000, 1000000, 2000000],
        'liquidity_threshold': [10000000, 50000000],
        # classifier parameters
        'class_weight': [0.45, 0.55, 0.65, 0.75, 0.85],
        'C': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'max_iter': [30, 60, 90, 120, 180, 240],
        'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'newton-cholesky'],
        'tol': [0.001, 0.01, 0.03, 0.1, 0.3],
    }


def prep(data, round_params, manifest):

    all_datetimes, bar_data = process_bars(manifest, data, round_params)

    split_data = split_sequential(bar_data, manifest.split_config)

    split_data, _ = process_manifest(manifest, split_data, round_params)

    cols = split_data[0].columns

    # Create data dictionary from splits
    data_dict = split_data_to_prep_output(split_data, cols, all_datetimes)
    # Scale features using training data statistics (fit/transform pattern)
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
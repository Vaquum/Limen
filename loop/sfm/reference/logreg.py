import polars as pl

from sklearn.linear_model import LogisticRegression

from loop.metrics.binary_metrics import binary_metrics
from loop.features import quantile_flag
from loop.features import compute_quantile_cutoff
from loop.features import kline_imbalance
from loop.features import vwap
from loop.indicators import wilder_rsi
from loop.indicators import atr
from loop.indicators import ppo
from loop.indicators import roc
from loop.transforms.logreg_transform import LogRegTransform
from loop.utils.shift_column import shift_column
from loop.manifest import Manifest

# TODO: placeholder no-op bar foramtion. To be tied to actual bar formation code
def adaptive_bar_formation(data, **kwargs):
    return data

def manifest():
    return (Manifest()
        .set_split_config(8, 1, 2)

        .set_bar_formation(adaptive_bar_formation,
            bar_type='bar_type',
            time_freq='time_freq',
            volume_threshold='volume_threshold',
            liquidity_threshold='liquidity_threshold')
        .set_required_bar_columns([
            'datetime', 'high', 'low', 'open', 'close', 'mean',
            'volume', 'maker_ratio', 'no_of_trades', 'maker_volume', 'maker_liquidity'
        ])

        .add_indicator(roc, period='roc_period')
        .add_indicator(atr, period=14)
        .add_indicator(ppo)
        .add_indicator(wilder_rsi)

        .add_feature(vwap)
        .add_feature(kline_imbalance)

        .with_target('quantile_flag')
            .add_fitted_transform(quantile_flag)
                .fit_param('_quantile_cutoff', compute_quantile_cutoff, col='roc_{roc_period}', q='q')
                .with_params(col='roc_{roc_period}', cutoff='_quantile_cutoff')
            .add_transform(shift_column, shift='shift', column='target_column')
            .done()

        .set_scaler(LogRegTransform)
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

    data_dict = manifest.prepare_data(data, round_params)

    # SFM-specific modifications can go here if needed

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
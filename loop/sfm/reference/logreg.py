import polars as pl

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
from loop.data import compute_data_bars
import loop.sfm.model.logreg

def manifest():
    
    return (Manifest()
        .set_split_config(8, 1, 2)

        .set_bar_formation(compute_data_bars,
            bar_type='bar_type',
            trade_threshold='trade_threshold',
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

        .with_model()
            .set_model_function(
                loop.sfm.model.logreg.model,
                solver='solver',
                penalty='penalty',
                dual=False,
                tol='tol',
                C='C',
                fit_intercept=True,
                intercept_scaling=1,
                class_weight='class_weight',
                max_iter='max_iter',
                random_state=None,
                verbose=0,
                warm_start=False,
                n_jobs=None,
            )
            .done()
    )


def params():

    return {
        # data prep parameters
        'shift': [-1, -2, -3, -4, -5],
        'q': [0.35, 0.38, 0.41, 0.44, 0.47, 0.50, 0.53],
        'roc_period': [1, 4, 12, 24, 144],
        'penalty': ['l2'],
        # bar formation parameters
        'bar_type': ['base', 'trade', 'volume', 'liquidity'],
        'trade_threshold': [5000, 10000, 30000, 100000, 500000],
        'volume_threshold': [100, 250, 500, 750, 1000, 5000],
        'liquidity_threshold': [50000, 1000000, 5000000, 50000000, 100000000],
        # classifier parameters
        'class_weight': [0.45, 0.55, 0.65, 0.75, 0.85],
        'C': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'max_iter': [30, 60, 90, 120, 180, 240],
        'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'newton-cholesky'],
        'tol': [0.001, 0.01, 0.03, 0.1, 0.3],
    }

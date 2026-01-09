from limen.features import quantile_flag
from limen.features import compute_quantile_cutoff
from limen.features import kline_imbalance
from limen.features import vwap
from limen.indicators import wilder_rsi
from limen.indicators import atr
from limen.indicators import ppo
from limen.indicators import roc
from limen.scalers import LogRegScaler
from limen.transforms import shift_column_transform
from limen.experiment import Manifest
from limen.sfd.reference_architecture import logreg_binary
from limen.data import HistoricalData


def params():

    return {
        # data prep parameters
        'shift': [-1, -2, -3, -4, -5],
        'q': [0.35, 0.38, 0.41, 0.44, 0.47, 0.50, 0.53],
        'roc_period': [1, 4, 12, 24, 144],
        'penalty': ['l2'],
        # classifier parameters
        'class_weight': [0.45, 0.55, 0.65, 0.75, 0.85],
        'C': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'max_iter': [30, 60, 90, 120, 180, 240],
        'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'newton-cholesky'],
        'tol': [0.001, 0.01, 0.03, 0.1, 0.3],
    }


def manifest():

    return (Manifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'start_date_limit': '2025-01-01'}
        )
        .set_test_data_source(method=HistoricalData._get_data_for_test)
        .set_split_config(8, 1, 2)

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
            .add_transform(shift_column_transform, shift='shift', column='target_column')
            .done()

        .set_scaler(LogRegScaler)

        .with_model(logreg_binary)
    )

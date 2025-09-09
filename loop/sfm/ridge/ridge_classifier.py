import math
import warnings

import numpy as np
import pandas as pd
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifier

from loop.features import (
    atr_percent_sma,
    ichimoku_cloud,
    close_position,
    distance_from_high,
    distance_from_low,
    gap_high,
    price_range_position,
    range_pct,
    quantile_flag,
    trend_strength,
    volume_regime,
)
from loop.indicators import (
    roc,
    ppo,
    rolling_volatility,
    wilder_rsi,
)
from loop.metrics.binary_metrics import binary_metrics
from loop.transforms.linear_transform import LinearTransform
from loop.utils.splits import split_sequential, split_data_to_prep_output

warnings.filterwarnings('ignore')


def params() -> dict:

    return {
        'shift': (-1,),
        'q': (0.32, 0.35, 0.37),
        'roc_period': (4,),
        'atr_sma_period': (14, 28, 42),
        'ppo_fast': (8, 12, 20),
        'ppo_slow': (26, 32, 40),
        'ppo_signal': (9, 12),
        'rsi_period': (8, 14,),
        'volatility_window': (12, 24),
        'high_distance_period': (20, 40),
        'low_distance_period': (20, 40),
        'price_range_position_period': (50, 100),
        'tenkan_period': (9, 14),
        'kijun_period': (26, 30,),
        'senkou_b_period': (52, 60),
        'displacement': (26, 30),
        'trend_fast_period': (10, 20),
        'trend_slow_period': (50, 100),
        'lookback': (50, 100),
        'alpha': (2.0, 5.0, 8.0),
        'max_iter': (400,),
        'tol': (0.0001,),
        'fit_intercept': (True,),
        'class_weight': ('balanced',),
        'solver': ('auto',),
        'pred_threshold': (0.55,),
        'random_state': (42,),
        'n_jobs': (8,),
    }


def prep(data: pl.DataFrame, round_params: dict) -> dict:
    
    all_datetimes = data['datetime'].to_list()

    roc_period = round_params['roc_period']
    ppo_fast = round_params['ppo_fast']
    ppo_slow = round_params['ppo_slow']
    ppo_signal = round_params['ppo_signal']
    rsi_period = round_params['rsi_period']
    volatility_window = round_params['volatility_window']
    atr_sma_period = round_params['atr_sma_period']
    high_distance_period = round_params['high_distance_period']
    low_distance_period = round_params['low_distance_period']
    price_range_position_period = round_params['price_range_position_period']
    tenkan_period = round_params['tenkan_period']
    kijun_period = round_params['kijun_period']
    senkou_b_period = round_params['senkou_b_period']
    displacement = round_params['displacement']
    trend_fast_period = round_params['trend_fast_period']
    trend_slow_period = round_params['trend_slow_period']
    lookback = round_params['lookback']
    q = round_params['q']
    shift = round_params['shift']

    rename_map = {
        f'roc_{roc_period}': 'roc',
        f'ppo_{ppo_fast}_{ppo_slow}': 'ppo',
        f'ppo_signal_{ppo_signal}': 'ppo_signal',
        f'wilder_rsi_{rsi_period}': 'wilder_rsi',
        f'close_volatility_{volatility_window}': 'close_volatility',
    }

    required_cols = [
        'datetime', 'hour', 'minute', 'more_trade',
        'std', 'maker_ratio',
        'ppo', 'ppo_signal', 'wilder_rsi', 'close_position',
        'distance_from_high', 'distance_from_low',
        'gap_high', 'price_range_position', 'range_pct',
        'atr_percent_sma', 'close_volatility', 'kijun', 'senkou_a',
        'trend_strength', 'volume_regime',
    ]

    data_processed = (
        data.lazy()
        .with_columns([
            pl.col('datetime').dt.hour().alias('hour'),
            pl.col('datetime').dt.minute().alias('minute'),
        ])
        .with_columns([
            (
                (pl.col('datetime').dt.weekday() < 5) &
                pl.col('hour').is_between(13, 20)
            ).alias('more_trade')
        ])
        .pipe(roc, period=roc_period)
        .pipe(ppo, fast_period=ppo_fast, slow_period=ppo_slow, signal_period=ppo_signal)
        .pipe(wilder_rsi, period=rsi_period)
        .pipe(rolling_volatility, column='close', window=volatility_window)
        .pipe(atr_percent_sma, period=atr_sma_period)
        .pipe(ichimoku_cloud, tenkan_period=tenkan_period, kijun_period=kijun_period,
              senkou_b_period=senkou_b_period, displacement=displacement)
        .pipe(volume_regime, lookback=lookback)
        .pipe(close_position)
        .pipe(trend_strength, fast_period=trend_fast_period, slow_period=trend_slow_period)
        .pipe(distance_from_high, period=high_distance_period)
        .pipe(distance_from_low, period=low_distance_period)
        .pipe(gap_high)
        .pipe(price_range_position, period=price_range_position_period)
        .pipe(range_pct)
        .rename(rename_map)
        .drop_nulls(subset=required_cols)
        .collect()
    )

    split_data = split_sequential(data_processed, (6, 2, 2))

    split_data[0], train_cutoff = quantile_flag(
        data=split_data[0], col='roc', q=q, return_cutoff=True
    )

    processed_splits = []
    for i, split_df in enumerate(split_data):
        if i == 0:
            processed_split = split_data[0]
        else:
            processed_split = quantile_flag(
                data=split_df, col='roc', q=q, cutoff=train_cutoff
            )
        processed_split = (
            processed_split
            .with_columns([
                pl.col('quantile_flag')
                .shift(shift)
                .alias('quantile_flag')
            ])
            .drop_nulls(subset=['quantile_flag'])
        )
        processed_splits.append(processed_split)

    cols = required_cols + ['quantile_flag']

    data_dict = split_data_to_prep_output(
        processed_splits, cols, all_datetimes
    )

    scaler = LinearTransform(x_train=data_dict['x_train'])

    for key in ('x_train', 'x_val', 'x_test'):
        data_dict[key] = scaler.transform(data_dict[key])

    data_dict['_scaler'] = scaler
    return data_dict


def model(data: dict, round_params: dict) -> dict:
    
    alpha = round_params['alpha']
    tol = round_params['tol']
    class_weight = round_params['class_weight']
    max_iter = round_params['max_iter']
    random_state = round_params['random_state']
    fit_intercept = round_params['fit_intercept']
    solver = round_params['solver']
    n_jobs = round_params['n_jobs']
    pred_threshold = round_params['pred_threshold']

    clf = RidgeClassifier(
        alpha=alpha,
        tol=tol,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        fit_intercept=fit_intercept,
        solver=solver,
    )

    clf.fit(data['x_train'], data['y_train'])

    calibrator = CalibratedClassifierCV(
        clf, method='sigmoid', cv='prefit', n_jobs=n_jobs
    )
    calibrator.fit(data['x_val'], data['y_val'])

    probs = calibrator.predict_proba(data['x_test'])[:, 1]
    preds = (probs >= pred_threshold).astype(np.int8)

    round_results = binary_metrics(data, preds, probs)
    round_results['_preds'] = preds

    return round_results

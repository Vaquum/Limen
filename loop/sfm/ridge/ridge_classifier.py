#!/usr/bin/env python3
'Long-Only Regime Binary Classifier Using Ridge Regression'

from loop.manifest import Manifest
from loop.features import atr_percent_sma
from loop.features import ichimoku_cloud
from loop.features import close_position
from loop.features import distance_from_high
from loop.features import distance_from_low
from loop.features import gap_high
from loop.features import price_range_position
from loop.features import range_pct
from loop.features import quantile_flag
from loop.features import trend_strength
from loop.features import volume_regime
from loop.features import compute_quantile_cutoff
from loop.indicators import roc
from loop.indicators import ppo
from loop.indicators import rolling_volatility
from loop.indicators import wilder_rsi
from loop.transforms.linear_transform import LinearTransform
from loop.utils import shift_column
from loop.sfm.model import ridge_binary


def params():

    return {
        'shift': [-1],
        'q': [0.32, 0.35, 0.37],
        'roc_period': [4],
        'atr_sma_period': [14, 28, 42],
        'ppo_fast': [8, 12, 20],
        'ppo_slow': [26, 32, 40],
        'ppo_signal': [9, 12],
        'rsi_period': [8, 14],
        'volatility_window': [12, 24],
        'high_distance_period': [20, 40],
        'low_distance_period': [20, 40],
        'price_range_position_period': [50, 100],
        'tenkan_period': [9, 14],
        'kijun_period': [26, 30],
        'senkou_b_period': [52, 60],
        'displacement': [26, 30],
        'trend_fast_period': [10, 20],
        'trend_slow_period': [50, 100],
        'lookback': [50, 100],
        'alpha': [2.0, 5.0, 8.0],
        'max_iter': [400],
        'tol': [0.0001],
        'fit_intercept': [True],
        'class_weight': ['balanced'],
        'solver': ['auto'],
        'use_calibration': [True],
        'calibration_method': ['sigmoid'],
        'calibration_cv': [3],
        'n_jobs': [8],
        'pred_threshold': [0.55],
        'random_state': [42],
    }


def manifest():

    return (Manifest()
        .set_split_config(6, 2, 2)
        
        .set_required_bar_columns([
            'datetime', 'std', 'maker_ratio',        
        ])
        
        # Indicators
        .add_indicator(roc, period='roc_period')
        .add_indicator(ppo, fast_period='ppo_fast', slow_period='ppo_slow', signal_period='ppo_signal')
        .add_indicator(wilder_rsi, period='rsi_period')

        # Features
        .add_feature(rolling_volatility, column='close', window='volatility_window')
        .add_feature(atr_percent_sma, period='atr_sma_period')
        .add_feature(ichimoku_cloud, tenkan_period='tenkan_period', kijun_period='kijun_period',
                    senkou_b_period='senkou_b_period', displacement='displacement')
        .add_feature(volume_regime, lookback='lookback')
        .add_feature(close_position)
        .add_feature(trend_strength, fast_period='trend_fast_period', slow_period='trend_slow_period')
        .add_feature(distance_from_high, period='high_distance_period')
        .add_feature(distance_from_low, period='low_distance_period')
        .add_feature(gap_high)
        .add_feature(price_range_position, period='price_range_position_period')
        .add_feature(range_pct)
        
        # Target
        .with_target('quantile_flag')
            .add_fitted_transform(quantile_flag)
                .fit_param('_cutoff', compute_quantile_cutoff, col='roc_{roc_period}', q='q')
                .with_params(col='roc_{roc_period}', cutoff='_cutoff')
            .add_transform(shift_column, shift='shift', column='target_column')
            .done()

        # Scaler
        .set_scaler(LinearTransform)

        # Model
        .with_model(ridge_binary)
    )

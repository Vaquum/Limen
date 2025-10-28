#!/usr/bin/env python3
'Long-Only Regime Binary Classifier Using Ridge Regression'

from loop.indicators import roc
from loop.indicators import ppo
from loop.indicators import rolling_volatility
from loop.indicators import wilder_rsi
from loop.indicators import atr
from loop.indicators import stochastic_oscillator
from loop.indicators import cci
from loop.indicators import bollinger_bands
from loop.features import ichimoku_cloud
from loop.features import close_position
from loop.features import gap_high
from loop.features import price_range_position
from loop.features import range_pct
from loop.features import quantile_flag
from loop.features import trend_strength
from loop.features import volume_regime
from loop.features import compute_quantile_cutoff
from loop.features import sma_crossover
from loop.manifest import Manifest
from loop.transforms.linear_transform import LinearTransform
from loop.utils import shift_column
from loop.sfm.model import ridge_binary

def params():

    return {
        # Target
        'shift': [-1, -2, -3],
        'q': [0.32, 0.35, 0.37],
        'roc_period': [1, 2, 4, 8],

        # Indicators & Features
        'ppo_fast': [8, 12, 20],
        'ppo_slow': [26, 32, 40],
        'ppo_signal': [9, 12, 15],
        'rsi_period': [8, 14],
        'atr_period': [6, 12, 24],
        'volatility_window': [12, 24],
        'price_range_position_period': [50, 100],
        'tenkan_period': [9, 14],
        'kijun_period': [26, 30],
        'senkou_b_period': [52, 60],
        'displacement': [26, 30],
        'trend_fast_period': [10, 20],
        'trend_slow_period': [50, 100],
        'lookback': [50, 100],
        'soch_window_k': [6, 12, 20],
        'soch_window_d': [4, 8],
        'cci_window': [10, 30, 80],
        'bb_window': [10, 30, 80],
        'bb_std': [1.5, 2.0, 3.0],
        'sma_cross_short_window': [6, 14, 30],
        'sma_cross_long_window': [50, 100, 150],

        # Model
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
        'pred_threshold': [0.45, 0.50, 0.55],
        'random_state': [42],
    }


def manifest():

    return (Manifest()
        .set_split_config(6, 2, 2)
        
        .set_required_bar_columns([
                'datetime', 'open', 'high', 'low', 'close'
        ])
        
        # # Indicators
        .add_indicator(roc, period='roc_period')
        .add_indicator(ppo, fast_period='ppo_fast', slow_period='ppo_slow', signal_period='ppo_signal')
        .add_indicator(wilder_rsi, period='rsi_period')
        .add_indicator(atr, period='atr_period')

        # # Features
        .add_feature(rolling_volatility, column='close', window='volatility_window')
        .add_feature(ichimoku_cloud, tenkan_period='tenkan_period', kijun_period='kijun_period',
                    senkou_b_period='senkou_b_period', displacement='displacement')
        .add_feature(volume_regime, lookback='lookback')
        .add_feature(close_position)
        .add_feature(trend_strength, fast_period='trend_fast_period', slow_period='trend_slow_period')
        .add_feature(gap_high)
        .add_feature(price_range_position, period='price_range_position_period')
        .add_feature(range_pct)
        
        .add_feature(stochastic_oscillator, window_k='soch_window_k', window_d='soch_window_d')
        .add_feature(cci, window='cci_window')
        .add_feature(sma_crossover, short_window='sma_cross_short_window', long_window='sma_cross_long_window')
        .add_feature(bollinger_bands, window='bb_window', num_std='bb_std')
        
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

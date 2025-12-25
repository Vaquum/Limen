#!/usr/bin/env python3
'''
TabPFN Dynamic Breakout Classifier
Binary classification with validation-based dynamic threshold tuning.

Uses balanced metric (precision * sqrt(trade_rate)) to find optimal
prediction threshold that balances signal quality with trade frequency.
'''

from loop.manifest import Manifest
from loop.indicators import roc, wilder_rsi, rolling_volatility, bollinger_bands
from loop.features import bollinger_position
from loop.features.forward_breakout_target import (
    forward_breakout_target,
    compute_forward_breakout_threshold
)
from loop.transforms.logreg_transform import LogRegTransform
from loop.sfm.model.tabpfn_binary_dynamic import tabpfn_binary_dynamic


TRAIN_SPLIT = 50
VAL_SPLIT = 20
TEST_SPLIT = 30


def params() -> dict[str, list]:

    return {
        'forward_periods': [4, 8, 12],
        'threshold_pct': [0.01, 0.015, 0.02],
        'n_ensemble_configurations': [4],
        'device': ['cpu'],
        'use_calibration': [True],
        'threshold_metric': ['balanced'],
    }


def manifest() -> Manifest:

    return (Manifest()
        .set_split_config(TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)
        .set_required_bar_columns(['datetime', 'open', 'high', 'low', 'close', 'volume'])

        .add_indicator(roc, period=1)
        .add_indicator(roc, period=4)
        .add_indicator(roc, period=12)
        .add_indicator(roc, period=24)

        .add_indicator(rolling_volatility, column='close', window=4)
        .add_indicator(rolling_volatility, column='close', window=12)
        .add_indicator(rolling_volatility, column='close', window=24)

        .add_indicator(wilder_rsi, period=14)

        .add_indicator(bollinger_bands, price_col='close', window=20, num_std=2.0)

        .add_feature(bollinger_position)

        .with_target('forward_breakout')
            .add_fitted_transform(forward_breakout_target)
                .fit_param('_threshold', compute_forward_breakout_threshold,
                          forward_periods='forward_periods',
                          threshold_pct='threshold_pct')
                .with_params(
                    forward_periods='forward_periods',
                    threshold='_threshold',
                    shift=-1
                )
            .done()

        .set_scaler(LogRegTransform)

        .with_model(tabpfn_binary_dynamic)
    )

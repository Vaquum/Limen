#!/usr/bin/env python3
'''
TabPFN Dynamic Breakout Classifier
Binary classification with validation-based dynamic threshold tuning.

Uses balanced metric (precision * sqrt(trade_rate)) to find optimal
prediction threshold that balances signal quality with trade frequency.
'''

from limen.experiment import Manifest
from limen.data import HistoricalData
from limen.indicators import roc, wilder_rsi, rolling_volatility, bollinger_bands, bollinger_position
from limen.features.forward_breakout_target import forward_breakout_target
from limen.sfd.reference_architecture.tabpfn_binary import tabpfn_binary


TRAIN_SPLIT = 50
VAL_SPLIT = 20
TEST_SPLIT = 30


def params() -> dict[str, list]:

    return {
        # Target params
        'forward_periods': [2, 4, 6, 8, 12, 24],
        'threshold_pct': [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03],

        # Model params
        'n_ensemble_configurations': [4, 8],
        'device': ['cpu'],
        'use_calibration': [True, False],
        'threshold_metric': ['balanced', 'f1', 'precision'],

        # Indicator params
        'rsi_period': [7, 14, 21],
        'bb_window': [10, 20, 30],
        'bb_std': [1.5, 2.0, 2.5],
    }


def manifest() -> Manifest:

    return (Manifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'start_date_limit': '2025-01-01'}
        )
        .set_test_data_source(
            method=HistoricalData._get_data_for_test,
            params={'n_rows': 1000}
        )
        .set_split_config(TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)
        .set_required_bar_columns(['datetime', 'open', 'high', 'low', 'close', 'volume'])

        .add_indicator(roc, period=1)
        .add_indicator(roc, period=4)
        .add_indicator(roc, period=12)
        .add_indicator(roc, period=24)

        .add_indicator(rolling_volatility, column='close', window=4)
        .add_indicator(rolling_volatility, column='close', window=12)
        .add_indicator(rolling_volatility, column='close', window=24)

        .add_indicator(wilder_rsi, period='rsi_period')

        .add_indicator(bollinger_bands, price_col='close', window='bb_window', num_std='bb_std')

        .add_indicator(bollinger_position)

        .with_target('forward_breakout')
            .add_transform(forward_breakout_target,
                forward_periods='forward_periods',
                threshold='threshold_pct',
                shift=-1
            )
            .done()

        .with_model(tabpfn_binary)
    )

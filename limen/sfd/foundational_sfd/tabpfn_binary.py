from typing import Any, cast

from limen.calibration import CalibratorProtocol
from limen.calibration import ThresholdOptimizerProtocol
from limen.calibration import grid_threshold_optimizer
from limen.calibration import sklearn_probability_calibrator
from limen.data import HistoricalData
from limen.experiment import MLManifest
from limen.experiment import Manifest
from limen.indicators import bollinger_bands
from limen.indicators import bollinger_position
from limen.indicators import roc
from limen.indicators import rolling_volatility
from limen.indicators import wilder_rsi
from limen.metrics.balanced_metric import balanced_metric
from limen.sfd.reference_architecture.tabpfn_binary import tabpfn_binary
from limen.targets import ForwardBreakoutTarget


TRAIN_SPLIT = 50
VAL_SPLIT = 20
TEST_SPLIT = 30


def params() -> dict[str, list[Any]]:

    return {
        # Target params
        'forward_periods': [2, 4, 6, 8, 12, 24],
        'threshold_pct': [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03],

        # Model params
        'n_ensemble_configurations': [4, 8],
        'device': ['cpu'],

        # Indicator params
        'rsi_period': [7, 14, 21],
        'bb_window': [10, 20, 30],
        'bb_std': [1.5, 2.0, 2.5],
    }


def manifest() -> Manifest:

    base = (MLManifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'start_date_limit': '2025-01-01'}
        )
        .set_test_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 7200, 'row_count_limit': 1000}
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

        .with_target_label(
            'forward_breakout',
            ForwardBreakoutTarget,
            transform_params={'forward_periods': 'forward_periods', 'threshold': 'threshold_pct', 'shift': -1},
        )
    )

    return (cast(MLManifest, base)
        .set_strict_mode(True)
        .with_reference_architecture(tabpfn_binary)

        .with_calibration()
        .probability_calibration(func=cast(CalibratorProtocol, sklearn_probability_calibrator), method='isotonic')
        .threshold_function(func=cast(ThresholdOptimizerProtocol, grid_threshold_optimizer), metric=balanced_metric)
        .done()
    )

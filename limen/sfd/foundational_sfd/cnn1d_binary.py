from typing import cast

from limen.data import HistoricalData
from limen.experiment import Manifest, MLManifest
from limen.features.lagged_features import lag_range
from limen.indicators.window_return import window_return
from limen.sfd.reference_architecture.cnn1d_binary import cnn1d_binary
from limen.targets import NextBarUpTarget

__all__ = ['manifest', 'params']


def params() -> dict[str, list[object]]:
    return {'lookback_end': [23, 47], 'out_channels': [8, 16], 'conv_layers': [1, 2],
            'kernel_size': [3], 'stride': [1], 'dilation': [1], 'bias': [True],
            'activation': ['relu'], 'pooling': ['mean'], 'dropout': [0.0],
            'epochs': [10], 'batch_size': [64], 'learning_rate': [0.001, 0.01],
            'weight_decay': [0.0], 'betas': [(0.9, 0.999)], 'eps': [1e-8], 'random_state': [42]}


def manifest() -> Manifest:
    base = (MLManifest().set_data_source(HistoricalData.get_spot_klines,
                {'kline_size': 3600, 'start_date_limit': '2025-01-01', 'row_count_limit': 1024})
            .set_split_config(8, 1, 2).add_indicator(window_return, period=1)
            .add_feature(lag_range, col='ret_1', start=0, end='lookback_end')
            .with_target_label('next_close_up', NextBarUpTarget))
    return cast(MLManifest, base).set_strict_mode(True).with_reference_architecture(cnn1d_binary)

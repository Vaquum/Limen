import numpy as np
import polars as pl

from limen.data import HistoricalData
from limen.experiment import Manifest
from limen.sfd.reference_architecture import random_binary


def params():

    return {
        'random_weights': [0.4, 0.5, 0.6],
        'breakout_threshold': [0.05, 0.1, 0.2],
        'shift': [-1, -2, -3]
    }


def manifest():
    return (Manifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'start_date_limit': '2025-01-01'}
        )
        .set_test_data_source(method=HistoricalData._get_data_for_test)
        .set_split_config(3, 1, 1)
        .set_required_bar_columns([
            'datetime', 'high', 'low', 'close', 'volume', 'maker_ratio',
            'no_of_trades'
        ])
        .with_target('outcome')
            .add_transform(lambda data: data.with_columns(
                pl.Series('outcome', np.random.randint(0, 2, size=data.height))
            ))
            .add_transform(lambda data: data[:-1])
            .done()
        .with_model(random_binary)
    )

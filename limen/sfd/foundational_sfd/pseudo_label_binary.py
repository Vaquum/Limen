from datetime import date
from typing import cast

from limen.data import HistoricalData
from limen.experiment import MLManifest
from limen.features.lagged_features import lag_range
from limen.indicators.window_return import window_return
from limen.sfd.reference_architecture.pseudo_label_binary import pseudo_label_binary, recorded_rows
from limen.targets import NextBarUpTarget

__all__ = ['manifest', 'params']


def params() -> dict[str, list[object]]:
    return {'lookback_end': [23], 'pool_path': ['inputs/spot_1h_20240101_20241231.parquet'],
            'pool_sha256': ['a634bbd4f3692fac7bc88512da588e7772f4bd1b749567606d5314e54d0adf72'],
            'pool_start': ['2024-01-10T00:00:00+00:00'], 'pool_end': ['2024-01-15T00:00:00+00:00'],
            'training_cutoff': ['2024-01-15T00:00:00+00:00'], 'observation_interval_seconds': [3600],
            'confidence_threshold': [.7, .8], 'upstream_params': [{'C': 1000., 'max_iter': 500}, {'C': 2000., 'max_iter': 500}],
            'downstream_params': [{'C': .5}, {'C': 1.}]}


def manifest() -> MLManifest:
    base = (MLManifest().set_data_source(HistoricalData.get_spot_klines,
                {'kline_size': 3600, 'start_date_limit': '2024-01-01', 'end_date_limit': '2024-01-22'})
            .set_split_dates(*(date.fromisoformat(value) for value in
                ('2024-01-01', '2024-01-10', '2024-01-18', '2024-01-20', '2024-01-20', '2024-01-22')))
            .add_indicator(window_return, period=1).add_feature(lag_range, col='ret_1', start=0, end='lookback_end')
            .with_target_label('next_close_up', NextBarUpTarget))
    return (cast(MLManifest, base).add_to_data_dict(recorded_rows).set_strict_mode(True)
            .with_reference_architecture(pseudo_label_binary))

from typing import cast

from limen.data import HistoricalData
from limen.experiment import MLManifest
from limen.features.lagged_features import lag_range
from limen.indicators.window_return import window_return
from limen.sfd.reference_architecture.weight_link_binary import recorded_rows, weight_link_binary
from limen.targets import NextBarUpTarget

__all__ = ['manifest', 'params']



def params() -> dict[str, list[object]]:
    return {'lookback_end': [11, 23], 'upstream_params': [{'C': .5}, {'C': 1.}],
            'downstream_params': [{'C': 1.}], 'n_splits': [3],
            'weight_floor': [.1, .25], 'weight_power': [1.], 'observation_interval_seconds': [3600]}


def manifest() -> MLManifest:
    base = (MLManifest().set_data_source(HistoricalData.get_spot_klines,
                {'kline_size': 3600, 'start_date_limit': '2025-01-01', 'row_count_limit': 1024})
            .set_split_config(8, 1, 2).add_indicator(window_return, period=1)
            .add_feature(lag_range, col='ret_1', start=0, end='lookback_end')
            .with_target_label('next_close_up', NextBarUpTarget))
    return (cast(MLManifest, base).add_to_data_dict(recorded_rows).set_strict_mode(True)
            .with_reference_architecture(weight_link_binary))

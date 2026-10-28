from typing import cast

from limen.data import HistoricalData
from limen.experiment import Manifest, MLManifest
from limen.features.lagged_features import lag_range
from limen.indicators.window_return import window_return
from limen.sfd.reference_architecture.gaussian_hmm_binary import gaussian_hmm_binary
from limen.targets import NextBarUpTarget

__all__ = ['manifest', 'params']


def params() -> dict[str, list[object]]:
    return {'lookback_end': [11, 23], 'n_components': [2, 3], 'covariance_type': ['diag'],
            'min_covar': [1e-3], 'startprob_prior': [1.0], 'transmat_prior': [1.0, 2.0],
            'means_prior': [0.0], 'means_weight': [0.0], 'covars_prior': [0.01],
            'covars_weight': [1.0], 'random_state': [42], 'n_iter': [30], 'tol': [0.01],
            'verbose': [False], 'implementation': ['scaling', 'log']}


def manifest() -> Manifest:
    base = (MLManifest().set_data_source(HistoricalData.get_spot_klines,
                {'kline_size': 3600, 'start_date_limit': '2025-01-01', 'row_count_limit': 2048})
            .set_split_config(8, 1, 2).add_indicator(window_return, period=1)
            .add_feature(lag_range, col='ret_1', start=0, end='lookback_end')
            .with_target_label('next_close_up', NextBarUpTarget))
    return cast(MLManifest, base).set_strict_mode(True).with_reference_architecture(gaussian_hmm_binary)

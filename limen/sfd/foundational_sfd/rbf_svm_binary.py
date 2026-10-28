from inspect import signature

from limen.data import HistoricalData
from limen.experiment import Manifest
from limen.sfd.foundational_sfd.logreg_binary import manifest as _base_manifest, params as _base_params
from limen.sfd.reference_architecture.logreg_binary import logreg_binary
from limen.sfd.reference_architecture.rbf_svm_binary import rbf_svm_binary

__all__ = ['manifest', 'params']


def params() -> dict[str, list[object]]:
    preparation = {name: values for name, values in _base_params().items()
                   if name not in signature(logreg_binary).parameters}
    return preparation | {
        'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto', 0.1],
        'shrinking': [True, False], 'tol': [1e-3, 1e-2], 'cache_size': [200],
        'class_weight': [None, 'balanced'], 'verbose': [False],
        'max_iter': [-1], 'random_state': [42],
    }


def manifest() -> Manifest:
    return (_base_manifest()
            .set_data_source(HistoricalData.get_spot_klines,
                             {'kline_size': 3600, 'start_date_limit': '2025-01-01', 'row_count_limit': 2048})
            .set_test_data_source(HistoricalData.get_spot_klines, {'kline_size': 7200, 'row_count_limit': 512})
            .with_reference_architecture(rbf_svm_binary))

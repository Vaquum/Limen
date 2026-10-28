import json
import re
from typing import cast

import numpy as np
import numpy.typing as npt
import polars as pl
from typing_extensions import override

from limen.sfd.reference_architecture._chronological_link import digest as _digest, probabilities as _out_of_sample, row_carriers as _row_carriers
from limen.sfd.reference_architecture.base import ReferenceModel
from limen.sfd.reference_architecture.logreg_binary import LogRegBinary

__all__ = ['FeatureLinkedBinary', 'feature_link_binary', 'recorded_rows']
_FloatArray = npt.NDArray[np.float64]
_MATRIX_DIMENSIONS = 2


def recorded_rows(data_dict: dict[str, object], split_data: list[pl.DataFrame],
                  round_params: dict[str, object], fitted_params: dict[str, object]) -> dict[str, object]:
    return _row_carriers(data_dict, split_data, round_params, fitted_params)


class FeatureLinkedBinary(ReferenceModel):
    '''Chronological out-of-sample logistic probability appended to downstream inputs.'''
    deterministic, inference_output = False, ('binary', True)

    def __init__(self) -> None:
        super().__init__()
        self.upstream: LogRegBinary | None = None
        self.downstream: LogRegBinary | None = None
        self._columns: list[str] = []
        self._indices: list[int] = []
        self._width = 0
        self._provenance: dict[str, object] = {}

    def _features(self, inputs: object) -> _FloatArray:
        if isinstance(inputs, pl.DataFrame):
            values = inputs.select(self._columns).to_numpy()
        else:
            matrix = np.asarray(inputs, dtype=np.float64)
            if matrix.ndim != _MATRIX_DIMENSIONS or matrix.shape[1] != self._width:
                raise ValueError('FeatureLinkedBinary requires fitted feature order and width')
            values = matrix[:, self._indices]
        features = np.asarray(values, dtype=np.float64)
        if not np.isfinite(features).all():
            raise ValueError('FeatureLinkedBinary requires finite raw returns')
        return features

    @override
    def train(self, data: dict[str, object], *, upstream_params: dict[str, object] | None = None,
              downstream_params: dict[str, object] | None = None, n_splits: int = 3,
              observation_interval_seconds: int = 3600, **params: object) -> 'FeatureLinkedBinary':
        fitted = cast(dict[str, object], data.get('_fitted_params', {}))
        if params or any(data.get(key) is not None for key in ('_scaler', '_pca')) or any(not key.startswith('_target_cls_') for key in fitted):
            raise ValueError('FeatureLinkedBinary requires its declared controls and unscaled raw return windows')
        if type(observation_interval_seconds) is not int or observation_interval_seconds <= 0:
            raise ValueError('FeatureLinkedBinary requires a positive integer observation interval')
        configurations = [{'max_iter': 200, 'random_state': 42} | (values or {}) for values in (upstream_params, downstream_params)]
        if any(config.get('prediction_calibration_config') is not None for config in configurations):
            raise ValueError('FeatureLinkedBinary does not configure calibration')
        inputs, rows = data['x_train'], data.get('_linkage_rows')
        if not isinstance(inputs, pl.DataFrame) or not isinstance(rows, dict) or not isinstance(cast(dict[str, object], rows).get('train'), pl.DataFrame):
            raise ValueError('FeatureLinkedBinary requires named returns and recorded training row times')
        matches = sorted((int(match[1]), column) for column in inputs.columns
                         if (match := re.fullmatch(r'ret_1_lag_(\d+)', column)))
        if not matches or [lag for lag, _ in matches] != list(range(len(matches))):
            raise ValueError('FeatureLinkedBinary requires contiguous ret_1_lag_0..N columns')
        self._columns = [column for _, column in matches]
        self._indices, self._width = [inputs.columns.index(column) for column in self._columns], inputs.width
        features, truth = self._features(inputs), np.asarray(data['y_train'])
        times = np.asarray(cast(pl.DataFrame, rows['train'])['datetime'].dt.epoch('ns'), dtype=np.int64)
        interval = observation_interval_seconds * 1_000_000_000
        if len(times) != len(features) or truth.shape != (len(features),) or not np.array_equal(np.unique(truth), [0, 1]):
            raise ValueError('FeatureLinkedBinary requires aligned recorded rows and observed classes 0 and 1')
        if not np.all(np.diff(times) == interval):
            raise ValueError('FeatureLinkedBinary row times must follow the declared observation interval')
        self.upstream, self.downstream = None, None
        out_of_sample, folds = _out_of_sample(features, truth, times, interval, n_splits,
                                              configurations[0], type(self).__name__)
        covered = np.flatnonzero(np.isfinite(out_of_sample))
        augmented = np.column_stack((features[covered], out_of_sample[covered]))
        self.upstream = LogRegBinary().train({'x_train': features, 'y_train': truth}, **configurations[0])
        self.downstream = LogRegBinary().train({'x_train': augmented, 'y_train': truth[covered]}, **configurations[1])
        self._provenance = {'linkage_kind': 'feature', 'linkage_components': 'LogRegBinary -> LogRegBinary',
            'linkage_training_rows': len(truth), 'linkage_downstream_rows': len(covered),
            'linkage_columns': json.dumps([*self._columns, 'upstream_probability']), 'linkage_features_sha256': _digest(features),
            'linkage_observed_sha256': _digest(truth), 'linkage_row_times_sha256': _digest(times, '<i8'),
            'linkage_downstream_indices_sha256': _digest(covered, '<i8'),
            'linkage_oof_sha256': _digest(out_of_sample[covered]), 'linkage_augmented_sha256': _digest(augmented),
            'linkage_final_upstream_cutoff_ns': int(times[-1] + 2 * interval),
            'linkage_fold_cutoffs': json.dumps(folds, separators=(',', ':'))}
        return self

    def _augmented(self, inputs: object) -> _FloatArray:
        if self.upstream is None:
            raise ValueError('FeatureLinkedBinary is not fitted')
        features = self._features(inputs)
        probability = np.asarray(self.upstream.predict({'x_test': features})['_probs'], dtype=np.float64)
        return np.column_stack((features, probability))

    @override
    def predict(self, data: dict[str, object]) -> dict[str, object]:
        if self.downstream is None:
            raise ValueError('FeatureLinkedBinary is not fitted')
        return cast(dict[str, object], self.downstream.predict({'x_test': self._augmented(data['x_test'])}))

    @override
    def evaluate(self, data: dict[str, object], inline_metrics: bool = True) -> dict[str, object]:
        if self.downstream is None:
            raise ValueError('FeatureLinkedBinary is not fitted')
        return cast(dict[str, object], self.downstream.evaluate(data | {'x_test': self._augmented(data['x_test'])}, inline_metrics)) | self._provenance


def feature_link_binary(data: dict[str, object], upstream_params: dict[str, object] | None = None,
                       downstream_params: dict[str, object] | None = None, n_splits: int = 3,
                                observation_interval_seconds: int = 3600) -> dict[str, object]:
    model = FeatureLinkedBinary().train(data, upstream_params=upstream_params, downstream_params=downstream_params,
        n_splits=n_splits,
        observation_interval_seconds=observation_interval_seconds)
    return model.evaluate(data) | {'_model': model}

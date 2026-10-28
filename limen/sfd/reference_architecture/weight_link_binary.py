import hashlib
import json
import re
from collections.abc import Iterator
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.model_selection import TimeSeriesSplit
from typing_extensions import override

from limen.sfd.reference_architecture.base import ReferenceModel
from limen.sfd.reference_architecture.logreg_binary import LogRegBinary

__all__ = ['WeightLinkedBinary', 'recorded_rows', 'weight_link_binary']
_FloatArray = npt.NDArray[np.float64]
_IndexArray = npt.NDArray[np.int64]
_MATRIX_DIMENSIONS = 2


class _ChronologicalSplits(Protocol):
    def split(self, inputs: _FloatArray) -> Iterator[tuple[_IndexArray, _IndexArray]]: ...


def recorded_rows(data_dict: dict[str, object], split_data: list[pl.DataFrame],
                  round_params: dict[str, object], fitted_params: dict[str, object]) -> dict[str, object]:
    del round_params, fitted_params
    return data_dict | {'_linkage_rows': {name: frame.select('datetime')
                        for name, frame in zip(('train', 'val', 'test'), split_data, strict=True)}}


def _digest(value: object, dtype: str = '<f8') -> str:
    array = np.asarray(value, dtype=dtype)
    header = json.dumps([array.shape, array.dtype.str], separators=(',', ':')).encode()
    return hashlib.sha256(header + array.tobytes(order='C')).hexdigest()


class WeightLinkedBinary(ReferenceModel):
    '''Observed out-of-sample logistic difficulty consumed by downstream row-weight fitting.'''
    deterministic, inference_output = False, ('binary', True)

    def __init__(self) -> None:
        super().__init__()
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
                raise ValueError('WeightLinkedBinary requires fitted feature order and width')
            values = matrix[:, self._indices]
        features = np.asarray(values, dtype=np.float64)
        if not np.isfinite(features).all():
            raise ValueError('WeightLinkedBinary requires finite raw returns')
        return features

    @override
    def train(self, data: dict[str, object], *, upstream_params: dict[str, object] | None = None,
              downstream_params: dict[str, object] | None = None, n_splits: int = 3,
              weight_floor: float = .1, weight_power: float = 1.,
              observation_interval_seconds: int = 3600, **params: object) -> 'WeightLinkedBinary':
        if params or any(data.get(key) is not None for key in ('_scaler', '_pca')):
            raise ValueError('WeightLinkedBinary requires its declared controls and unscaled raw return windows')
        if not 0 < weight_floor <= 1 or not np.isfinite(weight_power) or weight_power <= 0:
            raise ValueError('WeightLinkedBinary requires 0 < weight_floor <= 1 and finite positive weight_power')
        if type(observation_interval_seconds) is not int or observation_interval_seconds <= 0:
            raise ValueError('WeightLinkedBinary requires a positive integer observation interval')
        configurations = [{'max_iter': 200, 'random_state': 42} | (values or {}) for values in (upstream_params, downstream_params)]
        if any(config.get('prediction_calibration_config') is not None for config in configurations):
            raise ValueError('WeightLinkedBinary does not configure calibration')
        inputs, rows = data['x_train'], data.get('_linkage_rows')
        if not isinstance(inputs, pl.DataFrame) or not isinstance(rows, dict) or not isinstance(cast(dict[str, object], rows).get('train'), pl.DataFrame):
            raise ValueError('WeightLinkedBinary requires named returns and recorded training row times')
        matches = sorted((int(match[1]), column) for column in inputs.columns
                         if (match := re.fullmatch(r'ret_1_lag_(\d+)', column)))
        if not matches or [lag for lag, _ in matches] != list(range(len(matches))):
            raise ValueError('WeightLinkedBinary requires contiguous ret_1_lag_0..N columns')
        self._columns = [column for _, column in matches]
        self._indices, self._width = [inputs.columns.index(column) for column in self._columns], inputs.width
        features, truth = self._features(inputs), np.asarray(data['y_train'])
        times = np.asarray(cast(pl.DataFrame, rows['train'])['datetime'].dt.epoch('ns'), dtype=np.int64)
        interval = observation_interval_seconds * 1_000_000_000
        if len(times) != len(features) or truth.shape != (len(features),) or not np.array_equal(np.unique(truth), [0, 1]):
            raise ValueError('WeightLinkedBinary requires aligned recorded rows and observed classes 0 and 1')
        if not np.all(np.diff(times) == interval):
            raise ValueError('WeightLinkedBinary row times must follow the declared observation interval')
        self.downstream = None
        out_of_sample = np.full(len(truth), np.nan, dtype=np.float64)
        folds: list[dict[str, int]] = []
        for prefix, predict in cast(_ChronologicalSplits, TimeSeriesSplit(n_splits=n_splits)).split(features):
            cutoff = times[predict[0]] + interval
            train = prefix[times[prefix] + 2 * interval <= cutoff]
            if not np.array_equal(np.unique(truth[train]), [0, 1]):
                raise ValueError('WeightLinkedBinary requires both observed classes in each available fit prefix')
            upstream = LogRegBinary().train({'x_train': features[train], 'y_train': truth[train]}, **configurations[0])
            probability = np.asarray(upstream.predict({'x_test': features[predict]})['_probs'], dtype=np.float64)
            if probability.shape != predict.shape or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
                raise ValueError('WeightLinkedBinary requires aligned finite upstream probabilities in [0, 1]')
            out_of_sample[predict] = probability
            folds.append({'fit_rows': len(train), 'fit_label_available_at_ns': int((times[train] + 2 * interval).max()),
                          'prediction_available_at_ns': int(cutoff), 'prediction_first_row': int(predict[0]), 'prediction_rows': len(predict)})
        covered = np.flatnonzero(np.isfinite(out_of_sample))
        weights = weight_floor + (1 - weight_floor) * np.abs(truth[covered] - out_of_sample[covered]) ** weight_power
        self.downstream = LogRegBinary().train({'x_train': features[covered], 'y_train': truth[covered], '_sample_weight': weights}, **configurations[1])
        self._provenance = {'linkage_kind': 'weight', 'linkage_components': 'LogRegBinary -> LogRegBinary',
            'linkage_training_rows': len(truth), 'linkage_downstream_rows': len(covered),
            'linkage_columns': json.dumps(self._columns), 'linkage_features_sha256': _digest(features),
            'linkage_observed_sha256': _digest(truth), 'linkage_row_times_sha256': _digest(times, '<i8'),
            'linkage_downstream_indices_sha256': _digest(covered, '<i8'),
            'linkage_oof_sha256': _digest(out_of_sample[covered]), 'linkage_weights_sha256': _digest(weights),
            'linkage_fold_cutoffs': json.dumps(folds, separators=(',', ':'))}
        return self

    @override
    def predict(self, data: dict[str, object]) -> dict[str, object]:
        if self.downstream is None:
            raise ValueError('WeightLinkedBinary is not fitted')
        return cast(dict[str, object], self.downstream.predict({'x_test': self._features(data['x_test'])}))

    @override
    def evaluate(self, data: dict[str, object], inline_metrics: bool = True) -> dict[str, object]:
        if self.downstream is None:
            raise ValueError('WeightLinkedBinary is not fitted')
        return cast(dict[str, object], self.downstream.evaluate(data | {'x_test': self._features(data['x_test'])}, inline_metrics)) | self._provenance


def weight_link_binary(data: dict[str, object], upstream_params: dict[str, object] | None = None,
                       downstream_params: dict[str, object] | None = None, n_splits: int = 3,
                       weight_floor: float = .1, weight_power: float = 1.,
                       observation_interval_seconds: int = 3600) -> dict[str, object]:
    model = WeightLinkedBinary().train(data, upstream_params=upstream_params, downstream_params=downstream_params,
        n_splits=n_splits, weight_floor=weight_floor, weight_power=weight_power,
        observation_interval_seconds=observation_interval_seconds)
    return model.evaluate(data) | {'_model': model}

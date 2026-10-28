from datetime import timedelta
import hashlib
import json
from pathlib import Path
import re
from typing import cast

import numpy as np
import numpy.typing as npt
import polars as pl
from typing_extensions import override

from limen.sfd.reference_architecture._pseudo_pool import load_pool
from limen.sfd.reference_architecture.base import ReferenceModel
from limen.sfd.reference_architecture.logreg_binary import LogRegBinary
from limen.targets.next_bar_up import NextBarUpTarget

__all__ = ['PseudoLabelBinary', 'pseudo_label_binary', 'recorded_rows']
_FloatArray = npt.NDArray[np.float64]
_MATRIX_DIMENSIONS, _THRESHOLD = 2, .5


def recorded_rows(data_dict: dict[str, object], split_data: list[pl.DataFrame],
                  round_params: dict[str, object], fitted_params: dict[str, object]) -> dict[str, object]:
    del round_params, fitted_params
    if '_holdout_carriers' in data_dict:
        raise ValueError('PseudoLabelBinary does not admit external pools in protected holdout research')
    return data_dict | {'_linkage_rows': {name: frame.select('datetime')
                        for name, frame in zip(('train', 'val', 'test'), split_data, strict=True)}}


def _digest(value: object, dtype: str = '<f8') -> str:
    array = np.asarray(value, dtype=dtype)
    header = json.dumps([array.shape, array.dtype.str], separators=(',', ':')).encode()
    return hashlib.sha256(header + array.tobytes(order='C')).hexdigest()


class PseudoLabelBinary(ReferenceModel):
    '''Explicit generated training labels for a disjoint pool of recorded observations.'''
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
                raise ValueError('PseudoLabelBinary requires fitted feature order and width')
            values = matrix[:, self._indices]
        features = np.asarray(values, dtype=np.float64)
        if not np.isfinite(features).all():
            raise ValueError('PseudoLabelBinary requires finite raw return windows')
        return features

    @override
    def train(self, data: dict[str, object], *, pool_path: str, pool_sha256: str,
              pool_start: str, pool_end: str, training_cutoff: str,
              observation_interval_seconds: int, confidence_threshold: float = .8,
              upstream_params: dict[str, object] | None = None,
              downstream_params: dict[str, object] | None = None, **params: object) -> 'PseudoLabelBinary':
        fitted = cast(dict[str, object], data.get('_fitted_params', {}))
        if params or 'holdout' in data or any(not key.startswith('_target_cls_') for key in fitted) or any(data.get(key) is not None for key in ('_scaler', '_pca')):
            raise ValueError('PseudoLabelBinary requires unscaled raw windows outside protected holdout research')
        targets = [value for key, value in fitted.items() if key.startswith('_target_cls_')]
        if len(targets) != 1 or type(targets[0]) is not NextBarUpTarget:
            raise ValueError('PseudoLabelBinary requires fitted NextBarUpTarget labels with one-bar availability')
        if type(observation_interval_seconds) is not int or observation_interval_seconds <= 0:
            raise ValueError('PseudoLabelBinary requires a positive integer observation interval')
        if not _THRESHOLD < confidence_threshold <= 1:
            raise ValueError('PseudoLabelBinary confidence_threshold must be above 0.5 and at most 1')
        configurations = [{'max_iter': 200, 'random_state': 42} | (values or {}) for values in (upstream_params, downstream_params)]
        if any(config.get('prediction_calibration_config') is not None for config in configurations):
            raise ValueError('PseudoLabelBinary does not configure calibration')
        inputs, carriers = data['x_train'], data.get('_linkage_rows')
        if not isinstance(inputs, pl.DataFrame) or not isinstance(carriers, dict):
            raise ValueError('PseudoLabelBinary requires named raw returns and recorded split row times')
        rows = cast(dict[str, pl.DataFrame], carriers)
        if any(not isinstance(rows.get(name), pl.DataFrame) for name in ('train', 'val', 'test')):
            raise ValueError('PseudoLabelBinary requires all recorded split row carriers')
        if any(not isinstance(frame.schema.get('datetime'), pl.Datetime) or cast(pl.Datetime, frame.schema['datetime']).time_zone is None for frame in rows.values()):
            raise ValueError('PseudoLabelBinary requires timezone-aware recorded split row times')
        matches = sorted((int(match[1]), column) for column in inputs.columns
                         if (match := re.fullmatch(r'ret_1_lag_(\d+)', column)))
        if not matches or [lag for lag, _ in matches] != list(range(len(matches))):
            raise ValueError('PseudoLabelBinary requires contiguous ret_1_lag_0..N columns')
        self._columns = [column for _, column in matches]
        self._indices, self._width = [inputs.columns.index(column) for column in self._columns], inputs.width
        features, truth = self._features(inputs), np.asarray(data['y_train'])
        times = np.asarray(rows['train']['datetime'].dt.epoch('ns'), dtype=np.int64)
        interval = observation_interval_seconds * 1_000_000_000
        if len(times) != len(features) or truth.shape != (len(features),) or not np.array_equal(np.unique(truth), [0, 1]):
            raise ValueError('PseudoLabelBinary requires aligned observed training classes 0 and 1')
        if not np.all(np.diff(times) == interval):
            raise ValueError('PseudoLabelBinary training rows must follow the declared interval')
        pool, cutoff = load_pool(pool_path, pool_sha256, pool_start, pool_end, training_cutoff,
                                             observation_interval_seconds, self._columns, rows)
        upstream = LogRegBinary().train({'x_train': features, 'y_train': truth}, **configurations[0])
        pool_features = self._features(pool)
        probability = np.asarray(upstream.predict({'x_test': pool_features})['_probs'], dtype=np.float64)
        confidence = np.maximum(probability, 1 - probability)
        selected = confidence >= confidence_threshold
        if not selected.any():
            raise ValueError('PseudoLabelBinary selected no pool rows at the declared confidence threshold')
        generated = (probability[selected] > _THRESHOLD).astype(np.int8)
        augmented, labels = np.vstack((features, pool_features[selected])), np.concatenate((truth, generated))
        self.downstream = LogRegBinary().train({'x_train': augmented, 'y_train': labels}, **configurations[1])
        step = timedelta(seconds=observation_interval_seconds)
        records = [{'datetime': time.isoformat(), 'available_at': (time + step).isoformat(),
                    'label': int(label), 'probability': float(prob), 'confidence': float(certainty), 'origin': 'pseudo'}
                   for time, label, prob, certainty in zip(pool.filter(pl.Series(selected))['datetime'].to_list(),
                       generated, probability[selected], confidence[selected], strict=True)]
        payload = json.dumps(records, separators=(',', ':'))
        self._provenance = {'linkage_kind': 'pseudo_label', 'linkage_components': 'LogRegBinary -> LogRegBinary',
            'linkage_observed_rows': len(truth), 'linkage_pool_rows': pool.height, 'linkage_pseudo_rows': len(generated),
            'linkage_downstream_rows': len(labels), 'linkage_pool_path': str(Path(pool_path).resolve()),
            'linkage_pool_sha256': pool_sha256, 'linkage_columns': json.dumps(self._columns),
            'linkage_observed_features_sha256': _digest(features), 'linkage_observed_labels_sha256': _digest(truth),
            'linkage_observed_times_sha256': _digest(times, '<i8'), 'linkage_downstream_features_sha256': _digest(augmented),
            'linkage_downstream_labels_sha256': _digest(labels), 'linkage_upstream_params': json.dumps(configurations[0], sort_keys=True),
            'linkage_upstream_fit_cutoff_ns': int(times[-1] + 2 * interval), 'linkage_training_cutoff': cutoff.isoformat(),
            'linkage_selected_rows': payload, 'linkage_selected_sha256': hashlib.sha256(payload.encode()).hexdigest()}
        return self

    @override
    def predict(self, data: dict[str, object]) -> dict[str, object]:
        if self.downstream is None:
            raise ValueError('PseudoLabelBinary is not fitted')
        return cast(dict[str, object], self.downstream.predict({'x_test': self._features(data['x_test'])}))

    @override
    def evaluate(self, data: dict[str, object], inline_metrics: bool = True) -> dict[str, object]:
        if self.downstream is None:
            raise ValueError('PseudoLabelBinary is not fitted')
        return cast(dict[str, object], self.downstream.evaluate(data | {'x_test': self._features(data['x_test'])}, inline_metrics)) | self._provenance


def pseudo_label_binary(data: dict[str, object], pool_path: str, pool_sha256: str,
                        pool_start: str, pool_end: str, training_cutoff: str,
                        observation_interval_seconds: int, confidence_threshold: float = .8,
                        upstream_params: dict[str, object] | None = None,
                        downstream_params: dict[str, object] | None = None) -> dict[str, object]:
    model = PseudoLabelBinary().train(data, pool_path=pool_path, pool_sha256=pool_sha256,
        pool_start=pool_start, pool_end=pool_end, training_cutoff=training_cutoff,
        observation_interval_seconds=observation_interval_seconds, confidence_threshold=confidence_threshold,
        upstream_params=upstream_params, downstream_params=downstream_params)
    return model.evaluate(data) | {'_model': model}

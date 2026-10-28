import hashlib
import json
from collections.abc import Iterator
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.model_selection import TimeSeriesSplit

from limen.sfd.reference_architecture.logreg_binary import LogRegBinary

__all__: list[str] = []
_FloatArray = npt.NDArray[np.float64]
_IndexArray = npt.NDArray[np.int64]


class _ChronologicalSplits(Protocol):
    def split(self, inputs: _FloatArray) -> Iterator[tuple[_IndexArray, _IndexArray]]: ...


def row_carriers(data_dict: dict[str, object], split_data: list[pl.DataFrame],
                 round_params: dict[str, object], fitted_params: dict[str, object]) -> dict[str, object]:
    del round_params, fitted_params
    return data_dict | {'_linkage_rows': {name: frame.select('datetime')
                        for name, frame in zip(('train', 'val', 'test'), split_data, strict=True)}}


def digest(value: object, dtype: str = '<f8') -> str:
    array = np.asarray(value, dtype=dtype)
    header = json.dumps([array.shape, array.dtype.str], separators=(',', ':')).encode()
    return hashlib.sha256(header + array.tobytes(order='C')).hexdigest()


def probabilities(features: _FloatArray, truth: npt.NDArray[np.generic], times: _IndexArray,
                  interval: int, n_splits: int, upstream_params: dict[str, object], name: str
                  ) -> tuple[_FloatArray, list[dict[str, int]]]:
    out_of_sample = np.full(len(truth), np.nan, dtype=np.float64)
    folds: list[dict[str, int]] = []
    for prefix, predict in cast(_ChronologicalSplits, TimeSeriesSplit(n_splits=n_splits)).split(features):
        cutoff = times[predict[0]] + interval
        train = prefix[times[prefix] + 2 * interval <= cutoff]
        if not np.array_equal(np.unique(truth[train]), [0, 1]):
            raise ValueError(name + ' requires both observed classes in each available fit prefix')
        upstream = LogRegBinary().train({'x_train': features[train], 'y_train': truth[train]}, **upstream_params)
        probability = np.asarray(upstream.predict({'x_test': features[predict]})['_probs'], dtype=np.float64)
        if probability.shape != predict.shape or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError(name + ' requires aligned finite upstream probabilities in [0, 1]')
        out_of_sample[predict] = probability
        folds.append({'fit_rows': len(train), 'fit_label_available_at_ns': int((times[train] + 2 * interval).max()),
                      'prediction_available_at_ns': int(cutoff), 'prediction_first_row': int(predict[0]), 'prediction_rows': len(predict)})
    return out_of_sample, folds

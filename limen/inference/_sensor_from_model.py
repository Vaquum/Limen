from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Literal, TypeAlias, cast
from uuid import uuid4

import numpy as np
import numpy.typing as npt
import polars as pl

from limen.inference._python_sensor import HistoryT, Infer, MapOutput, OutputT, PredictionReason, StateT, decision_rows, output_rows, utc_cutoff

__all__ = ['Missing', 'ModelInput', 'ModelPairs', 'Predict', 'Prepare', 'identity_pairs', 'model_infer']

ModelPairs: TypeAlias = Sequence[tuple[int | float | None, float | None]]
Prepare: TypeAlias = Callable[[HistoryT, Mapping[str, object]], pl.DataFrame]
Predict: TypeAlias = Callable[[StateT, pl.DataFrame, Mapping[str, object]], OutputT]
Missing: TypeAlias = Literal['block', 'model']
ModelInput: TypeAlias = Literal['auto', 'frame', 'array']
_METADATA = ('datetime', 'reason', 'available_at_ns')


def _vector(value: object) -> Sequence[object]:
    if isinstance(value, np.ndarray):
        array = cast(npt.NDArray[np.generic], value)
        if array.ndim != 1:
            raise ValueError('Python Sensor output must have exactly one dimension')
        values = cast(Sequence[object], array.tolist())
    elif isinstance(value, pl.Series):
        values = cast(Sequence[object], value.to_list())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = cast(Sequence[object], value)
    else:
        raise ValueError('Python Sensor output must be a one-dimensional sequence')
    if any(isinstance(item, (np.ndarray, pl.Series, Sequence)) and not isinstance(item, (str, bytes)) for item in values):
        raise ValueError('Python Sensor output must have exactly one dimension')
    return values


def _features(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError('Python Sensor feature_columns must be a sequence of strings')
    if not isinstance(value, (Sequence, np.ndarray, pl.Series)):
        raise TypeError('Python Sensor feature_columns must be a sequence of strings')
    names = _vector(cast(object, value))
    if any(not isinstance(name, str) for name in names):
        raise TypeError('Python Sensor feature_columns must contain strings')
    columns = cast(tuple[str, ...], tuple(names))
    if not columns or any(not name.strip() for name in columns) or len(set(columns)) != len(columns):
        raise ValueError('Python Sensor feature_columns must be nonempty, unique and nonblank')
    return columns


def _default_pairs(output: object, count: int) -> Sequence[object]:
    values = _vector(output)
    if len(values) != count:
        raise ValueError('Python Sensor prediction count must match eligible feature rows')
    return [((1 if value else 0) if isinstance(value, (bool, np.bool_)) else value, None) for value in values]


def _feature_reasons(frame: pl.DataFrame, missing: Missing) -> list[PredictionReason | None]:
    if missing == 'model':
        return [None] * frame.height
    invalid = frame.select(pl.any_horizontal([
        pl.col(name).is_null() | (~pl.col(name).is_finite() if dtype in (pl.Float32(), pl.Float64()) else pl.lit(False))
        for name, dtype in frame.schema.items()
    ])).to_series().to_list()
    first_valid = next((i for i, blocked in enumerate(invalid) if not blocked), frame.height)
    return [('warm-up' if i < first_valid else 'null-features') if blocked else None for i, blocked in enumerate(invalid)]


def identity_pairs(output: ModelPairs) -> ModelPairs:
    return output


def model_infer(model: StateT, selection_cutoff: datetime, prepare: Prepare[HistoryT] | None,
                feature_columns: Sequence[str] | None, predict: Predict[StateT, OutputT] | None,
                map_output: MapOutput[OutputT] | None, state_id: str | None,
                missing: Missing, model_input: ModelInput) -> tuple[Infer[HistoryT, StateT, ModelPairs], str]:
    cutoff = utc_cutoff(selection_cutoff)
    if prepare is not None and not callable(prepare):
        raise TypeError('Python Sensor prepare must be callable or None')
    if predict is not None and not callable(predict):
        raise TypeError('Python Sensor predict must be callable or None')
    if map_output is not None and not callable(map_output):
        raise TypeError('Python Sensor map_output must be callable or None')
    if missing not in ('block', 'model'):
        raise ValueError('Python Sensor missing must be block or model')
    if model_input not in ('auto', 'frame', 'array'):
        raise ValueError('Python Sensor model_input must be auto, frame or array')
    declared: object = getattr(model, 'feature_names_in_', None)
    recorded = None if declared is None else _features(declared)
    columns = _features(feature_columns) if feature_columns is not None else recorded
    method: object = getattr(model, 'predict', None)
    if predict is None and not callable(method):
        raise TypeError('Python Sensor fitted model must expose callable predict')
    default_predict = cast(Callable[[object], object], method)

    def infer(history: HistoryT, fitted: StateT, params: Mapping[str, object]) -> tuple[pl.DataFrame, ModelPairs]:
        prepared = history if prepare is None else prepare(history, params)
        if not isinstance(prepared, pl.DataFrame):
            raise TypeError('Python Sensor preparation must return a Polars DataFrame')
        timestamps, declared_reasons, _availability = decision_rows(prepared)
        selected = _features([name for name in prepared.columns if name not in _METADATA]) if columns is None else columns
        if any(name not in prepared.columns for name in selected):
            raise ValueError('Python Sensor selected feature columns are absent from prepared data')
        features = prepared.select(selected)
        feature_reasons = _feature_reasons(features, missing)
        reasons = [declared or feature for declared, feature in zip(declared_reasons, feature_reasons, strict=True)]
        metadata = prepared.select([name for name in _METADATA if name in prepared.columns]).with_columns(pl.Series('reason', reasons, dtype=pl.String))
        indices = [i for i, (stamp, reason) in enumerate(zip(timestamps, reasons, strict=True)) if stamp >= cutoff and reason is None]
        pairs: list[object] = [(None, None)] * prepared.height
        if indices:
            eligible = features[indices]
            if predict is not None:
                output: object = predict(fitted, eligible, params)
            else:
                model_frame = model_input == 'frame' or (model_input == 'auto' and recorded is not None)
                output = default_predict(eligible if model_frame else eligible.to_numpy())
            mapped = _default_pairs(output, len(indices)) if map_output is None else output_rows(map_output(cast(OutputT, output)), len(indices))
            for index, pair in zip(indices, mapped, strict=True):
                pairs[index] = pair
        return metadata, cast(ModelPairs, pairs)

    return infer, str(uuid4()) if state_id is None else state_id

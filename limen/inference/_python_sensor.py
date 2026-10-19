from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Generic, Literal, TypeVar, cast

import polars as pl

__all__ = ['BarPrediction', 'HistoryT', 'Infer', 'MapOutput', 'OutputT', 'PredictionReason', 'PythonSensorBase', 'StateT', 'utc_cutoff', 'decision_rows', 'scalar', 'mapped_values', 'output_rows']

PredictionReason = Literal['warm-up', 'inside-training-window', 'null-features', 'sensor-error']
HistoryT = TypeVar('HistoryT')
StateT = TypeVar('StateT')
OutputT = TypeVar('OutputT')
Infer = Callable[[HistoryT, StateT, Mapping[str, object]], tuple[pl.DataFrame, OutputT]]
MapOutput = Callable[[OutputT], Sequence[tuple[int | float | None, float | None]]]
_PAIR_SIZE = 2


@dataclass
class BarPrediction:

    '''Prediction result for a single bar.'''

    datetime: object
    prediction: int | float | None
    probability: float | None
    reason: PredictionReason | None
    available_at_ns: int | None = None
    trade_contract_digest: str | None = None


def _construction_params(params: object, infer: object, map_output: object, state_id: object, output_id: object) -> Mapping[str, object]:
    if not isinstance(params, Mapping):
        raise TypeError('Python Sensor params must be a string-keyed mapping')
    source = cast(Mapping[object, object], params)
    if any(not isinstance(key, str) for key in source):
        raise TypeError('Python Sensor params must have string keys')
    if not callable(infer) or not callable(map_output):
        raise TypeError('Python Sensor infer and map_output must be callable')
    if not isinstance(state_id, str):
        raise TypeError('Python Sensor state_id must be a string')
    if not state_id.strip():
        raise ValueError('Python Sensor state_id must not be empty')
    if output_id is not None:
        if not isinstance(output_id, str):
            raise TypeError('Python Sensor output_id must be a string or None')
        if not output_id.strip():
            raise ValueError('Python Sensor output_id must not be empty')
    return MappingProxyType(cast(dict[str, object], dict(source)))


def utc_cutoff(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError('Python Sensor cutoffs must be datetimes')
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('Python Sensor cutoffs must be timezone-aware')
    return value.astimezone(timezone.utc)


def decision_rows(metadata: object) -> tuple[list[datetime], list[PredictionReason | None], list[int | None]]:
    if not isinstance(metadata, pl.DataFrame) or 'datetime' not in metadata.columns:
        raise ValueError('Python Sensor metadata requires a datetime column')
    timestamps = metadata['datetime']
    dtype = timestamps.dtype
    if not isinstance(dtype, pl.Datetime) or dtype.time_zone != 'UTC' or dtype.time_unit not in ('ms', 'us', 'ns'):
        raise ValueError('Python Sensor datetime must use UTC Datetime metadata')
    if timestamps.null_count() or not timestamps.is_sorted() or timestamps.n_unique() != metadata.height:
        raise ValueError('Python Sensor datetimes must be non-null, unique and increasing')
    epoch_ns = cast(list[int], timestamps.dt.epoch('ns').to_list())
    if dtype.time_unit == 'ns' and any(value % 1000 for value in epoch_ns):
        raise ValueError('Python Sensor datetime cannot lose nanosecond precision')
    reasons: list[PredictionReason | None] = [None] * metadata.height
    if 'reason' in metadata.columns:
        if metadata['reason'].dtype not in (pl.String(), pl.Null()):
            raise ValueError('Python Sensor reason must use String or Null metadata')
        raw_reasons = cast(list[object], metadata['reason'].to_list())
        if any(reason not in (None, 'warm-up', 'null-features') for reason in raw_reasons):
            raise ValueError('Python Sensor reason may declare only warm-up or null-features')
        reasons = cast(list[PredictionReason | None], raw_reasons)
    availability: list[int | None] = [None] * metadata.height
    if 'available_at_ns' in metadata.columns:
        if metadata['available_at_ns'].dtype not in (pl.Int64(), pl.Null()):
            raise ValueError('Python Sensor availability must use Int64 or Null metadata')
        availability = cast(list[int | None], metadata['available_at_ns'].to_list())
        if any(value is not None and value < stamp for value, stamp in zip(availability, epoch_ns, strict=True)):
            raise ValueError('Python Sensor availability cannot precede its decision timestamp')
    return cast(list[datetime], timestamps.to_list()), reasons, availability


def output_rows(mapped: object, count: int) -> Sequence[object]:
    if not isinstance(mapped, Sequence) or isinstance(mapped, (str, bytes)):
        raise ValueError('Python Sensor map_output must return a sequence of pairs')
    rows = cast(Sequence[object], mapped)
    if len(rows) != count:
        raise ValueError('Python Sensor mapped row count must match decision metadata')
    return rows


def scalar(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('Python Sensor mapped values must be real numeric scalars')
    if isinstance(value, Integral):
        return int(value)
    number = float(value)
    if not isfinite(number):
        raise ValueError('Python Sensor mapped values must be finite')
    return number


def mapped_values(row: object) -> tuple[int | float | None, float | None, PredictionReason | None]:
    if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
        return None, None, 'sensor-error'
    pair = cast(Sequence[object], row)
    if len(pair) != _PAIR_SIZE:
        return None, None, 'sensor-error'
    try:
        prediction = scalar(pair[0])
        probability = None if pair[1] is None else float(scalar(pair[1]))
        if probability is not None and not 0 <= probability <= 1:
            raise ValueError('Python Sensor probability must lie in [0, 1]')
    except (ValueError, OverflowError):
        return None, None, 'sensor-error'
    return prediction, probability, None


@dataclass(frozen=True, eq=False)
class PythonSensorBase(Generic[HistoryT, StateT, OutputT]):

    fitted_state: StateT = field(repr=False)
    params: Mapping[str, object] = field(repr=False)
    infer: Infer[HistoryT, StateT, OutputT] = field(repr=False)
    map_output: MapOutput[OutputT] = field(repr=False)
    state_id: str
    fitting_cutoff: datetime
    selection_cutoff: datetime
    output_id: str | None = None
    permutation_id: str | None = field(default=None, init=False)
    manifest_id: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        params = _construction_params(self.params, self.infer, self.map_output, self.state_id, self.output_id)
        fitting = utc_cutoff(self.fitting_cutoff)
        selection = utc_cutoff(self.selection_cutoff)
        if fitting > selection:
            raise ValueError('Python Sensor fitting_cutoff must not exceed selection_cutoff')
        object.__setattr__(self, 'params', params)
        object.__setattr__(self, 'fitting_cutoff', fitting)
        object.__setattr__(self, 'selection_cutoff', selection)

    @property
    def origin(self) -> Literal['python']:
        return 'python'

    @property
    def prediction_mode(self) -> Literal['python']:
        return 'python'

    @property
    def round_params(self) -> dict[str, object]:
        return dict(self.params)

    @property
    def trade_contract(self) -> None:
        return None

    def __call__(self, raw_klines: HistoryT) -> list[BarPrediction]:
        return self.predict_all(raw_klines)

    def predict(self, raw_klines: HistoryT) -> BarPrediction:
        results = self.predict_all(raw_klines)
        return results[-1] if results else BarPrediction(None, None, None, 'warm-up')

    def predict_all(self, raw_klines: HistoryT) -> list[BarPrediction]:
        metadata, output = self.infer(raw_klines, self.fitted_state, self.params)
        timestamps, reasons, availability = decision_rows(metadata)
        rows = output_rows(self.map_output(output), len(timestamps))
        results: list[BarPrediction] = []
        for stamp, declared_reason, available, row in zip(timestamps, reasons, availability, rows, strict=True):
            if stamp < self.selection_cutoff:
                prediction, probability, reason = None, None, 'inside-training-window'
            elif declared_reason is not None:
                prediction, probability, reason = None, None, declared_reason
            else:
                prediction, probability, reason = mapped_values(row)
            results.append(BarPrediction(stamp, prediction, probability, reason, available_at_ns=available))
        return results

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from statistics import mean
from typing import Generic, Literal, cast
from uuid import uuid4

import polars as pl

from limen.inference.sensor import Sensor
from limen.inference._python_sensor import BarPrediction, HistoryT, PredictionReason, PythonSensorBase, mapped_values, utc_cutoff

__all__ = ['Aggregate', 'CohortMap', 'CohortPrepare', 'HistoryT', 'PythonCohortBase', 'cohort_fields']

Pair = tuple[int | float | None, float | None]
Aggregate = Literal['mean', 'vote'] | Callable[[Sequence[BarPrediction]], Pair]
CohortMap = Callable[[Sensor, BarPrediction], Pair]
CohortPrepare = Callable[[HistoryT, Sensor], object]
_PRIORITY: dict[PredictionReason, int] = {'warm-up': 0, 'null-features': 1, 'sensor-error': 2, 'inside-training-window': 3}


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f'Python Cohort {name} must be a string')
    if not value.strip():
        raise ValueError(f'Python Cohort {name} must not be blank')
    return value


def cohort_fields(sensors: object, aggregate: object, prepare: object, map_output: object,
                  mapped_output_id: object, state_id: object, selection_cutoff: object,
                  ) -> tuple[tuple[Sensor, ...], str, datetime]:
    if not isinstance(sensors, Sequence) or isinstance(sensors, (str, bytes)):
        raise TypeError('Python Cohort sensors must be a sequence of native Sensors')
    supplied = cast(Sequence[object], sensors)
    if any(not isinstance(member, Sensor) for member in supplied):
        raise TypeError('Python Cohort members must be native Sensors')
    members = tuple(cast(Sequence[Sensor], supplied))
    if not members:
        raise ValueError('Python Cohort sensors must not be empty')
    if isinstance(aggregate, str):
        if aggregate not in ('mean', 'vote'):
            raise ValueError('Python Cohort aggregate must be mean, vote or callable')
    elif not callable(aggregate):
        raise TypeError('Python Cohort aggregate must be mean, vote or callable')
    if any(callback is not None and not callable(callback) for callback in (prepare, map_output)):
        raise TypeError('Python Cohort preparation and mapping must be callable')
    if map_output is None and mapped_output_id is not None:
        raise ValueError('Python Cohort mapped_output_id requires map_output')
    if map_output is not None:
        if mapped_output_id is None:
            raise ValueError('Python Cohort mapping requires mapped_output_id')
        _ = _identifier(mapped_output_id, 'mapped_output_id')
    identity = str(uuid4()) if state_id is None else _identifier(state_id, 'state_id')
    cutoff = None if selection_cutoff is None else utc_cutoff(selection_cutoff)
    cutoffs: list[datetime] = [] if cutoff is None else [cutoff]
    meanings: list[str | None] = []
    objects: set[int] = set()
    identities: set[tuple[str, ...]] = set()
    for member in members:
        if id(member) in objects:
            raise ValueError('Python Cohort repeats a Sensor object')
        objects.add(id(member))
        key: tuple[str, ...] | None = None
        if isinstance(member, PythonSensorBase):
            key = ('python', member.state_id)
            cutoffs.append(member.selection_cutoff)
            meanings.append(member.output_id)
        else:
            if map_output is None or cutoff is None:
                raise ValueError('Compiled members require mapping, mapped_output_id and selection_cutoff')
            if member.manifest_id is not None and member.permutation_id is not None:
                key = ('compiled', member.manifest_id, member.permutation_id)
        if key is not None:
            if key in identities:
                raise ValueError(f'Python Cohort repeats member identity {key}')
            identities.add(key)
    if map_output is None and (any(value is None for value in meanings) or len(set(meanings)) != 1):
        raise ValueError('Python Cohort requires matching declared Sensor output_id values')
    return members, identity, max(cutoffs)


def _member_label(index: int, member: Sensor) -> str:
    identity = member.state_id if isinstance(member, PythonSensorBase) else f'{member.manifest_id}:{member.permutation_id}'
    return f'Python Cohort member {index} ({identity})'


def _snapshot(rows: object, label: str) -> tuple[BarPrediction, ...]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError(f'{label}: predictions must be a sequence')
    result: list[BarPrediction] = []
    previous: datetime | None = None
    for raw in cast(Sequence[object], rows):
        if not isinstance(raw, BarPrediction):
            raise ValueError(f'{label}: predictions must be BarPrediction values')
        stamp = raw.datetime
        if not isinstance(stamp, datetime) or stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
            raise ValueError(f'{label}: decision timestamps must be non-null UTC datetimes')
        if previous is not None and stamp <= previous:
            raise ValueError(f'{label}: decision timestamps must be unique and increasing')
        previous = stamp
        reason = raw.reason
        if reason is not None and reason not in ('warm-up', 'null-features', 'sensor-error', 'inside-training-window'):
            raise ValueError(f'{label}: unknown prediction reason')
        available = raw.available_at_ns
        if available is not None:
            if type(available) is not int:
                raise ValueError(f'{label}: availability must be integer nanoseconds or None')
            epoch = datetime(1970, 1, 1, tzinfo=stamp.tzinfo)
            elapsed = stamp - epoch
            stamp_ns = ((elapsed.days * 86400 + elapsed.seconds) * 1_000_000 + elapsed.microseconds) * 1000
            if available < stamp_ns:
                raise ValueError(f'{label}: availability cannot precede its decision timestamp')
        if reason is None:
            prediction, probability, reason = mapped_values((raw.prediction, raw.probability))
        else:
            prediction, probability = None, None
        result.append(BarPrediction(stamp, prediction, probability, reason, available, raw.trade_contract_digest))
    return tuple(result)


def _member_predictions(member: Sensor, prepared: object, label: str) -> tuple[BarPrediction, ...]:
    if isinstance(member, PythonSensorBase):
        predict = cast(Callable[[object], list[BarPrediction]], member.predict_all)
        return _snapshot(predict(prepared), label)
    if not isinstance(prepared, pl.DataFrame):
        raise TypeError(f'{label}: compiled Sensor history must be a Polars DataFrame')
    return _snapshot(member.predict_all(prepared), label)


def _builtin(aggregate: Literal['mean', 'vote'], bars: Sequence[BarPrediction]) -> Pair:
    predictions = [cast(int | float, bar.prediction) for bar in bars]
    if aggregate == 'vote':
        if any(value not in (0, 1) for value in predictions):
            return None, None
        return int(sum(predictions) > len(predictions) / 2), None
    if len(bars) == 1:
        return bars[0].prediction, bars[0].probability
    probability = mean([cast(float, bar.probability) for bar in bars]) if all(bar.probability is not None for bar in bars) else None
    return mean(predictions), probability


@dataclass(frozen=True, eq=False)
class PythonCohortBase(Generic[HistoryT]):

    members: tuple[Sensor, ...]
    state_id: str
    selection_cutoff: datetime
    _aggregate: Aggregate = field(default='mean', repr=False)
    _prepare: CohortPrepare[HistoryT] | None = field(default=None, repr=False)
    _map_output: CohortMap | None = field(default=None, repr=False)
    manifest_id: str | None = field(default=None, init=False)
    cohort_id: str | None = field(default=None, init=False)

    @property
    def origin(self) -> Literal['python']:
        return 'python'

    @property
    def prediction_mode(self) -> Literal['python']:
        return 'python'

    def set_members(self, members: object) -> None:
        raise TypeError('Python Cohort membership is fixed; construct a new Cohort.from_python instead.')

    def __call__(self, raw_klines: HistoryT) -> list[BarPrediction]:
        return self.predict_all(raw_klines)

    def predict(self, raw_klines: HistoryT) -> BarPrediction:
        rows = self.predict_all(raw_klines)
        return rows[-1] if rows else BarPrediction(None, None, None, 'warm-up')

    def predict_all(self, raw_klines: HistoryT) -> list[BarPrediction]:
        batches: list[tuple[BarPrediction, ...]] = []
        for index, member in enumerate(self.members):
            prepared = raw_klines if self._prepare is None else self._prepare(raw_klines, member)
            rows = _member_predictions(member, prepared, _member_label(index, member))
            if batches and [bar.datetime for bar in rows] != [bar.datetime for bar in batches[0]]:
                raise ValueError(f'{_member_label(index, member)}: decision timestamps/count do not align')
            batches.append(rows)
        return [self._aggregate_row(tuple(bars)) for bars in zip(*batches, strict=True)]

    def _aggregate_row(self, bars: tuple[BarPrediction, ...]) -> BarPrediction:
        stamp = cast(datetime, bars[0].datetime)
        known = [bar.available_at_ns for bar in bars]
        available = max(cast(list[int], known)) if all(value is not None for value in known) else None
        reasons = [bar.reason for bar in bars if bar.reason is not None]
        reason: PredictionReason | None = max(reasons, key=_PRIORITY.__getitem__) if reasons else None
        if stamp < self.selection_cutoff:
            reason = 'inside-training-window'
        if reason is not None:
            return BarPrediction(stamp, None, None, reason, available)
        mapped: list[BarPrediction] = []
        for member, bar in zip(self.members, bars, strict=True):
            pair = (bar.prediction, bar.probability) if self._map_output is None else self._map_output(member, replace(bar))
            prediction, probability, invalid = mapped_values(pair)
            if invalid is not None:
                return BarPrediction(stamp, None, None, invalid, available)
            mapped.append(BarPrediction(stamp, prediction, probability, None, bar.available_at_ns))
        pair = self._aggregate(mapped) if callable(self._aggregate) else _builtin(self._aggregate, mapped)
        prediction, probability, reason = mapped_values(pair)
        return BarPrediction(stamp, prediction, probability, reason, available)

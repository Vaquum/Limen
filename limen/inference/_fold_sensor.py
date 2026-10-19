from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from datetime import date, datetime, timezone
from functools import wraps
from typing import Protocol, TypeVar, cast

import polars as pl

from limen.experiment._fold_fingerprint import interval
from limen.inference._python_sensor import BarPrediction, decision_rows

__all__: list[str] = []


class LiveSensor(Protocol):
    def predict_all(self, raw_klines: pl.DataFrame) -> list[BarPrediction]: ...


_S = TypeVar('_S', bound=LiveSensor)


def single_guard(func: Callable[[_S, pl.DataFrame], BarPrediction]) -> Callable[[_S, pl.DataFrame], BarPrediction]:
    @wraps(func)
    def predict(sensor: _S, frame: pl.DataFrame) -> BarPrediction:
        if hasattr(sensor, '_fold_context'):
            rows = sensor.predict_all(frame)
            return rows[-1] if rows else BarPrediction(None, None, None, 'warm-up')
        return func(sensor, frame)
    return predict


def reject_manual(reference: Mapping[str, object], sensor: object) -> None:
    sfd = cast(Mapping[str, object], reference.get('sfd', {}))
    manifest = cast(Mapping[str, object], sfd.get('manifest', {}))
    if manifest.get('split_walk_forward') is not None and not hasattr(sensor, '_fold_context'):
        raise ValueError('Walk-forward Sensor requires validated Trainer.train(..., fold=k) evidence')


def reject_legacy_members(members: Sequence[object], *, qualified_folds: bool = False) -> None:
    if any(getattr(member, 'origin', None) == 'python' for member in members):
        raise ValueError('Cohort cannot bind Python-origin Sensors; use Cohort.from_python.')
    if not qualified_folds and any(getattr(member, 'fold', None) is not None for member in members):
        raise ValueError('Cohort requires fold-qualified member selection for fold Sensors')


def last_inside(sensor: object, dt: object, manifest: object) -> bool:
    context = cast(Mapping[str, object] | None, getattr(sensor, '_fold_context', None))
    if context is not None:
        return isinstance(dt, datetime) and (int(dt.timestamp()) * 1_000_000 + dt.microsecond) * 1000 < cast(int, cast(Mapping[str, object], context['campaign'])['selection_cutoff_ns'])
    dates = cast(tuple[date, date, date, date, date, date] | None, getattr(manifest, 'split_dates', None))
    if dates is None or dt is None:
        return False
    train_start, _, val_start, val_end, test_start, test_end = dates
    dt_date = dt.date() if isinstance(dt, datetime) else cast(date, dt)
    masked = train_start <= dt_date < test_end
    if masked and getattr(manifest, 'val_predict_guard', None) is False and val_start <= dt_date < val_end:
        masked = False
    if masked and getattr(manifest, 'test_predict_guard', None) is False and test_start <= dt_date < test_end:
        masked = False
    return masked


def inside_mask(sensor: object, frame: pl.DataFrame, manifest: object) -> list[bool]:
    return [last_inside(sensor, dt, manifest) for dt in frame['datetime'].to_list()] if 'datetime' in frame.columns else [False] * len(frame)


def begin(sensor: object, manifest: object, frame: pl.DataFrame) -> None:
    context = cast(Mapping[str, object] | None, getattr(sensor, '_fold_context', None))
    if context is None:
        return
    times, _, _ = decision_rows(frame)
    campaign = cast(Mapping[str, object], context['campaign'])
    source = cast(Mapping[str, object], campaign['source'])
    step, origin = cast(int, source['interval_ns']), cast(int, source['first_ns'])
    if any((value - origin) % step for value in cast(list[int], frame['datetime'].dt.epoch('ns').to_list())):
        raise ValueError('Fold Sensor input clock conflicts with the recorded native grid')
    if 'base_interval' in frame.columns and interval(frame, step / 1_000_000_000) != step:
        raise ValueError('Fold Sensor base_interval changed')
    trial = cast(Mapping[str, object], context['trial'])
    prep = cast(Mapping[str, object], trial['preparation'])
    formed = prep['formation'] is not None and prep['bar_type'] != 'base'
    vars(sensor)['_fold_clock'] = None if formed else times
    def capture(prepared: pl.DataFrame) -> None:
        stamps, _, _ = decision_rows(prepared)
        vars(sensor)['_fold_clock'] = stamps
    vars(manifest)['_fold_decision_clock'] = capture


def recover(sensor: object, manifest: object, error: Exception, fallback: int) -> list[BarPrediction]:
    if not hasattr(sensor, '_fold_context'):
        return [BarPrediction(None, None, None, 'sensor-error') for _ in range(fallback)]
    times = cast(list[datetime] | None, getattr(sensor, '_fold_clock', None))
    if times is None:
        raise ValueError('Fold Sensor formation failed before its decision clock could be established') from error
    return [BarPrediction(dt, None, None, 'inside-training-window' if last_inside(sensor, dt, manifest) else 'sensor-error') for dt in times]


class FoldView:
    _fold_context: dict[str, object]

    @property
    def fold(self) -> int:
        return cast(int, self._fold_context['fold'])

    @property
    def n_folds(self) -> int:
        return cast(int, cast(Mapping[str, object], self._fold_context['fold_witness'])['n_folds'])

    @property
    def provenance_digest(self) -> str:
        return cast(str, self._fold_context['provenance_digest'])

    @property
    def fold_provenance(self) -> dict[str, object]:
        return copy.deepcopy(self._fold_context)

    @property
    def fitting_cutoff(self) -> datetime:
        value = cast(int, cast(Mapping[str, object], self._fold_context['fold_witness'])['fitting_cutoff_ns'])
        return datetime.fromtimestamp(value // 1_000_000_000, timezone.utc).replace(microsecond=(value // 1000) % 1_000_000)

    @property
    def selection_cutoff(self) -> datetime:
        value = cast(int, cast(Mapping[str, object], self._fold_context['campaign'])['selection_cutoff_ns'])
        return datetime.fromtimestamp(value // 1_000_000_000, timezone.utc).replace(microsecond=(value // 1000) % 1_000_000)

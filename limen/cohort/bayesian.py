from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from math import fsum, isfinite
from typing import cast

import polars as pl

from limen.cohort.cohort import Cohort
from limen.cohort._python_cohort import CohortPrepare, HistoryT, PythonCohortBase
from limen.inference._python_sensor import BarPrediction, PythonSensorBase
from limen.inference.sensor import Sensor

__all__ = ['bayesian_cohort']


def bayesian_cohort(sensors: Sequence[Sensor], evidence: pl.DataFrame, *,
                    alpha: float = 1.0, beta: float = 1.0,
                    prepare: CohortPrepare[HistoryT] | None = None,
                    state_id: str | None = None,
                    selection_cutoff: datetime | None = None) -> PythonCohortBase[HistoryT]:
    '''Bind observed binary correctness evidence to a fixed, reliability-weighted Cohort.'''
    if any(not isinstance(member, PythonSensorBase) for member in sensors):
        raise ValueError('Bayesian Cohort requires live Python Sensors')
    baseline = Cohort.from_python(sensors, aggregate='vote', prepare=prepare,
                                  state_id=state_id, selection_cutoff=selection_cutoff)
    members = baseline.members
    if any(isinstance(value, bool) or not isfinite(value) or value <= 0 for value in (alpha, beta)) or not isfinite(alpha + beta):
        raise ValueError('Bayesian Cohort priors must be positive and finite, including their sum')
    required = {'sensor_id', 'datetime', 'available_at', 'prediction', 'outcome', 'source'}
    if not required.issubset(evidence.columns):
        raise ValueError(f'Bayesian evidence requires columns {sorted(required)}')
    admitted = evidence.select(sorted(required))
    if any(admitted.null_count().row(0)):
        raise ValueError('Bayesian evidence must not contain null values')
    for name in ('sensor_id', 'source'):
        if admitted[name].dtype != pl.String or admitted[name].str.strip_chars().eq('').any():
            raise ValueError(f'Bayesian evidence {name} must contain nonblank strings')
    for name in ('datetime', 'available_at'):
        dtype = admitted[name].dtype
        if not isinstance(dtype, pl.Datetime) or dtype.time_zone != 'UTC':
            raise ValueError(f'Bayesian evidence {name} must use UTC Datetime values')
        if (admitted[name].dt.epoch('ns') % 1000 != 0).any():
            raise ValueError('Bayesian evidence timestamps cannot lose nanosecond precision')
    for name in ('prediction', 'outcome'):
        if not admitted[name].dtype.is_numeric() or not admitted[name].is_in([0, 1]).all():
            raise ValueError(f'Bayesian evidence {name} must contain numeric binary values')
    if admitted.select('sensor_id', 'datetime').n_unique() != admitted.height:
        raise ValueError('Bayesian evidence repeats a Sensor decision')
    if admitted.select((pl.col('available_at') <= pl.col('datetime')).any()).item():
        raise ValueError('Bayesian evidence outcomes must become available after their prediction decision')
    identities = [cast(PythonSensorBase[object, object, object], member).state_id for member in members]
    if set(admitted['sensor_id'].to_list()) - set(identities):
        raise ValueError('Bayesian evidence names an unknown Sensor')
    weights: list[float] = []
    cutoffs = [baseline.selection_cutoff]
    for identity, member in zip(identities, members, strict=True):
        rows = admitted.filter(pl.col('sensor_id') == identity)
        cutoff = cast(PythonSensorBase[object, object, object], member).selection_cutoff
        if rows.filter(pl.col('datetime') < cutoff).height:
            raise ValueError(f'Bayesian evidence for {identity} is inside its Sensor selection history')
        correct = rows.select((pl.col('prediction') == pl.col('outcome')).sum()).item()
        weights.append((alpha + correct) / (alpha + beta + rows.height))
        if rows.height:
            cutoffs.append(cast(datetime, rows['available_at'].max()))
    frozen_weights = tuple(weights)

    def aggregate(bars: Sequence[BarPrediction]) -> tuple[int, None]:
        if any(bar.prediction not in (0, 1) for bar in bars):
            raise ValueError('Bayesian Cohort predictions must be binary')
        positive = fsum(weight for weight, bar in zip(frozen_weights, bars, strict=True) if bar.prediction == 1)
        return int(positive > fsum(frozen_weights) / 2), None

    return Cohort.from_python(members, aggregate=aggregate, prepare=prepare,
                              state_id=baseline.state_id, selection_cutoff=max(cutoffs))

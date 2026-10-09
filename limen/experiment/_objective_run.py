from collections.abc import Mapping, MutableMapping, Sequence
import csv
import math
from numbers import Real
from pathlib import Path

import polars as pl

from limen.experiment._objective import ObjectiveConfig
from limen.experiment.reducer.budget_reducer import BudgetReducer
from limen.experiment.reducer.correlation_reducer import CorrelationReducer
from limen.experiment.reducer.focus_reducer import FocusReducer
from limen.experiment.reducer.pruning_strategy import PruningStrategy
from limen.experiment.reducer.sanity_reducer import SanityReducer
from limen.experiment.reducer.saturation_reducer import SaturationReducer


def run_objective(manifest: object) -> ObjectiveConfig | None:
    objective = getattr(manifest, 'objective', None)
    if objective is not None and not isinstance(objective, ObjectiveConfig):
        raise TypeError('Manifest objective must be an ObjectiveConfig or None')
    return objective


def validate_objective_reducers(manifest: object, reducers: Sequence[PruningStrategy]) -> None:
    objective = run_objective(manifest)
    if objective is not None:
        for reducer in reducers:
            metric_driven = isinstance(reducer, (CorrelationReducer, FocusReducer, SanityReducer, SaturationReducer))
            if isinstance(reducer, BudgetReducer):
                metric_driven = getattr(reducer, '_trim_strategy') == 'worst_first'
            if metric_driven:
                if getattr(reducer, '_metric') != objective.column:
                    raise ValueError(f'{type(reducer).__name__} metric must be {objective.column} for this objective')
                if isinstance(reducer, (CorrelationReducer, FocusReducer, BudgetReducer)) and getattr(reducer, '_maximize') != objective.maximize:
                    raise ValueError(f'{type(reducer).__name__} maximize conflicts with objective direction')


def finalize_objective_result(manifest: object, result: MutableMapping[str, object], succeeded: bool) -> None:
    objective = run_objective(manifest)
    if objective is not None:
        score = result.get(objective.column)
        if succeeded:
            if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(float(score)):
                raise ValueError(f'Successful objective round requires finite {objective.column}')
            result[objective.column] = float(score)
        else:
            result[objective.column] = None


def objective_frame(manifest: object, rows: Sequence[Mapping[str, object]]) -> pl.DataFrame:
    objective = run_objective(manifest)
    overrides: dict[str, type[pl.DataType]] = {} if objective is None else {objective.column: pl.Float64}
    return pl.DataFrame([dict(row) for row in rows], schema_overrides=overrides)


def validate_objective_header(manifest: object, header: Sequence[str] | None) -> None:
    objective = run_objective(manifest)
    if objective is not None and objective.column not in (header or ()):
        raise ValueError(f'Cannot append or resume objective results without {objective.column}')


def add_objective_metadata(manifest: object, metadata: MutableMapping[str, object]) -> None:
    objective = run_objective(manifest)
    if objective is not None:
        metadata['objective'] = objective.as_dict()


def validate_objective_resume(manifest: object, metadata: Mapping[str, object], csv_path: Path) -> None:
    objective = run_objective(manifest)
    declaration = None if objective is None else objective.as_dict()
    if metadata.get('objective') != declaration:
        raise ValueError('Cannot resume with a different objective declaration')
    if objective is not None:
        if not csv_path.exists():
            raise ValueError('Cannot resume objective run without results.csv')
        with csv_path.open(newline='') as stream:
            validate_objective_header(manifest, next(csv.reader(stream), None))


__all__ = ['add_objective_metadata', 'finalize_objective_result', 'objective_frame', 'run_objective',
           'validate_objective_header', 'validate_objective_reducers', 'validate_objective_resume']

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

import polars as pl

from limen.experiment._objective_run import finalize_objective_result
import limen.experiment._holdout_composition as _protection
from limen.experiment._objective_run import objective_frame
from limen.experiment._objective_run import run_objective
from limen.experiment._objective_run import validate_objective_header
from limen.experiment._parallel_worker import RoundPayload
from limen.experiment._result_fields import validate_result
from limen.experiment._walk_forward_run import WalkForwardRun
from limen.experiment.manifest_core import Manifest

__all__ = ['ParallelCommit', 'context_parameter_keys', 'failed_parameters']


class _AppendRoundData(Protocol):
    def __call__(self, path: Path, identity: str, parameters: dict[str, object],
                 predictions: object, alignment: dict[str, object],
                 *, round_index: int) -> None: ...


def context_parameter_keys(metadata: Mapping[str, object]) -> list[str]:
    keys = metadata.get('context_parameter_keys')
    if not isinstance(keys, list) or any(not isinstance(key, str) or key.startswith('_') for key in cast(list[object], keys)) or len(set(cast(list[str], keys))) != len(cast(list[str], keys)):
        raise ValueError('Cannot resume parallel run without valid context parameter keys')
    return cast(list[str], keys)


def failed_parameters(checkpoint: Mapping[str, object],
                      rows: Sequence[Mapping[str, str | None]]) -> list[dict[str, object]]:
    expected = [(index, row['id']) for index, row in enumerate(rows) if row['strict_mode_error']]
    records = checkpoint.get('failed_round_params', [])
    if not isinstance(records, list) or len(cast(list[object], records)) != len(expected):
        raise ValueError('Cannot resume parallel run without typed failed-round parameter evidence')
    failed: list[dict[str, object]] = []
    for value, (index, identity) in zip(cast(list[object], records), expected, strict=True):
        if not isinstance(value, dict):
            raise ValueError('Cannot resume parallel run with malformed failed-round parameter evidence')
        entry = cast(dict[str, object], value)
        if type(entry.get('_round_index')) is not int or entry['_round_index'] != index or entry.get('round_id') != identity:
            raise ValueError('Parallel checkpoint parameter evidence differs from failed CSV occurrences')
        failed.append(entry)
    return failed


class ParallelCommit:
    def __init__(self, host: object, csv_path: Path,
                 round_data_path: Path | None,
                 failed_round_params: list[dict[str, object]]) -> None:
        super().__init__()
        self.host, self.csv_path, self.round_data_path = host, csv_path, round_data_path
        self.failed_round_params = failed_round_params
        attributes = cast(dict[str, object], vars(host))
        self.manifest = cast(Manifest | None, attributes['manifest'])
        self.params = cast(dict[str, object], attributes['params'])
        self.walk_forward = cast(WalkForwardRun | None, attributes['_walk_forward'])
        append_round_data = getattr(host, '_append_round_data', None)
        if not callable(append_round_data):
            raise TypeError('Parallel host requires the round-data writer')
        self.append_round_data = cast(_AppendRoundData, append_round_data)
        self._has_success = False
        if csv_path.exists():
            with csv_path.open(newline='') as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames:
                    raise ValueError('Parallel results.csv has no header')
                self.header = list(reader.fieldnames)
                self._has_success = any(not row.get('strict_mode_error') for row in reader)
            validate_objective_header(self.manifest, self.header)
        else:
            self.header = list(dict.fromkeys([
                *self.params, 'id', '_round_index', 'strict_mode_error',
                'execution_time', '_warnings', '_convergence_warning',
            ]))
            if getattr(self.manifest, 'ablation_config', None) is not None:
                self.header.append('_dropped_features')
            objective = run_objective(self.manifest)
            if objective is not None:
                self.header.append(objective.column)
            with csv_path.open('w', newline='') as stream:
                csv.writer(stream).writerow(self.header)
            if round_data_path is not None:
                round_data_path.touch(exist_ok=False)

    def _extend_header(self, row: dict[str, object]) -> None:
        extra = [key for key in row if key not in self.header]
        if extra:
            if self._has_success:
                raise ValueError(f'Parallel result columns changed after the first success: {extra}')
            with self.csv_path.open(newline='') as stream:
                previous = list(csv.DictReader(stream))
            self.header.extend(extra)
            with self.csv_path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, self.header)
                writer.writeheader()
                writer.writerows(previous)

    def commit(self, round_params: dict[str, object], payload: RoundPayload) -> None:
        index, identity = round_params['_round_index'], round_params['_id']
        if not isinstance(index, int) or isinstance(index, bool) or not isinstance(identity, str):
            raise ValueError('Parallel occurrence requires an integer index and string identity')
        _protection.verify_worker(self.manifest, cast(pl.DataFrame, vars(self.host)['data']), payload.alignment, identity)
        parameters = round_params | {key: value for key, value in payload.parameters.items() if not key.startswith('_')}
        validate_result(parameters, payload.results, succeeded=payload.succeeded)
        row = dict(payload.results)
        predictions = row.pop('_preds', [])
        for key in ('models', '_model', 'extras'):
            _ = row.pop(key, None)
        row.update(parameters)
        row['id'] = identity
        row['execution_time'] = payload.elapsed
        row['_warnings'] = json.dumps(payload.warnings)
        row['_convergence_warning'] = payload.convergence
        _ = row.setdefault('strict_mode_error', None)
        if not payload.succeeded and not row['strict_mode_error']:
            row['strict_mode_error'] = 'StrictModeError'
        if getattr(self.manifest, 'ablation_config', None) is not None:
            row['_dropped_features'] = json.dumps(payload.parameters.get('_dropped_features', []))
        finalize_objective_result(self.manifest, row, payload.succeeded)
        self._extend_header(row)
        if payload.succeeded and self.walk_forward is not None:
            self.walk_forward.accept(identity, {'_alignment': payload.alignment})
        with self.csv_path.open('a', newline='') as stream:
            csv.writer(stream).writerow(['' if row.get(key) is None else row[key] for key in self.header])
        if payload.succeeded and self.round_data_path is not None:
            self.append_round_data(self.round_data_path, identity, payload.parameters,
                                   predictions, payload.alignment, round_index=index)
        batch = objective_frame(self.manifest, [row])
        attributes = cast(dict[str, object], vars(self.host))
        existing = cast(pl.DataFrame | None, attributes['experiment_log'])
        attributes['experiment_log'] = batch if existing is None else pl.concat([existing, batch], how='diagonal_relaxed')
        if not payload.succeeded:
            self.failed_round_params.append({'_round_index': index, 'round_id': identity,
                                             'round_params': dict(payload.parameters)})
        self._has_success = self._has_success or payload.succeeded

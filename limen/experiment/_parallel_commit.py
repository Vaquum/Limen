from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Protocol, cast

import polars as pl

from limen.experiment._objective_run import finalize_objective_result
from limen.experiment._objective_run import objective_frame
from limen.experiment._objective_run import run_objective
from limen.experiment._objective_run import validate_objective_header
from limen.experiment._parallel_worker import RoundPayload
from limen.experiment._walk_forward_run import WalkForwardRun
from limen.experiment.manifest_core import Manifest

__all__ = ['ParallelCommit']


class _AppendRoundData(Protocol):
    def __call__(self, path: Path, identity: str, parameters: dict[str, object],
                 predictions: object, alignment: dict[str, object],
                 *, round_index: int) -> None: ...


class ParallelCommit:
    def __init__(self, host: object, csv_path: Path,
                 round_data_path: Path | None) -> None:
        super().__init__()
        self.host, self.csv_path, self.round_data_path = host, csv_path, round_data_path
        self.manifest = cast(Manifest | None, getattr(host, 'manifest'))
        self.params = cast(dict[str, object], getattr(host, 'params'))
        self.walk_forward = cast(WalkForwardRun | None, getattr(host, '_walk_forward'))
        self.append_round_data = cast(_AppendRoundData, getattr(host, '_append_round_data'))
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
        row = dict(payload.results)
        predictions = row.pop('_preds', [])
        for key in ('models', '_model', 'extras'):
            _ = row.pop(key, None)
        row.update(round_params)
        row.update({key: value for key, value in payload.parameters.items() if not key.startswith('_')})
        row['id'] = identity
        row['execution_time'] = round(payload.elapsed, 2)
        row['_warnings'] = json.dumps(payload.warnings)
        row['_convergence_warning'] = payload.convergence
        _ = row.setdefault('strict_mode_error', None)
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
        existing = cast(pl.DataFrame | None, getattr(self.host, 'experiment_log'))
        setattr(self.host, 'experiment_log', batch if existing is None else pl.concat([existing, batch], how='diagonal_relaxed'))
        self._has_success = self._has_success or payload.succeeded

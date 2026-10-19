from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sqlite3
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn import metrics as _metrics

from limen.experiment import _holdout_history as history
from limen.experiment._holdout_freeze import encoded

roc_auc_score = cast(Callable[[Sequence[int], Sequence[float]], float], vars(_metrics)['roc_auc_score'])
_BINARY_CLASSES = 2

class Metric(TypedDict):
    value: float | None
    reason: str | None
    numerator: float | None
    denominator: int | None
    support_rows: int


def metric(value: float | None, support: int, *, reason: str | None = None,
           numerator: float | None = None, denominator: int | None = None) -> Metric:
    return {'value': value, 'reason': reason, 'numerator': numerator, 'denominator': denominator, 'support_rows': support}


def classification(labels: list[int | None], predictions: list[int], probabilities: list[float | None]) -> dict[str, Metric]:
    valid = [(label, pred) for label, pred in zip(labels, predictions, strict=True) if label is not None]
    n = len(valid)
    tp = sum(label == pred == 1 for label, pred in valid)
    populations = {'accuracy': (sum(label == pred for label, pred in valid), n),
                   'precision': (tp, sum(pred == 1 for _, pred in valid)),
                   'recall': (tp, sum(label == 1 for label, _ in valid)),
                   'fpr': (sum(label == 0 and pred == 1 for label, pred in valid), sum(label == 0 for label, _ in valid))}
    result = {key: metric(num / den if den else None, n, reason=None if den else 'zero-denominator' if n else 'no-labels', numerator=float(num), denominator=den) for key, (num, den) in populations.items()}
    pairs = [(label, prob) for label, prob in zip(labels, probabilities, strict=True) if label is not None and prob is not None]
    reason = 'no-labels' if not n else 'no-probability' if not pairs else 'one-class' if len({label for label, _ in pairs}) < _BINARY_CLASSES else None
    result['auc'] = metric(float(roc_auc_score([label for label, _ in pairs], [prob for _, prob in pairs])) if reason is None else None, len(pairs), reason=reason)
    return result


def return_metrics(values: Mapping[str, float], net: npt.NDArray[np.float64], pos: npt.NDArray[np.float64]) -> dict[str, Metric]:
    result: dict[str, Metric] = {}
    for key, value in values.items():
        support = int((net > 0).sum()) if key == 'avg_win_bps' else int((net < 0).sum()) if key == 'avg_loss_bps' else len(net)
        numerator = float((net > 0).sum()) if key == 'wins_per_bar' else float(((pos > 0) & ~np.concatenate(([False], pos[:-1] > 0))).sum()) if key == 'trades_per_bar' else None
        result[key] = metric(float(value) if math.isfinite(value) else None, support,
                             reason=None if math.isfinite(value) else 'undefined-return-statistic',
                             numerator=numerator, denominator=len(net) if numerator is not None else None)
    return result


_MEMBER = pl.List(pl.Struct({'index': pl.Int64, 'id': pl.String, 'prediction': pl.Int64, 'probability': pl.Float64, 'reason': pl.String}))
_SCENARIO = pl.List(pl.Struct({'index': pl.Int64, **dict.fromkeys(('raw_position', 'raw_gross', 'raw_net', 'raw_cost', 'deployed_position', 'deployed_gross', 'deployed_net', 'deployed_cost'), pl.Float64)}))
_COLUMNS = {'open_ns': pl.Int64, 'close_ns': pl.Int64, 'role': pl.String, 'source_row_sha256': pl.String,
            'members': _MEMBER, 'aggregate_prediction': pl.Int64, 'aggregate_probability': pl.Float64,
            'aggregate_reason': pl.String, 'label': pl.Int64, 'label_reason': pl.String, 'scenarios': _SCENARIO}
_REPORT_KEYS: set[str] = set(('schema_version', 'attempt_id', 'assessment_index', 'state', 'created_ns', 'finished_ns', 'project_id', 'reservation', 'spec', 'spec_sha256', 'state_sha256', 'scenarios_sha256', 'source', 'source_column_map', 'source_schema', 'planned', 'prior', 'exposure', 'rows_path', 'rows_sha256', 'classification', 'scenarios'))


def retained_rows(raw: pl.DataFrame, step: int, start: int, tape: list[dict[str, object]],
                  labels: list[int | None], evaluated: bool, scenarios: list[list[dict[str, object]]]) -> tuple[pl.DataFrame, dict[str, str]]:
    times = raw['datetime'].dt.epoch('ns').to_list()
    offset = sum(value < start for value in times)
    rows: list[dict[str, object]] = []
    mapping = {name: name if name in ('open', 'high', 'low', 'close', 'volume') else f'source_{name}' for name in raw.columns}
    for index, timestamp in enumerate(times):
        final = index >= offset
        point = tape[index - offset] if final else {}
        source = raw.slice(index, 1)
        row: dict[str, object] = {'open_ns': timestamp, 'close_ns': timestamp + step, 'role': 'final' if final else 'context',
            'source_row_sha256': sha256(source.write_ipc(None, compression='uncompressed').getvalue()).hexdigest(),
            'members': point.get('members'), 'aggregate_prediction': point.get('prediction'), 'aggregate_probability': point.get('probability'), 'aggregate_reason': point.get('reason'),
            'label': labels[index - offset] if final else None,
            'label_reason': ('terminal-label-unavailable' if index == len(times) - 1 else None) if evaluated and final else 'no-labels' if final else None,
            'scenarios': scenarios[index - offset] if final else None}
        row.update({mapping[name]: float(value) if name in ('open', 'high', 'low', 'close', 'volume') and value is not None else value for name, value in raw.row(index, named=True).items()})
        rows.append(row)
    schema = {**_COLUMNS, **{mapping[name]: pl.Float64 if name in ('open', 'high', 'low', 'close', 'volume') else dtype for name, dtype in raw.schema.items()}}
    return pl.DataFrame(rows, schema=schema), mapping


def exposure(reservation: Mapping[str, object], prior: Mapping[str, object], provenances: list[dict[str, object]], start: int, end: int) -> dict[str, object]:
    attempts = cast(list[dict[str, object]], prior['attempts'])
    if reservation['state'] == 'released':
        classification, reasons = 'post-release', ['reservation-released']
    elif any(int(cast(int, row['context_start_ns'])) < end and start < int(cast(int, row['end_ns'])) for row in attempts):
        classification, reasons = 'subsequent', ['prior-planned-support-intersects-final']
    elif any(cast(dict[str, object], proof.get('source', {})).get('kind') != 'hosted_native' for proof in provenances):
        classification, reasons = 'unknown', ['non-hosted-or-caller-training-source']
    else:
        classification, reasons = 'initial-recorded', ['conditional-on-local-completed-research-coverage']
    return {'class': classification, 'reasons': reasons, 'unknown_external_exposure': True}


def _path(root: Path, value: object, attempt_id: str, filename: str) -> Path:
    expected = f'manifests/holdout/{attempt_id}/{filename}'
    if value != expected:
        raise ValueError(f'Holdout attempt {attempt_id} has an invalid artifact path')
    target = (root / expected).resolve()
    if not target.is_relative_to((root / 'manifests/holdout').resolve()):
        raise ValueError(f'Holdout attempt {attempt_id} artifact escapes evidence directory')
    return target


def _sync_replace(temp: Path, target: Path) -> None:
    with temp.open('rb') as stream:
        os.fsync(stream.fileno())
    _ = temp.replace(target)
    directory = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def artifacts(root: Path, report: dict[str, object], rows: pl.DataFrame) -> dict[str, object]:
    attempt_id = cast(str, report['attempt_id'])
    rows_relative = f'manifests/holdout/{attempt_id}/rows.parquet'
    report_relative = f'manifests/holdout/{attempt_id}/report.json'
    target = _path(root, rows_relative, attempt_id, 'rows.parquet')
    target.parent.mkdir(parents=True, exist_ok=False)
    temp = target.with_suffix('.parquet.tmp')
    rows.write_parquet(temp)
    _sync_replace(temp, target)
    rows_hash = sha256(target.read_bytes()).hexdigest()
    report.update(rows_path=rows_relative, rows_sha256=rows_hash)
    target = _path(root, report_relative, attempt_id, 'report.json')
    temp = target.with_suffix('.json.tmp')
    _ = temp.write_bytes(encoded(report))
    _sync_replace(temp, target)
    return {'rows_path': rows_relative, 'rows_sha256': rows_hash, 'report_path': report_relative, 'report_sha256': sha256(target.read_bytes()).hexdigest()}


def read_holdout_report(project_root: str | Path, attempt_id: str) -> dict[str, object]:
    """Read completed evidence, verifying hashes and schemas without evaluating."""
    from limen.experiment._holdout_attempts import completed
    try:
        root = history.resolve_root(project_root)
        record = completed(root, attempt_id)
        report_file = _path(root, record['report_path'], attempt_id, 'report.json')
        rows_file = _path(root, record['rows_path'], attempt_id, 'rows.parquet')
        if sha256(report_file.read_bytes()).hexdigest() != record['report_sha256'] or sha256(rows_file.read_bytes()).hexdigest() != record['rows_sha256']:
            raise ValueError('Artifact hash mismatch')
        report = cast(dict[str, object], json.loads(report_file.read_bytes()))
        _ = encoded(report)
        if set(report) != _REPORT_KEYS or report['schema_version'] != 1 or report['state'] != 'completed' or report['attempt_id'] != attempt_id:
            raise ValueError('Invalid report schema or identity')
        for key in ('project_id', 'assessment_index', 'created_ns', 'finished_ns', 'spec_sha256', 'state_sha256', 'scenarios_sha256', 'rows_path', 'rows_sha256'):
            if report[key] != record[key]:
                raise ValueError(f'Report/history {key} mismatch')
        rows = pl.read_parquet(rows_file)
        if any(rows.schema.get(name) != dtype for name, dtype in _COLUMNS.items()):
            raise ValueError('Invalid retained rows schema')
        mapping = cast(dict[str, str], report['source_column_map'])
        if set(rows.columns) != set(_COLUMNS) | set(mapping.values()) or rows.is_empty() or set(rows['role'].to_list()) != {'context', 'final'}:
            raise ValueError('Invalid retained source/role columns')
        return report
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, pl.exceptions.PolarsError) as exc:
        raise ValueError(f'Holdout attempt {attempt_id}: {exc}') from exc

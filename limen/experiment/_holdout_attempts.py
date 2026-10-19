from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import sqlite3
import time
from typing import cast
from uuid import uuid4

from limen.experiment import _holdout_history as history
from limen.experiment._holdout_freeze import encoded

_VERSION = 2
_HISTORY = cast(str, vars(history)['_HISTORY'])
_identity = cast(Callable[[Path, sqlite3.Connection], tuple[str, int]], vars(history)['_identity'])
_SCHEMA = """CREATE TABLE attempts (
 id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES project(id),
 assessment_index INTEGER NOT NULL CHECK(assessment_index>=1),
 reservation_id TEXT NOT NULL REFERENCES reservations(id), revision INTEGER NOT NULL,
 reservation_state TEXT NOT NULL CHECK(reservation_state IN ('active','released')),
 state TEXT NOT NULL CHECK(state IN ('started','completed','failed')),
 context_start_ns INTEGER NOT NULL, start_ns INTEGER NOT NULL, end_ns INTEGER NOT NULL,
 spec_sha256 TEXT NOT NULL, state_sha256 TEXT NOT NULL, scenarios_sha256 TEXT NOT NULL,
 prior_json TEXT NOT NULL, support_json TEXT NOT NULL,
 report_path TEXT, report_sha256 TEXT, rows_path TEXT, rows_sha256 TEXT,
 error_json TEXT, created_ns INTEGER NOT NULL, finished_ns INTEGER,
 CHECK(context_start_ns<start_ns AND start_ns<end_ns), UNIQUE(project_id,assessment_index))"""


def migrate(root: Path) -> None:
    with history.admission(root), history.connection(root / _HISTORY) as db:
        _ = _identity(root, db)
        if db.execute('PRAGMA user_version').fetchone()[0] == _VERSION:
            return
    with history.admission(root, exclusive=True), history.connection(root / _HISTORY) as db:
        _ = _identity(root, db)
        _ = db.execute('BEGIN IMMEDIATE')
        if db.execute('PRAGMA user_version').fetchone()[0] == 1:
            _ = db.execute(_SCHEMA)
            _ = db.execute('CREATE INDEX attempt_support ON attempts(project_id,context_start_ns,end_ns)')
            _ = db.execute('PRAGMA user_version=2')
        db.commit()


def reservation(root: Path, reservation_id: str, start: str | None, end: str | None, context: str) -> dict[str, object]:
    binding = history.read_binding(root)
    with history.connection(root / _HISTORY) as db:
        row = db.execute("SELECT * FROM reservations WHERE id=? AND project_id=? AND state!='withdrawn'", (history.identifier(reservation_id), binding.project_id)).fetchone()
        if row is None:
            raise ValueError('Holdout reservation is unknown or withdrawn')
        first = history.utc_day(start) if start is not None else int(row['start_ns'])
        last = history.utc_day(end) if end is not None else int(row['end_ns'])
        if not int(row['start_ns']) <= first < last <= int(row['end_ns']) or history.utc_day(context) >= first:
            raise ValueError('Holdout final bounds must be contained and follow context_start')
        return {'id': reservation_id, 'project_id': binding.project_id, 'revision': binding.revision, 'state': row['state'],
                'start_ns': first, 'end_ns': last, 'interval_seconds': int(row['interval_seconds'])}


def _prior(db: sqlite3.Connection, project: str, context: int, start: int, end: int) -> dict[str, object]:
    attempts = [{'id': row['id'], 'index': row['assessment_index'], 'state_at_allocation': row['state'], 'context_start_ns': row['context_start_ns'], 'start_ns': row['start_ns'], 'end_ns': row['end_ns']}
                for row in db.execute('SELECT * FROM attempts WHERE project_id=? AND context_start_ns<? AND end_ns>? ORDER BY assessment_index', (project, end, context))]
    reservations = [dict(row) for row in db.execute("SELECT id,state,start_ns,end_ns FROM reservations WHERE project_id=? AND state!='withdrawn' AND start_ns<? AND end_ns>? ORDER BY start_ns,id", (project, end, context))]
    prior_final = [row['id'] for row in attempts if row['start_ns'] < end and start < row['end_ns']]
    reserved_context = [row['id'] for row in attempts if row['context_start_ns'] < end and start < row['start_ns'] and any(item['start_ns'] < min(row['start_ns'], end) and max(row['context_start_ns'], start) < item['end_ns'] for item in reservations)]
    current_reserved = [row['id'] for row in reservations if row['start_ns'] < start and context < row['end_ns']]
    ordinary = [row['id'] for row in attempts if row['context_start_ns'] < start and context < row['start_ns'] and not any(item['start_ns'] < min(start, row['start_ns']) and max(context, row['context_start_ns']) < item['end_ns'] for item in reservations)]
    return {'attempts': attempts, 'reservations': reservations, 'overlaps': {'prior_final_ids': prior_final,
            'prior_reserved_context_ids': reserved_context, 'current_reserved_context_ids': current_reserved, 'ordinary_context_ids': ordinary}}


def allocate(root: Path, declaration: Mapping[str, object], planned: dict[str, int], *, spec: str, state: str, scenarios: str) -> dict[str, object]:
    with history.connection(root / _HISTORY) as db:
        _ = db.execute('BEGIN IMMEDIATE')
        project, revision = _identity(root, db)
        row = db.execute('SELECT state FROM reservations WHERE id=?', (declaration['id'],)).fetchone()
        if revision != declaration['revision'] or row is None or row['state'] != declaration['state']:
            raise ValueError('Holdout reservation changed before allocation')
        index = int(db.execute('SELECT COALESCE(MAX(assessment_index),0)+1 FROM attempts WHERE project_id=?', (project,)).fetchone()[0])
        prior = _prior(db, project, planned['context_start_ns'], planned['start_ns'], planned['end_ns'])
        support = {'planned': planned, 'actual': None}
        value, now = str(uuid4()), time.time_ns()
        _ = db.execute("INSERT INTO attempts (id,project_id,assessment_index,reservation_id,revision,reservation_state,state,context_start_ns,start_ns,end_ns,spec_sha256,state_sha256,scenarios_sha256,prior_json,support_json,created_ns) VALUES (?,?,?,?,?,?,'started',?,?,?,?,?,?,?,?,?)",
            (value, project, index, declaration['id'], revision, declaration['state'], planned['context_start_ns'], planned['start_ns'], planned['end_ns'], spec, state, scenarios, encoded(prior).decode(), encoded(support).decode(), now))
        db.commit()
        return {'id': value, 'project_id': project, 'assessment_index': index, 'created_ns': now, 'prior': prior, 'planned': planned}


def finalize(root: Path, attempt_id: str, *, source: object, artifacts: Mapping[str, object] | None = None,
             error: Mapping[str, object] | None = None, finished_ns: int | None = None) -> None:
    with history.connection(root / _HISTORY) as db:
        _ = db.execute('BEGIN IMMEDIATE')
        row = db.execute("SELECT support_json FROM attempts WHERE id=? AND state='started'", (attempt_id,)).fetchone()
        if row is None:
            raise ValueError(f'Holdout attempt {attempt_id} is unknown or already finalized')
        support = json.loads(row['support_json'])
        support['actual'] = source
        files = artifacts or {}
        result = db.execute("UPDATE attempts SET state=?,support_json=?,report_path=?,report_sha256=?,rows_path=?,rows_sha256=?,error_json=?,finished_ns=? WHERE id=? AND state='started'",
            ('failed' if error is not None else 'completed', encoded(support).decode(), files.get('report_path'), files.get('report_sha256'), files.get('rows_path'), files.get('rows_sha256'), encoded(error).decode() if error is not None else None, finished_ns if finished_ns is not None else time.time_ns(), attempt_id))
        if result.rowcount != 1:
            raise ValueError(f'Holdout attempt {attempt_id} finalization failed')
        db.commit()


def completed(root: Path, attempt_id: str) -> dict[str, object]:
    with history.admission(root), history.connection(root / _HISTORY) as db:
        project, _ = _identity(root, db)
        if db.execute('PRAGMA user_version').fetchone()[0] != _VERSION:
            raise ValueError(f'Holdout attempt {attempt_id} is unknown')
        row = db.execute("SELECT * FROM attempts WHERE id=? AND project_id=? AND state='completed'", (history.identifier(attempt_id), project)).fetchone()
        if row is None:
            raise ValueError(f'Holdout attempt {attempt_id} is unknown or not completed')
        return cast(dict[str, object], dict(row))

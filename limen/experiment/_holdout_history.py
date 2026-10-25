from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import cast
from uuid import UUID, uuid4


_BUSY_TIMEOUT_SECONDS = 5.0
_UUID_VERSION = 4
_ATTEMPT_VERSION = 2
_LEASES: ContextVar[tuple[tuple[Path, bool], ...]] = ContextVar('holdout_leases', default=())
_HISTORY = 'manifests/holdout.sqlite3'
_MARKER = 'manifests/holdout-project.json'
_LEASE = 'manifests/holdout-lease.sqlite3'
_SIDECARS = ('', '-journal', '-wal', '-shm')
_SCHEMA = (
    'CREATE TABLE project (id TEXT PRIMARY KEY, revision INTEGER NOT NULL CHECK(revision>=1))',
    "CREATE TABLE reservations (id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES project(id), start_ns INTEGER NOT NULL, end_ns INTEGER NOT NULL CHECK(end_ns>start_ns), interval_seconds INTEGER NOT NULL CHECK(interval_seconds>0 AND 86400%interval_seconds=0), state TEXT NOT NULL CHECK(state IN ('active','withdrawn','released')), created_ns INTEGER NOT NULL, changed_ns INTEGER NOT NULL, created_revision INTEGER NOT NULL CHECK(created_revision>=1))",
    'CREATE INDEX reservation_bounds ON reservations(project_id,start_ns,end_ns)',
    "CREATE TABLE research (id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES project(id), revision INTEGER NOT NULL, role TEXT NOT NULL CHECK(role IN ('experiment','trainer','replay')), state TEXT NOT NULL CHECK(state IN ('started','completed','failed')), manifest_id TEXT, round_id TEXT, provenance_json TEXT NOT NULL, created_ns INTEGER NOT NULL, finished_ns INTEGER, error TEXT)",
    'CREATE INDEX research_revision ON research(project_id,revision)',
)


class HoldoutEvidenceError(ValueError):
    """A protected operation cannot establish its current research evidence."""


@dataclass(frozen=True)
class ProjectBinding:
    root: Path
    project_id: str | None
    revision: int
    cutoff_ns: int | None
    interval_seconds: int | None


def utc_day(value: str) -> int:
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value) is None:
        raise ValueError('Holdout date must be YYYY-MM-DD')
    day = date.fromisoformat(value)
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()) * 1_000_000_000


def identifier(value: str) -> str:
    parsed = UUID(value)
    if parsed.version != _UUID_VERSION or str(parsed) != value:
        raise ValueError('Holdout identifier must be a canonical UUID4')
    return value


def _git(root: Path, *args: str) -> str:
    from limen.cli.git_utils import run_git
    result = run_git(list(args), cwd=root)
    result.check_returncode()
    return result.stdout.strip()


def resolve_root(value: str | Path) -> Path:
    from limen.yaml.config import find_project_root
    root = find_project_root(Path(value))
    if root is None:
        raise ValueError('Holdout requires a project containing limen.toml')
    _ = _git(root, 'rev-parse', '--is-inside-work-tree')
    return root


def _exclude(root: Path) -> None:
    paths = [name + suffix for name in (_HISTORY, _LEASE) for suffix in _SIDECARS] + ['manifests/holdout']
    if _git(root, 'ls-files', '--', *paths):
        raise ValueError('Holdout history/evidence must not be tracked in Git')
    location, _, prefix = _git(root, 'rev-parse', '--git-path', 'info/exclude', '--show-prefix').partition('\n')
    target = Path(location)
    target = target if target.is_absolute() else root / target
    old = target.read_text() if target.exists() else ''
    entries = [f'/{prefix}{name}{suffix}' for name in (_HISTORY, _LEASE) for suffix in _SIDECARS]
    entries.append(f'/{prefix}manifests/holdout/')
    missing = [entry for entry in entries if entry not in old.splitlines()]
    if missing:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('a') as stream:
            _ = stream.write(('\n' if old and not old.endswith('\n') else '') + '\n'.join(missing) + '\n')


@contextmanager
def connection(path: Path, *, create: bool = False) -> Iterator[sqlite3.Connection]:
    target = str(path) if create else path.resolve().as_uri() + '?mode=rw'
    db = sqlite3.connect(target, uri=not create, timeout=_BUSY_TIMEOUT_SECONDS, isolation_level=None)
    db.row_factory = sqlite3.Row
    try:
        _ = db.execute('PRAGMA foreign_keys=ON')
        yield db
    finally:
        db.close()


def _initialize_lease(db: sqlite3.Connection) -> None:
    if db.execute('PRAGMA journal_mode').fetchone()[0] != 'delete':
        raise ValueError('Holdout admission requires rollback-journal storage')
    version = db.execute('PRAGMA user_version').fetchone()[0]
    if version == 0:
        _ = db.execute('BEGIN IMMEDIATE')
        if db.execute('PRAGMA user_version').fetchone()[0] == 0:
            _ = db.execute('CREATE TABLE IF NOT EXISTS lease (id INTEGER PRIMARY KEY CHECK(id=1))')
            _ = db.execute('INSERT OR IGNORE INTO lease VALUES (1)')
            _ = db.execute('PRAGMA user_version=1')
        db.commit()
    elif version != 1:
        raise ValueError('Unsupported holdout lease schema')


@contextmanager
def admission(root: Path, *, exclusive: bool = False) -> Iterator[None]:
    root = root.resolve()
    held = next((entry for entry in _LEASES.get() if entry[0] == root), None)
    if held is not None:
        if exclusive and not held[1]:
            raise ValueError('Cannot change reservations inside active research')
        yield
        return
    _exclude(root)
    (root / 'manifests').mkdir(exist_ok=True)
    with connection(root / _LEASE, create=True) as db:
        _initialize_lease(db)
        _ = db.execute('BEGIN EXCLUSIVE' if exclusive else 'BEGIN')
        _ = db.execute('SELECT id FROM lease').fetchall()
        token = _LEASES.set((*_LEASES.get(), (root, exclusive)))
        try:
            yield
        finally:
            _LEASES.reset(token)
            db.rollback()


def _write_marker(root: Path, project_id: str) -> None:
    target = root / _MARKER
    temp = target.with_name(f'{target.name}.{uuid4()}.tmp')
    _ = temp.write_text(json.dumps({'schema_version': 1, 'project_id': project_id, 'history': _HISTORY}, sort_keys=True) + '\n')
    _ = temp.replace(target)


def _identity(root: Path, db: sqlite3.Connection) -> tuple[str, int]:
    if db.execute('PRAGMA journal_mode').fetchone()[0] != 'delete':
        raise ValueError('Holdout history requires rollback-journal storage')
    if db.execute('PRAGMA user_version').fetchone()[0] not in (1, 2):
        raise ValueError('Unsupported holdout history schema; do not downgrade history')
    rows = db.execute('SELECT id,revision FROM project').fetchall()
    if len(rows) != 1:
        raise ValueError('Holdout history requires exactly one project identity')
    project_id, revision = str(rows[0]['id']), int(rows[0]['revision'])
    _ = identifier(project_id)
    marker = root / _MARKER
    if marker.exists():
        value = json.loads(marker.read_text())
        if value != {'schema_version': 1, 'project_id': project_id, 'history': _HISTORY}:
            raise ValueError('Holdout project marker/history mismatch')
    else:
        _write_marker(root, project_id)
    return project_id, revision


def read_binding(root: Path) -> ProjectBinding:
    if not (root / _HISTORY).exists():
        if (root / _MARKER).exists():
            raise ValueError('Holdout project marker exists but history is missing')
        return ProjectBinding(root, None, 0, None, None)
    with connection(root / _HISTORY) as db:
        project_id, revision = _identity(root, db)
        rows = db.execute("SELECT start_ns,interval_seconds FROM reservations WHERE state='active' ORDER BY start_ns").fetchall()
        return ProjectBinding(root, project_id, revision, int(rows[0]['start_ns']) if rows else None, int(rows[0]['interval_seconds']) if rows else None)


def bind(*hints: str | Path | None) -> ProjectBinding | None:
    from limen.yaml.config import find_project_root
    roots = {root for hint in (*hints, Path.cwd()) if hint is not None and (root := find_project_root(Path(hint))) is not None}
    if len(roots) > 1:
        raise ValueError('Holdout project root hints disagree')
    if not roots:
        return None
    root = roots.pop()
    from limen.cli.git_utils import run_git
    probe = run_git(['rev-parse', '--is-inside-work-tree'], cwd=root)
    if probe.returncode:
        if (root / _MARKER).exists() or (root / _HISTORY).exists():
            raise ValueError('Protected holdout history requires its Git project')
        return None
    with admission(root):
        return read_binding(root)


def check_binding(binding: ProjectBinding) -> None:
    try:
        current = read_binding(binding.root)
    except ValueError as exc:
        raise HoldoutEvidenceError(str(exc)) from exc
    if current != binding:
        raise HoldoutEvidenceError('Holdout reservation revision changed; construct a new research object')


def reserve_holdout(project_root: str | Path, start: str, end: str, *, interval_seconds: int) -> str:
    """Exclude the continuous suffix from the earliest active reservation onward."""
    root = resolve_root(project_root)
    first, last = utc_day(start), utc_day(end)
    if first >= last or type(interval_seconds) is not int or interval_seconds <= 0 or 86400 % interval_seconds:
        raise ValueError('Holdout requires start<end and an integer native interval dividing 86400')
    with admission(root, exclusive=True):
        path = root / _HISTORY
        if not path.exists():
            if (root / _MARKER).exists():
                raise ValueError('Holdout history is missing; restore it before reserving')
            with connection(path, create=True) as db:
                _ = db.execute('BEGIN IMMEDIATE')
                for statement in _SCHEMA:
                    _ = db.execute(statement)
                _ = db.execute('INSERT INTO project VALUES (?,1)', (str(uuid4()),))
                _ = db.execute('PRAGMA user_version=1')
                db.commit()
        with connection(path) as db:
            _ = db.execute('BEGIN IMMEDIATE')
            project_id, revision = _identity(root, db)
            if db.execute("SELECT 1 FROM reservations WHERE state!='withdrawn' AND start_ns<? AND end_ns>?", (last, first)).fetchone():
                raise ValueError('Holdout intersects an active or released reservation')
            if db.execute("SELECT 1 FROM reservations WHERE state='active' AND interval_seconds!=?", (interval_seconds,)).fetchone():
                raise ValueError('Active holdouts must share one native interval')
            value, now = str(uuid4()), time.time_ns()
            _ = db.execute("INSERT INTO reservations VALUES (?,?,?,?,?, 'active',?,?,?)", (value, project_id, first, last, interval_seconds, now, now, revision + 1))
            _ = db.execute('UPDATE project SET revision=revision+1')
            db.commit()
            return value


def _retire(root_value: str | Path, reservation_id: str, state: str) -> None:
    root = resolve_root(root_value)
    _ = identifier(reservation_id)
    with admission(root, exclusive=True):
        if not (root / _HISTORY).exists():
            raise ValueError('Holdout history is missing')
        with connection(root / _HISTORY) as db:
            _ = db.execute('BEGIN IMMEDIATE')
            _ = _identity(root, db)
            row = db.execute('SELECT state,created_revision FROM reservations WHERE id=?', (reservation_id,)).fetchone()
            if row is None or row['state'] != 'active':
                raise ValueError('Holdout is unknown or no longer active')
            if state == 'withdrawn' and db.execute('SELECT 1 FROM research WHERE revision>=?', (row['created_revision'],)).fetchone():
                raise ValueError('A used holdout cannot be withdrawn; release it explicitly')
            if state == 'withdrawn' and db.execute('PRAGMA user_version').fetchone()[0] == _ATTEMPT_VERSION and db.execute('SELECT 1 FROM attempts WHERE revision>=?', (row['created_revision'],)).fetchone():
                raise ValueError('A used holdout cannot be withdrawn; release it explicitly')
            _ = db.execute('UPDATE reservations SET state=?,changed_ns=? WHERE id=?', (state, time.time_ns(), reservation_id))
            _ = db.execute('UPDATE project SET revision=revision+1')
            db.commit()


def withdraw_holdout(project_root: str | Path, reservation_id: str) -> None:
    """Retire an unused reservation without deleting its identity or history."""
    _retire(project_root, reservation_id, 'withdrawn')


def release_holdout(project_root: str | Path, reservation_id: str) -> None:
    """Lift research exclusion while preserving the post-release exposure record."""
    _retire(project_root, reservation_id, 'released')


def start_research(binding: ProjectBinding, role: str, manifest_id: str | None = None) -> str:
    if role not in ('experiment', 'trainer', 'replay'):
        raise ValueError('Unknown holdout research role')
    with admission(binding.root):
        check_binding(binding)
        if binding.project_id is None or binding.cutoff_ns is None:
            raise ValueError('Protected research requires active reservation history')
        value = str(uuid4())
        with connection(binding.root / _HISTORY) as db:
            _ = db.execute('BEGIN IMMEDIATE')
            _ = db.execute("INSERT INTO research VALUES (?,?,?,?, 'started',?,NULL,'{}',?,NULL,NULL)", (value, binding.project_id, binding.revision, role, manifest_id, time.time_ns()))
            db.commit()
        return value


def finish_research(binding: ProjectBinding, record_id: str, provenance: Mapping[str, object], *, error: str | None = None) -> None:
    _ = identifier(record_id)
    with admission(binding.root), connection(binding.root / _HISTORY) as db:
        check_binding(binding)
        _ = db.execute('BEGIN IMMEDIATE')
        round_id = provenance.get('round_id')
        if round_id is not None and not isinstance(round_id, str):
            raise ValueError('Holdout round identity must be a string')
        result = db.execute("UPDATE research SET state=?,round_id=?,provenance_json=?,finished_ns=?,error=? WHERE id=? AND project_id=? AND revision=? AND state='started'", ('failed' if error is not None else 'completed', round_id, json.dumps(dict(provenance), sort_keys=True, allow_nan=False), time.time_ns(), error, record_id, binding.project_id, binding.revision))
        if result.rowcount != 1:
            raise ValueError('Holdout research record is unknown, stale or already finalized')
        db.commit()


def research_record(binding: ProjectBinding, record_id: str) -> Mapping[str, object]:
    with admission(binding.root), connection(binding.root / _HISTORY) as db:
        row = db.execute('SELECT * FROM research WHERE id=? AND project_id=?', (identifier(record_id), binding.project_id)).fetchone()
        if row is None:
            raise ValueError('Unknown holdout research record')
        return cast(Mapping[str, object], dict(row))

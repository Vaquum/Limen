"""PYTEST_DONT_REWRITE: the controller imports this assertion-free plugin."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import coverage
import pytest


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY_ENV = 'LIMEN_PROOF_WORKER_DIRECTORY'
RECEIPT_KEY = 'limen_proof'


@dataclass
class _Worker:
    id: str
    measurement: coverage.Coverage | None = None
    backend: str | None = None
    completed: list[str] = field(default_factory=list)


_WORKER = pytest.StashKey[_Worker]()


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption('loadgroup', False):
        shared = {item.path for item in items if 'fold_campaigns' in item.fixturenames}
        for item in items:
            if item.get_closest_marker('xdist_group') is None:
                group = 'fold-campaigns' if item.path in shared else item.nodeid.split('::', 1)[0]
                item.add_marker(pytest.mark.xdist_group(group))


def _require_backend(backend: str) -> None:
    if backend not in ('SysMonitor', 'CTracer', 'PyTracer'):
        raise ValueError(f'Unrecognized worker coverage backend: {backend}')
    if (os.getenv('LIMEN_COVERAGE_RUN') == '1'
            and os.getenv('COVERAGE_CORE') == 'sysmon'
            and (sys.version_info < (3, 12) or backend != 'SysMonitor')):
        raise ValueError('The pinned covered proof requires actual SysMonitor workers')


@contextmanager
def _empty_data_warning(measurement: coverage.Coverage) -> Iterator[None]:
    original = measurement.get_option('run:disable_warnings')
    measurement.set_option('run:disable_warnings', [*original, 'no-data-collected'])
    try:
        yield
    finally:
        measurement.set_option('run:disable_warnings', original)


@contextmanager
def worker_coverage() -> Iterator[Path | None]:
    previous = os.environ.pop(DIRECTORY_ENV, None)
    try:
        if coverage.Coverage.current() is None:
            yield None
        else:
            with TemporaryDirectory(prefix='limen-proof-') as directory:
                os.environ[DIRECTORY_ENV] = directory
                yield Path(directory)
    finally:
        if previous is None:
            os.environ.pop(DIRECTORY_ENV, None)
        else:
            os.environ[DIRECTORY_ENV] = previous


def _data_path(directory: Path, worker_id: str) -> Path:
    if Path(worker_id).name != worker_id or worker_id in ('.', '..'):
        raise ValueError(f'Invalid worker database identity: {worker_id!r}')
    return directory / worker_id


def _line_evidence(data: coverage.CoverageData) -> tuple[str, int]:
    executed = []
    for filename in sorted(data.measured_files()):
        lines = sorted(data.lines(filename) or [])
        if lines:
            executed.append((filename, lines))
    encoded = json.dumps(executed, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest(), sum(len(lines) for _, lines in executed)


def pytest_sessionstart(session: pytest.Session) -> None:
    worker_input = getattr(session.config, 'workerinput', None)
    if worker_input is None:
        return
    worker = _Worker(worker_input['workerid'])
    session.config.stash[_WORKER] = worker
    directory = os.getenv(DIRECTORY_ENV)
    if directory is None:
        return
    measurement = coverage.Coverage(
        config_file=str(ROOT / 'pyproject.toml'),
        data_file=str(_data_path(Path(directory), worker.id)),
        context=worker.id,
    )
    worker.measurement = measurement
    measurement.start()
    worker.backend = str(dict(measurement.sys_info())['core'])
    _require_backend(worker.backend)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None):
    worker = item.config.stash.get(_WORKER, None)
    if (worker is not None and worker.measurement is not None
            and coverage.Coverage.current() is not worker.measurement):
        raise RuntimeError('Worker lost its active coverage collector before execution')
    result = yield
    if worker is not None:
        if worker.measurement is not None and coverage.Coverage.current() is not worker.measurement:
            raise RuntimeError('Worker lost its active coverage collector during execution')
        worker.completed.append(item.nodeid)
    return result


def pytest_sessionfinish(session: pytest.Session) -> None:
    worker = session.config.stash.get(_WORKER, None)
    if worker is None:
        return
    receipt = {
        'id': worker.id,
        'pid': os.getpid(),
        'backend': worker.backend,
        'test_count': len(worker.completed),
        'completed_ids': worker.completed,
        'data_sha256': None,
        'lines_sha256': None,
        'line_count': None,
    }
    if worker.measurement is not None:
        measurement = worker.measurement
        measurement.stop()
        before = measurement.get_option('run:disable_warnings')
        filters = warnings.filters[:]
        with _empty_data_warning(measurement):
            data = measurement.get_data()
            lines_sha256, line_count = _line_evidence(data)
            measurement.save()
        after = measurement.get_option('run:disable_warnings')
        if before != after or filters != warnings.filters:
            raise RuntimeError('Worker coverage save changed the warning policy')
        receipt.update(
            data_sha256=hashlib.sha256(Path(data.data_filename()).read_bytes()).hexdigest(),
            lines_sha256=lines_sha256,
            line_count=line_count,
            warning_options_before=before,
            warning_options_after=after,
            warning_filters_unchanged=filters == warnings.filters,
        )
    session.config.workeroutput[RECEIPT_KEY] = receipt


def merge_worker_coverage(directory: Path, worker_ids: Mapping[str, dict]) -> None:
    measured = []
    pids = set()
    for worker_id, receipt in worker_ids.items():
        if receipt['id'] != worker_id or not isinstance(receipt['pid'], int) or receipt['pid'] <= 0:
            raise ValueError(f'Invalid coverage receipt identity for {worker_id}')
        if receipt['pid'] in pids:
            raise ValueError('Covered workers must have distinct process IDs')
        pids.add(receipt['pid'])
        _require_backend(receipt['backend'])
        path = _data_path(directory, worker_id)
        if not path.is_file():
            raise ValueError(f'Missing worker coverage database: {worker_id}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['data_sha256']:
            raise ValueError(f'Worker coverage database digest mismatch: {worker_id}')
        data = coverage.CoverageData(basename=str(path))
        data.read()
        lines_sha256, line_count = _line_evidence(data)
        if lines_sha256 != receipt['lines_sha256'] or line_count != receipt['line_count']:
            raise ValueError(f'Missing or changed executed coverage evidence: {worker_id}')
        if line_count and data.measured_contexts() != {worker_id}:
            raise ValueError(f'Worker coverage context mismatch: {worker_id}')
        measured.append(data)
    outer = coverage.Coverage.current()
    if outer is None:
        raise ValueError('Covered workers lost the outer coverage collector')
    with _empty_data_warning(outer):
        target = outer.get_data()
    for data in measured:
        target.update(data)

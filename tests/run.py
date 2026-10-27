from __future__ import annotations

import logging
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from tests.utils.cleanup import cleanup_csv_files
from tests.utils.cleanup import setup_cleanup_handlers
from tests.utils.proof_coverage import RECEIPT_KEY
from tests.utils.proof_coverage import merge_worker_coverage
from tests.utils.proof_coverage import worker_coverage
from tests.utils.runtime_tracking import DEFAULT_SLOWEST_TESTS_LIMIT
from tests.utils.runtime_tracking import build_runtime_profile
from tests.utils.runtime_tracking import format_duration
from tests.utils.runtime_tracking import sorted_test_records
from tests.utils.runtime_tracking import utc_now_iso
from tests.utils.runtime_tracking import write_runtime_profile


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class RuntimeProfilePlugin:
    def __init__(self) -> None:
        self.suite_started_at = utc_now_iso()
        self.suite_started_clock = time.perf_counter()
        self.test_started_at: dict[str, str] = {}
        self.test_records: list[dict[str, Any]] = []
        self._records_by_execution: dict[tuple[str, str], dict] = {}
        self._events: dict[tuple[str, str], list[pytest.TestReport]] = {}
        self._completions: list[tuple[str, str]] = []
        self.collections: dict[str, tuple[str, ...]] = {}
        self.workers: dict[str, dict | None] = {}
        self.errors: list[str] = []
        self.requested_workers = 0
        self.distribution = 'no'
        self.reconciled = False
        self.combine_seconds = 0.0

    def pytest_configure(self, config: pytest.Config) -> None:
        self.requested_workers = config.getoption('numprocesses', 0) or 0
        self.distribution = config.getoption('dist', 'no')

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodeready(self, node) -> None:
        self.workers[node.gateway.id] = None

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids) -> None:
        worker_id = node.gateway.id
        if worker_id in self.collections:
            self.errors.append(f'Duplicate worker collection: {worker_id}')
        self.collections[worker_id] = tuple(ids)

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node, error) -> None:
        worker_id = node.gateway.id
        self.workers[worker_id] = getattr(node, 'workeroutput', {}).get(RECEIPT_KEY)
        if error is not None:
            self.errors.append(f'Worker {worker_id} failed: {error}')

    @property
    def parallel(self) -> bool:
        return bool(self.requested_workers or self.workers)

    def assert_complete(self) -> None:
        if not self.parallel:
            return
        if self.requested_workers and len(self.workers) != self.requested_workers:
            self.errors.append('Worker count differs from the requested process count')
        if self.workers.keys() != self.collections.keys():
            self.errors.append('Missing native worker collection')
        native = next(iter(self.collections.values()), ())
        if any(ids != native for ids in self.collections.values()):
            self.errors.append('Workers collected different ordered case lists')
        if len(native) != len(set(native)):
            self.errors.append('Native collection contains duplicate node IDs')
        pids = set()
        for worker_id, receipt in self.workers.items():
            if receipt is None:
                self.errors.append(f'Missing worker receipt: {worker_id}')
                continue
            if receipt['id'] != worker_id or not isinstance(receipt['pid'], int) or receipt['pid'] <= 0:
                self.errors.append(f'Invalid worker identity: {worker_id}')
            if receipt['pid'] in pids:
                self.errors.append('Workers reported the same process ID')
            pids.add(receipt['pid'])
            completed = [nodeid for worker, nodeid in self._completions if worker == worker_id]
            if receipt['completed_ids'] != completed or receipt['test_count'] != len(completed):
                self.errors.append(f'Worker completion receipt differs from phase events: {worker_id}')
        for (worker_id, nodeid), reports in self._events.items():
            phases = [report.when for report in reports]
            expected = ['setup', 'call', 'teardown'] if reports[0].when == 'setup' and reports[0].passed else ['setup', 'teardown']
            if phases != expected:
                self.errors.append(f'Incomplete or duplicate phase events: {worker_id} {nodeid}')
        if self.distribution == 'each':
            expected_completions = Counter((worker, nodeid) for worker in self.workers for nodeid in native)
            actual_completions = Counter(self._completions)
        else:
            expected_completions = Counter(native)
            actual_completions = Counter(nodeid for _, nodeid in self._completions)
        if actual_completions != expected_completions:
            self.errors.append('Completed case multiplicity differs from native collection')
        if set(self._events) != set(self._completions):
            self.errors.append('Execution ledger contains unfinished cases')
        self.reconciled = not self.errors
        if self.errors:
            raise ValueError('; '.join(self.errors))

    def pytest_runtest_protocol(
        self,
        item: pytest.Item,
        nextitem: pytest.Item | None,
    ) -> None:
        self.test_started_at[item.nodeid] = utc_now_iso()

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        key = (_worker_id(report), report.nodeid)
        reports = self._events.setdefault(key, [])
        duplicate = any(previous.when == report.when for previous in reports)
        reports.append(report)
        if report.when == 'teardown':
            self._completions.append(key)
        if duplicate:
            self.errors.append(f'Duplicate phase event: {key} {report.when}')
            return
        if (report.when == 'call'
                or (report.when in ('???', 'teardown') and report.failed)
                or (report.when == 'setup' and (report.failed or report.skipped))):
            self._record(report)
        if key in self._records_by_execution:
            record = self._records_by_execution[key]
            starts = [event.start for event in reports if event.start > 0]
            stops = [event.stop for event in reports if event.stop > 0]
            record['started_at'] = _report_time(min(starts)) if starts else ''
            record['finished_at'] = _report_time(max(stops)) if stops else ''

    def build_profile(self) -> dict[str, Any]:
        suite_finished_at = utc_now_iso()
        total_duration_seconds = time.perf_counter() - self.suite_started_clock
        profile = build_runtime_profile(
            test_records=self.test_records,
            suite_started_at=self.suite_started_at,
            suite_finished_at=suite_finished_at,
            total_duration_seconds=total_duration_seconds,
        )
        if self.parallel:
            native = next(iter(self.collections.values()), ())
            workers = []
            for worker_id, receipt in self.workers.items():
                if receipt is None:
                    workers.append(dict.fromkeys(('pid', 'backend', 'test_count', 'data_sha256', 'lines_sha256', 'line_count')) | {'id': worker_id})
                else:
                    workers.append({key: value for key, value in receipt.items() if key != 'completed_ids'})
            profile['parallel'] = {
                'worker_count': len(self.workers),
                'reconciled': self.reconciled,
                'collected_count': len(native),
                'collected_ids_sha256': hashlib.sha256(json.dumps(native, separators=(',', ':')).encode()).hexdigest(),
                'combine_seconds': round(self.combine_seconds, 6),
                'distribution': self.distribution,
                'execution_count': len(self._events),
                'workers': workers,
                'errors': self.errors,
            }
        return profile

    def _record(self, report: pytest.TestReport) -> None:
        status = _report_status(report)
        worker_id = _worker_id(report)
        key = (worker_id, report.nodeid)
        record = self._records_by_execution.get(key)
        if record is None:
            record = {
                'test_name': report.nodeid,
                'module': _module_name(report.nodeid),
                'status': 'unknown',
                'duration_seconds': 0.0,
                'started_at': self.test_started_at.get(
                    report.nodeid,
                    self.suite_started_at,
                ),
                'finished_at': utc_now_iso(),
            }
            if worker_id:
                record['worker_id'] = worker_id
            self._records_by_execution[key] = record
            self.test_records.append(record)

        record['status'] = _merge_status(str(record['status']), status)
        record['duration_seconds'] = round(
            float(record['duration_seconds']) + float(report.duration),
            6,
        )
        record['finished_at'] = utc_now_iso()

        if status == 'failed':
            record['error'] = report.longreprtext
        if status == 'skipped' and record['status'] == 'skipped':
            record['skip_reason'] = str(report.longrepr)
        if record['status'] != 'failed':
            record.pop('error', None)
        if record['status'] != 'skipped':
            record.pop('skip_reason', None)


def _worker_id(report: pytest.TestReport) -> str:
    node = getattr(report, 'node', None)
    return node.gateway.id if node is not None else getattr(report, 'worker_id', '')


def _report_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def _report_status(report: pytest.TestReport) -> str:
    if report.passed:
        return 'passed'
    if report.failed:
        return 'failed'
    if report.skipped:
        return 'skipped'
    return 'unknown'


def _merge_status(current_status: str, new_status: str) -> str:
    if current_status == 'failed' or new_status == 'failed':
        return 'failed'
    if current_status == 'skipped' or new_status == 'skipped':
        return 'skipped'
    if current_status == 'passed' or new_status == 'passed':
        return 'passed'
    return new_status


def _module_name(nodeid: str) -> str:
    module_path = nodeid.split('::', 1)[0]
    if module_path.endswith('.py'):
        module_path = module_path[:-3]
    return module_path.replace('/', '.').replace('\\', '.')


def _runtime_profile_output_path() -> Path | None:
    configured_path = os.getenv('LIMEN_RUNTIME_PROFILE_PATH')
    if configured_path:
        return Path(configured_path)

    if os.getenv('LIMEN_COVERAGE_RUN') or os.getenv('CI'):
        return Path('coverage-artifacts/test_runtime_profile.json')

    return None


def _runtime_slowest_tests_limit() -> int:
    configured_limit = os.getenv('LIMEN_RUNTIME_SLOWEST_LIMIT')
    if not configured_limit:
        return DEFAULT_SLOWEST_TESTS_LIMIT

    try:
        parsed_limit = int(configured_limit)
    except ValueError as exc:
        raise ValueError(
            'LIMEN_RUNTIME_SLOWEST_LIMIT must be a positive integer',
        ) from exc
    if parsed_limit <= 0:
        raise ValueError('LIMEN_RUNTIME_SLOWEST_LIMIT must be a positive integer')

    return parsed_limit


def _pytest_args(argv: list[str]) -> list[str]:
    if argv:
        return argv
    # Default discovery includes test_acceptance_gates.py, test_parallel_rounds.py and test_round_signals.py.
    return ['--assert=plain', '-n', '4', '--dist', 'loadgroup', '--max-worker-restart=0', 'tests']


def _write_profile(plugin: RuntimeProfilePlugin, slowest_tests_limit: int) -> None:
    profile = plugin.build_profile()
    output_path = _runtime_profile_output_path()
    if output_path is not None:
        write_runtime_profile(profile, output_path)
        logger.info('Runtime profile written to %s', output_path)

    suite = profile['suite']
    logger.info(
        'Test suite runtime: %s across %d tests (%d passed, %d failed, %d skipped)',
        format_duration(float(suite['duration_seconds'])),
        suite['test_count'],
        suite['passed_count'],
        suite['failed_count'],
        suite.get('skipped_count', 0),
    )

    slowest_tests = sorted_test_records(
        profile,
        limit=slowest_tests_limit,
    )
    if slowest_tests:
        logger.info('Slowest tests:')
        for record in slowest_tests:
            logger.info(
                '  %s [%s] %s',
                record['test_name'],
                record['status'],
                format_duration(float(record['duration_seconds'])),
            )


def main() -> int:
    if not __debug__:
        logger.error('The native test runner requires enabled Python assertions')
        return 2
    _ = os.environ.setdefault('POLARS_MAX_THREADS', '1')
    for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_NUM_THREADS', 'OMP_THREAD_LIMIT'):
        os.environ[variable] = '1'
    setup_cleanup_handlers()
    try:
        slowest_tests_limit = _runtime_slowest_tests_limit()
    except ValueError as exc:
        logger.error(str(exc))
        return 2

    plugin = RuntimeProfilePlugin()
    try:
        with worker_coverage() as directory:
            return_code = pytest.main(
                ['-p', 'tests.utils.proof_coverage', *_pytest_args(sys.argv[1:])],
                plugins=[plugin],
            )
            try:
                plugin.assert_complete()
                if directory is not None and plugin.parallel:
                    started = time.perf_counter()
                    try:
                        merge_worker_coverage(directory, plugin.workers)
                    finally:
                        plugin.combine_seconds = time.perf_counter() - started
            except (ValueError, OSError) as exc:
                plugin.reconciled = False
                if not plugin.errors:
                    plugin.errors.append(str(exc))
                logger.error('Incomplete parallel proof: %s', exc)
                return_code = return_code or 1
    finally:
        _write_profile(plugin, slowest_tests_limit)
        cleanup_csv_files()

    return int(return_code)


if __name__ == '__main__':
    raise SystemExit(main())

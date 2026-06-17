from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from tests.utils.cleanup import cleanup_csv_files
from tests.utils.cleanup import setup_cleanup_handlers
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
        self._records_by_nodeid: dict[str, dict[str, Any]] = {}

    def pytest_runtest_protocol(
        self,
        item: pytest.Item,
        nextitem: pytest.Item | None,
    ) -> None:
        self.test_started_at[item.nodeid] = utc_now_iso()

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when == 'call':
            self._record(report)
            return

        if report.when == 'setup' and (report.failed or report.skipped):
            self._record(report)
            return

        if report.when == 'teardown' and report.failed:
            self._record(report)

    def build_profile(self) -> dict[str, Any]:
        suite_finished_at = utc_now_iso()
        total_duration_seconds = time.perf_counter() - self.suite_started_clock
        return build_runtime_profile(
            test_records=self.test_records,
            suite_started_at=self.suite_started_at,
            suite_finished_at=suite_finished_at,
            total_duration_seconds=total_duration_seconds,
        )

    def _record(self, report: pytest.TestReport) -> None:
        status = _report_status(report)
        record = self._records_by_nodeid.get(report.nodeid)
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
            self._records_by_nodeid[report.nodeid] = record
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
    return ['tests']


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
    setup_cleanup_handlers()
    try:
        slowest_tests_limit = _runtime_slowest_tests_limit()
    except ValueError as exc:
        logger.error(str(exc))
        return 2

    plugin = RuntimeProfilePlugin()
    try:
        return_code = pytest.main(_pytest_args(sys.argv[1:]), plugins=[plugin])
    finally:
        _write_profile(plugin, slowest_tests_limit)
        cleanup_csv_files()

    return int(return_code)


if __name__ == '__main__':
    raise SystemExit(main())

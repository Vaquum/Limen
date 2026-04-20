from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    import logging


DEFAULT_SLOWEST_TESTS_LIMIT = 10
DEFAULT_THRESHOLD_BANDS_SECONDS = (1.0, 5.0, 10.0)

TestCallable = Callable[[], None]


@dataclass
class TestSuiteRunResult:
    exit_code: int
    failure_traceback: str | None
    profile: dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace(
        '+00:00',
        'Z',
    )


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f'{seconds:.3f}s'

    minutes, remaining_seconds = divmod(seconds, 60)
    return f'{int(minutes)}m {remaining_seconds:05.3f}s'


def build_runtime_profile(
    test_records: list[dict[str, Any]],
    suite_started_at: str,
    suite_finished_at: str,
    total_duration_seconds: float,
) -> dict[str, Any]:
    passed_count = sum(record['status'] == 'passed' for record in test_records)
    failed_count = sum(record['status'] == 'failed' for record in test_records)

    return {
        'schema_version': 1,
        'suite': {
            'started_at': suite_started_at,
            'finished_at': suite_finished_at,
            'duration_seconds': round(total_duration_seconds, 6),
            'test_count': len(test_records),
            'passed_count': passed_count,
            'failed_count': failed_count,
        },
        'tests': test_records,
    }


def write_runtime_profile(profile: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, indent=2) + '\n',
        encoding='utf-8',
    )


def load_runtime_profile(profile_path: Path) -> dict[str, Any]:
    profile = json.loads(profile_path.read_text(encoding='utf-8'))

    if 'suite' not in profile or 'tests' not in profile:
        raise ValueError('runtime profile must include suite and tests sections')

    return profile


def load_runtime_budget(budget_path: Path) -> dict[str, Any]:
    budget = json.loads(budget_path.read_text(encoding='utf-8'))

    max_total_seconds = float(budget['max_total_seconds'])
    if max_total_seconds <= 0:
        raise ValueError('max_total_seconds must be positive')

    slowest_tests_limit = int(budget.get('slowest_tests_limit', DEFAULT_SLOWEST_TESTS_LIMIT))
    if slowest_tests_limit <= 0:
        raise ValueError('slowest_tests_limit must be positive')

    threshold_bands = normalize_threshold_bands(
        budget.get('threshold_bands_seconds'),
    )

    budget['max_total_seconds'] = max_total_seconds
    budget['slowest_tests_limit'] = slowest_tests_limit
    budget['threshold_bands_seconds'] = list(threshold_bands)
    return budget


def normalize_threshold_bands(
    threshold_bands: list[float] | tuple[float, ...] | None,
) -> tuple[float, ...]:
    if threshold_bands is None:
        return DEFAULT_THRESHOLD_BANDS_SECONDS

    normalized = tuple(float(band) for band in threshold_bands)
    if not normalized:
        raise ValueError('threshold_bands_seconds must not be empty')

    if any(band <= 0 for band in normalized):
        raise ValueError('threshold_bands_seconds values must be positive')

    return normalized


def sorted_test_records(
    profile: dict[str, Any],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    records = sorted(
        profile['tests'],
        key=lambda record: float(record['duration_seconds']),
        reverse=True,
    )

    if limit is None:
        return records

    return records[:limit]


def count_tests_above_thresholds(
    profile: dict[str, Any],
    threshold_bands: list[float] | tuple[float, ...] | None = None,
) -> list[tuple[float, int]]:
    normalized_bands = normalize_threshold_bands(threshold_bands)
    records = profile['tests']

    return [
        (
            band,
            sum(float(record['duration_seconds']) > band for record in records),
        )
        for band in normalized_bands
    ]


def evaluate_runtime_budget(
    profile: dict[str, Any],
    budget: dict[str, Any],
) -> dict[str, Any]:
    observed_total_seconds = float(profile['suite']['duration_seconds'])
    max_total_seconds = float(budget['max_total_seconds'])
    overage_seconds = max(0.0, observed_total_seconds - max_total_seconds)

    return {
        'within_budget': observed_total_seconds <= max_total_seconds,
        'observed_total_seconds': observed_total_seconds,
        'max_total_seconds': max_total_seconds,
        'overage_seconds': overage_seconds,
    }


def render_runtime_summary_markdown(
    profile: dict[str, Any],
    budget: dict[str, Any] | None = None,
) -> str:
    suite = profile['suite']
    threshold_bands = (
        budget['threshold_bands_seconds']
        if budget is not None
        else list(DEFAULT_THRESHOLD_BANDS_SECONDS)
    )
    slowest_tests_limit = (
        int(budget['slowest_tests_limit'])
        if budget is not None
        else DEFAULT_SLOWEST_TESTS_LIMIT
    )

    lines = [
        '## Test Runtime',
        '',
        f"- Total runtime: `{format_duration(float(suite['duration_seconds']))}`",
        f"- Tests recorded: `{suite['test_count']}`",
        f"- Passed: `{suite['passed_count']}`",
        f"- Failed: `{suite['failed_count']}`",
    ]

    if budget is not None:
        verdict = evaluate_runtime_budget(profile, budget)
        status = 'PASS' if verdict['within_budget'] else 'FAIL'
        delta_label = 'Headroom' if verdict['within_budget'] else 'Overage'
        delta_seconds = (
            verdict['max_total_seconds'] - verdict['observed_total_seconds']
            if verdict['within_budget']
            else verdict['overage_seconds']
        )
        lines.extend([
            f"- Budget status: `{status}`",
            f"- Budget ceiling: `{format_duration(verdict['max_total_seconds'])}`",
            f"- {delta_label}: `{format_duration(abs(delta_seconds))}`",
        ])

    lines.extend([
        '',
        '### Threshold Counts',
        '',
        '| Threshold | Count |',
        '| --- | ---: |',
    ])

    for band, count in count_tests_above_thresholds(profile, threshold_bands):
        lines.append(f'| > {band:g}s | {count} |')

    lines.extend([
        '',
        f'### Slowest Tests (Top {slowest_tests_limit})',
        '',
        '| Test | Status | Duration |',
        '| --- | --- | ---: |',
    ])

    for record in sorted_test_records(profile, limit=slowest_tests_limit):
        test_name = str(record['test_name']).replace('|', '\\|')
        lines.append(
            '| '
            f'{test_name} | {record["status"]} | '
            f'{format_duration(float(record["duration_seconds"]))} |'
        )

    return '\n'.join(lines)


def write_runtime_summary(summary_file: Path, summary_markdown: str) -> None:
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open('a', encoding='utf-8') as summary_handle:
        summary_handle.write(summary_markdown + '\n')


def execute_test_suite(
    tests: list[TestCallable],
    logger: logging.Logger,
    profile_output_path: Path | None = None,
    slowest_tests_limit: int = DEFAULT_SLOWEST_TESTS_LIMIT,
) -> TestSuiteRunResult:
    suite_started_at = utc_now_iso()
    suite_started_clock = time.perf_counter()
    test_records: list[dict[str, Any]] = []
    failure_traceback: str | None = None

    for test in tests:
        test_started_at = utc_now_iso()
        test_started_clock = time.perf_counter()

        try:
            test()
            status = 'passed'
            error_message = None

        except Exception as exc:
            status = 'failed'
            error_message = f'{type(exc).__name__}: {exc}'
            failure_traceback = traceback.format_exc()

        duration_seconds = time.perf_counter() - test_started_clock
        test_finished_at = utc_now_iso()

        test_record = {
            'test_name': test.__name__,
            'module': test.__module__,
            'status': status,
            'duration_seconds': round(duration_seconds, 6),
            'started_at': test_started_at,
            'finished_at': test_finished_at,
        }
        if error_message is not None:
            test_record['error'] = error_message

        test_records.append(test_record)

        if status == 'passed':
            logger.info('✅ %s: PASSED (%.3fs)', test.__name__, duration_seconds)
            continue

        logger.error(
            '❌ %s: FAILED (%.3fs) - %s',
            test.__name__,
            duration_seconds,
            error_message,
        )
        break

    total_duration_seconds = time.perf_counter() - suite_started_clock
    suite_finished_at = utc_now_iso()
    profile = build_runtime_profile(
        test_records=test_records,
        suite_started_at=suite_started_at,
        suite_finished_at=suite_finished_at,
        total_duration_seconds=total_duration_seconds,
    )

    if profile_output_path is not None:
        write_runtime_profile(profile, profile_output_path)
        logger.info('Runtime profile written to %s', profile_output_path)

    suite = profile['suite']
    logger.info(
        'Test suite runtime: %s across %d tests (%d passed, %d failed)',
        format_duration(float(suite['duration_seconds'])),
        suite['test_count'],
        suite['passed_count'],
        suite['failed_count'],
    )

    slowest_tests = sorted_test_records(profile, limit=slowest_tests_limit)
    if slowest_tests:
        logger.info(
            'Slowest tests (top %d):',
            min(slowest_tests_limit, len(slowest_tests)),
        )
        for record in slowest_tests:
            logger.info(
                '  %s [%s] %s',
                record['test_name'],
                record['status'],
                format_duration(float(record['duration_seconds'])),
            )

    exit_code = 1 if failure_traceback is not None else 0
    return TestSuiteRunResult(
        exit_code=exit_code,
        failure_traceback=failure_traceback,
        profile=profile,
    )

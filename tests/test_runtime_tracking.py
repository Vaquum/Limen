import json
import logging
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.utils.runtime_tracking import execute_test_suite
from tests.utils.runtime_tracking import load_runtime_profile
from tests.utils.runtime_tracking import write_runtime_summary


def _make_runtime_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def test_execute_test_suite_writes_profile_and_stops_on_first_failure() -> None:
    execution_order: list[str] = []

    def passing_test() -> None:
        execution_order.append('passing')

    def failing_test() -> None:
        execution_order.append('failing')
        raise RuntimeError('boom')

    def trailing_test() -> None:
        execution_order.append('trailing')

    with TemporaryDirectory() as tmpdir:
        profile_path = Path(tmpdir) / 'runtime_profile.json'
        result = execute_test_suite(
            tests=[passing_test, failing_test, trailing_test],
            logger=_make_runtime_logger('tests.runtime_tracking.failure'),
            profile_output_path=profile_path,
            slowest_tests_limit=2,
        )
        profile = load_runtime_profile(profile_path)

    assert result.exit_code == 1
    assert result.failure_traceback is not None
    assert 'RuntimeError: boom' in result.failure_traceback
    assert execution_order == ['passing', 'failing']
    assert profile['suite']['test_count'] == 2
    assert profile['suite']['passed_count'] == 1
    assert profile['suite']['failed_count'] == 1
    assert profile['tests'][1]['status'] == 'failed'
    assert profile['tests'][1]['error'] == 'RuntimeError: boom'


def test_runtime_gate_writes_summary_and_passes_when_within_budget() -> None:
    profile = {
        'schema_version': 1,
        'suite': {
            'started_at': '2026-04-20T07:00:00.000Z',
            'finished_at': '2026-04-20T07:00:03.500Z',
            'duration_seconds': 3.5,
            'test_count': 2,
            'passed_count': 2,
            'failed_count': 0,
        },
        'tests': [
            {
                'test_name': 'test_fast',
                'module': 'tests.test_runtime_tracking',
                'status': 'passed',
                'duration_seconds': 0.5,
                'started_at': '2026-04-20T07:00:00.000Z',
                'finished_at': '2026-04-20T07:00:00.500Z',
            },
            {
                'test_name': 'test_slow',
                'module': 'tests.test_runtime_tracking',
                'status': 'passed',
                'duration_seconds': 3.0,
                'started_at': '2026-04-20T07:00:00.500Z',
                'finished_at': '2026-04-20T07:00:03.500Z',
            },
        ],
    }
    budget = {
        'schema_version': 1,
        'max_total_seconds': 5,
        'slowest_tests_limit': 2,
        'threshold_bands_seconds': [1, 5, 10],
    }

    with TemporaryDirectory() as tmpdir:
        profile_path = Path(tmpdir) / 'runtime_profile.json'
        budget_path = Path(tmpdir) / 'runtime_budget.json'
        summary_path = Path(tmpdir) / 'runtime_summary.md'

        profile_path.write_text(json.dumps(profile), encoding='utf-8')
        budget_path.write_text(json.dumps(budget), encoding='utf-8')

        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                '-m',
                'tests.runtime_gate',
                '--profile',
                str(profile_path),
                '--budget',
                str(budget_path),
                '--summary-file',
                str(summary_path),
                '--enforce',
            ],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=False,
        )

        summary = summary_path.read_text(encoding='utf-8')

    assert result.returncode == 0
    assert 'Runtime budget passed' in result.stdout
    assert 'Budget status: `PASS`' in summary
    assert '| > 1s | 1 |' in summary
    assert 'test_slow' in summary


def test_runtime_gate_fails_when_profile_exceeds_budget() -> None:
    profile = {
        'schema_version': 1,
        'suite': {
            'started_at': '2026-04-20T07:00:00.000Z',
            'finished_at': '2026-04-20T07:00:06.200Z',
            'duration_seconds': 6.2,
            'test_count': 1,
            'passed_count': 1,
            'failed_count': 0,
        },
        'tests': [
            {
                'test_name': 'test_over_budget',
                'module': 'tests.test_runtime_tracking',
                'status': 'passed',
                'duration_seconds': 6.2,
                'started_at': '2026-04-20T07:00:00.000Z',
                'finished_at': '2026-04-20T07:00:06.200Z',
            },
        ],
    }
    budget = {
        'schema_version': 1,
        'max_total_seconds': 5,
        'slowest_tests_limit': 10,
        'threshold_bands_seconds': [1, 5, 10],
    }

    with TemporaryDirectory() as tmpdir:
        profile_path = Path(tmpdir) / 'runtime_profile.json'
        budget_path = Path(tmpdir) / 'runtime_budget.json'

        profile_path.write_text(json.dumps(profile), encoding='utf-8')
        budget_path.write_text(json.dumps(budget), encoding='utf-8')

        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                '-m',
                'tests.runtime_gate',
                '--profile',
                str(profile_path),
                '--budget',
                str(budget_path),
                '--enforce',
            ],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=False,
        )

    assert result.returncode == 1
    assert 'Runtime budget exceeded' in result.stderr


def test_write_runtime_summary_appends_to_existing_summary_file() -> None:
    with TemporaryDirectory() as tmpdir:
        summary_path = Path(tmpdir) / 'runtime_summary.md'
        summary_path.write_text('Existing section\n', encoding='utf-8')

        write_runtime_summary(summary_path, '## Test Runtime\n\n- Total runtime: `1.000s`')

        summary = summary_path.read_text(encoding='utf-8')

    assert summary == 'Existing section\n## Test Runtime\n\n- Total runtime: `1.000s`\n'

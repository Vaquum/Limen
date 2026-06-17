from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]


def _run_profiled_test_module(source: str) -> tuple[subprocess.CompletedProcess[str], dict]:
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_path = tmp_path / 'test_profiled.py'
        profile_path = tmp_path / 'runtime_profile.json'
        test_path.write_text(source, encoding='utf-8')

        env = dict(os.environ)
        env['LIMEN_RUNTIME_PROFILE_PATH'] = str(profile_path)

        result = subprocess.run(
            [sys.executable, '-m', 'tests.run', str(test_path), '-q'],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        profile = json.loads(profile_path.read_text(encoding='utf-8'))

    return result, profile


def test_tests_run_delegates_to_pytest_collection() -> None:
    source = (ROOT / 'tests' / 'run.py').read_text(encoding='utf-8')

    assert 'pytest.main' in source
    assert 'execute_test_suite' not in source
    assert 'from tests.test_' not in source


def test_tests_run_writes_pytest_runtime_profile() -> None:
    with TemporaryDirectory() as tmpdir:
        profile_path = Path(tmpdir) / 'runtime_profile.json'
        env = dict(os.environ)
        env['LIMEN_RUNTIME_PROFILE_PATH'] = str(profile_path)

        result = subprocess.run(
            [
                sys.executable,
                '-m',
                'tests.run',
                'tests/test_runtime_tracking.py::test_write_runtime_summary_appends_to_existing_summary_file',
                '-q',
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        profile = json.loads(profile_path.read_text(encoding='utf-8'))

    assert result.returncode == 0, result.stderr
    assert profile['suite']['test_count'] == 1
    assert profile['suite']['passed_count'] == 1
    assert profile['suite']['failed_count'] == 0
    assert profile['tests'][0]['test_name'].endswith(
        'test_write_runtime_summary_appends_to_existing_summary_file',
    )


def test_tests_run_counts_call_and_teardown_failure_once() -> None:
    result, profile = _run_profiled_test_module(
        """
import pytest


@pytest.fixture
def failing_teardown():
    yield
    raise RuntimeError('teardown failed')


def test_passes_then_teardown_fails(failing_teardown):
    assert True
""",
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert profile['suite']['test_count'] == 1
    assert profile['suite']['passed_count'] == 0
    assert profile['suite']['failed_count'] == 1
    assert len(profile['tests']) == 1
    assert profile['tests'][0]['status'] == 'failed'


def test_tests_run_counts_setup_and_teardown_failure_once() -> None:
    result, profile = _run_profiled_test_module(
        """
import pytest


@pytest.fixture
def prepared_resource():
    yield
    raise RuntimeError('teardown failed')


@pytest.fixture
def failing_setup():
    raise RuntimeError('setup failed')


def test_setup_fails_after_resource_setup(prepared_resource, failing_setup):
    raise AssertionError('unreachable')
""",
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert profile['suite']['test_count'] == 1
    assert profile['suite']['passed_count'] == 0
    assert profile['suite']['failed_count'] == 1
    assert len(profile['tests']) == 1
    assert profile['tests'][0]['status'] == 'failed'


def test_tests_run_rejects_invalid_slowest_limit_before_pytest() -> None:
    with TemporaryDirectory() as tmpdir:
        profile_path = Path(tmpdir) / 'runtime_profile.json'
        env = dict(os.environ)
        env['LIMEN_RUNTIME_PROFILE_PATH'] = str(profile_path)
        env['LIMEN_RUNTIME_SLOWEST_LIMIT'] = 'invalid'

        result = subprocess.run(
            [
                sys.executable,
                '-m',
                'tests.run',
                'tests/test_runtime_tracking.py::test_write_runtime_summary_appends_to_existing_summary_file',
                '-q',
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        profile_exists = profile_path.exists()

    assert result.returncode == 2
    assert 'LIMEN_RUNTIME_SLOWEST_LIMIT must be a positive integer' in result.stderr
    assert not profile_exists

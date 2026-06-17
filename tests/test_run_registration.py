from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]


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

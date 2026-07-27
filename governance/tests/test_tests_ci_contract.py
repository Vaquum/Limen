from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_WORKFLOW = REPO_ROOT / '.github/workflows/pr_checks_tests.yml'
EXPECTED_TEST_COMMAND = 'python -m coverage run -m tests.run'


def test_pr_checks_tests_workflow_exists() -> None:
    assert TESTS_WORKFLOW.exists()


def test_pr_checks_tests_pins_python_and_runtime_suite_command() -> None:
    workflow = TESTS_WORKFLOW.read_text(encoding='utf-8')

    assert "python-version: '3.10'" in workflow
    # The suite needs the full research stack, so it installs the
    # research-env set rather than the runtime/build pair upstream uses.
    assert 'pip install --require-hashes -r requirements/ci/research-env.txt' in workflow
    assert 'pip install --no-deps -e .' in workflow
    assert EXPECTED_TEST_COMMAND in workflow
    # No soft-fail pathway in any job that is a required check. The one
    # `continue-on-error` in this workflow is on the artifact download in
    # `publish_coverage_pr_comment`, which is informational and must not fail
    # the PR when the upstream test job produced no artifact; the job then
    # checks for the file and skips. Asserting per-job keeps that allowance
    # from silently widening to a gate.
    payload = yaml.safe_load(workflow)
    gating_jobs = {
        name: job for name, job in payload['jobs'].items()
        if name != 'publish_coverage_pr_comment'
    }
    assert gating_jobs
    for name, job in gating_jobs.items():
        assert 'continue-on-error' not in job, name
        for step in job.get('steps', []):
            assert 'continue-on-error' not in step, (name, step.get('name'))

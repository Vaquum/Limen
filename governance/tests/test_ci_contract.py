from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
READINESS_WORKFLOW = REPO_ROOT / '.github/workflows/pr_merge_readiness.yml'
RULESET_SNAPSHOT = REPO_ROOT / '.github/rulesets/main.json'


def test_post_merge_changelog_workflow_removed() -> None:
    assert not (REPO_ROOT / '.github/workflows/pr_post_changelog.yml').exists()


def test_slice_workflows_removed() -> None:
    workflows = REPO_ROOT / '.github/workflows'
    for name in (
        'pr_checks_slice.yml',
        'pr_checks_slice_on_issue.yml',
        'pr_checks_slice_sweep.yml',
        'slice_closeout_guard.yml',
    ):
        assert not (workflows / name).exists(), name
    assert not (REPO_ROOT / '.github/ISSUE_TEMPLATE/slice.yml').exists()
    assert not (REPO_ROOT / 'governance/slice_gate.py').exists()
    assert not (REPO_ROOT / 'governance/interface_contract.py').exists()


def test_prd_workflow_contract() -> None:
    workflow = (REPO_ROOT / '.github/workflows/pr_checks_prd.yml').read_text(encoding='utf-8')

    assert 'name: pr_checks_prd' in workflow
    assert 'types: [opened, edited, synchronize, reopened, ready_for_review]' in workflow
    assert 'governance/prd_gate.py' in workflow
    assert '--pr-title' in workflow
    assert '--pr-body-file' in workflow
    assert '--repo' in workflow
    assert 'issues: read' in workflow
    # No soft-fail pathway: no `|| true`, no continue-on-error on any step.
    assert '|| true' not in workflow
    assert 'continue-on-error' not in workflow


def test_prd_on_issue_workflow_contract() -> None:
    workflow = (REPO_ROOT / '.github/workflows/pr_checks_prd_on_issue.yml').read_text(encoding='utf-8')

    assert 'name: pr_checks_prd_on_issue' in workflow
    assert 'types: [edited, labeled, unlabeled, closed, reopened, deleted]' in workflow
    assert 'governance/prd_gate.py' in workflow
    # Same-parser scan: linked PRs are found with the gate's own
    # closing-keyword regex (a GraphQL-first path misses deleted
    # issues, the gap this workflow exists to close).
    assert 'CLOSING_KEYWORD_RE' in workflow
    assert 'closedByPullRequestsReferences' in workflow
    # Rerun-first delivery, and rerun-ONLY: no check-run POSTs means
    # no API-posted layer to reconcile.
    assert 'actions/runs/$RUN_ID/rerun' in workflow
    # Canonical runs are matched on branch and repository, never by
    # head SHA alone (two PRs can share a commit).
    assert 'headRefName' in workflow
    assert 'head_branch == $branch' in workflow
    assert 'head_repository.full_name == $repo' in workflow
    assert 'actions: write' in workflow
    assert 'checks: write' not in workflow
    assert 'check-runs"' not in workflow
    assert 'check-runs?' not in workflow
    # Quiescence before comparison (an in-flight run may have read
    # the issue before the event); a moved head is left to its own
    # run; the rerun is retried; failures fail loud.
    assert 'quiescence' in workflow
    assert 'seq 1 21' in workflow
    assert 'continue 2' in workflow
    assert 'for attempt in 1 2 3' in workflow
    assert 'after 3 attempts' in workflow
    assert 'moved' in workflow
    assert 'could not be revalidated' in workflow
    assert 'done < <(jq' in workflow
    assert '|| true' not in workflow
    assert 'continue-on-error' not in workflow


def test_merge_readiness_workflow_contract() -> None:
    workflow = READINESS_WORKFLOW.read_text(encoding='utf-8')

    # pull_request_review_thread is a webhook event only, not an
    # Actions trigger — the defect that stalled the downstream first
    # delivery; comment activity and suite completions refresh instead.
    # Matched as a trigger key so the workflow's own explanatory
    # comment does not satisfy the assertion.
    assert '\n  pull_request_review_thread:' not in workflow
    assert 'pull_request_review:' in workflow
    assert 'pull_request_review_comment:' in workflow
    assert 'check_suite:' in workflow
    assert '<!-- merge-readiness -->' in workflow

    # Queued, not cancelled. The group exists to serialise the
    # read-then-create on the marker comment, and cancelling is not
    # serialising -- it kills the earlier run, which then surfaces as a
    # failed check on the PR this workflow exists to report on.
    parsed = yaml.safe_load(workflow)
    assert parsed['concurrency']['cancel-in-progress'] is False
    assert 'required-check inventory unavailable (fail-closed)' in workflow
    assert 'pull-requests: write' in workflow
    # One concurrency lane per PR across every event type, so parallel
    # runs cannot race the read-then-create on the marker comment.
    expected_group = (
        '${{ github.workflow }}-'
        '${{ github.event.pull_request.number || '
        'github.event.check_suite.pull_requests[0].number || '
        'github.event.check_suite.head_sha || github.run_id }}'
    )
    assert parsed['concurrency']['group'] == expected_group, (
        'readiness events must share one PR concurrency lane'
    )
    # This snapshot excludes the reporter. Limen's separate classic
    # protection still requires it and can block a merge on cancellation.
    snapshot = json.loads(RULESET_SNAPSHOT.read_text(encoding='utf-8'))
    for rule in snapshot['rules']:
        if rule['type'] == 'required_status_checks':
            contexts = [c['context'] for c in rule['parameters']['required_status_checks']]
            assert 'pr_merge_readiness' not in contexts


def test_merge_readiness_rejects_event_specific_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A review event must not acquire a separate marker-comment lane."""
    workflow = READINESS_WORKFLOW.read_text(encoding='utf-8')
    group = yaml.safe_load(workflow)['concurrency']['group']
    assert workflow.count(group) == 1
    mutated = workflow.replace(group, group + '-${{ github.event_name }}', 1)
    path = tmp_path / 'pr_merge_readiness.yml'
    path.write_text(mutated, encoding='utf-8')
    monkeypatch.setattr(sys.modules[__name__], 'READINESS_WORKFLOW', path)

    with pytest.raises(AssertionError, match='one PR concurrency lane'):
        test_merge_readiness_workflow_contract()


def test_update_changelog_script_removed() -> None:
    assert not (REPO_ROOT / 'governance/update_changelog.py').exists()


def test_typing_gate_setup_failures_exit_2() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / 'repo'
        (tmp / 'governance').mkdir(parents=True)
        for name in ('typing_gate.py', '_common.py'):
            shutil.copy2(REPO_ROOT / 'governance' / name, tmp / 'governance' / name)
        (tmp / 'pyproject.toml').write_text('not = [valid\n', encoding='utf-8')

        result = subprocess.run(
            [
                # sys.executable, not 'python3': the gate imports tomli on the
                # 3.10 floor, and only the interpreter running this suite has
                # the dependencies installed. A bare 'python3' picks up the
                # system interpreter, where the import fails and the gate exits
                # 1 on ModuleNotFoundError instead of 2 on the setup failure
                # this test is actually asserting.
                sys.executable,
                'governance/typing_gate.py',
                '--pyright-json',
                '/tmp/missing-pyright.json',
                '--bootstrap',
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=tmp,
        )

    assert result.returncode == 2
    assert result.stdout == ''
    assert 'typing_gate: cannot parse pyproject.toml:' in result.stderr

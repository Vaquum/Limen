from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SLICE_WORKFLOW = REPO_ROOT / '.github/workflows/pr_checks_slice.yml'
CLOSEOUT_GUARD_WORKFLOW = REPO_ROOT / '.github/workflows/slice_closeout_guard.yml'
ON_ISSUE_WORKFLOW = REPO_ROOT / '.github/workflows/pr_checks_slice_on_issue.yml'
SWEEP_WORKFLOW = REPO_ROOT / '.github/workflows/pr_checks_slice_sweep.yml'
READINESS_WORKFLOW = REPO_ROOT / '.github/workflows/pr_merge_readiness.yml'
RULESET_SNAPSHOT = REPO_ROOT / '.github/rulesets/main.json'


def test_post_merge_changelog_workflow_removed() -> None:
    assert not (REPO_ROOT / '.github/workflows/pr_post_changelog.yml').exists()


def test_slice_closeout_guard_workflow_contract() -> None:
    workflow = CLOSEOUT_GUARD_WORKFLOW.read_text(encoding='utf-8')

    assert 'issues:\n    types: [closed]' in workflow
    assert "if: contains(github.event.issue.labels.*.name, 'slice')" in workflow
    assert 'issues: write' in workflow
    assert 'closedByPullRequestsReferences' in workflow
    # Evidence quality: only successful check runs may become closeout
    # evidence — a failed required check must fail the writer loud.
    assert "select(.conclusion == \"success\")" in workflow
    # An appended duplicate Done Means section must not receive (or
    # shadow) evidence, and a hand-typed Merge SHA on a no-PR close
    # must be reachable from main to count.
    assert 'expected exactly one Done Means section' in workflow
    assert 'compare/main...$CLAIMED' in workflow
    assert r"r'^##+ Done Means\b.*?^##+ Author Checks\b'" in workflow
    # Fill: a merged closing PR gets the evidence fields written in place.
    assert "if: steps.closing_pr.outputs.pr_number != ''" in workflow
    assert 'gh issue edit "$ISSUE_NUMBER" --repo "$GITHUB_REPOSITORY" --body-file new_body.txt' in workflow
    # Verify: an incomplete-evidence close with no merged PR, or any
    # guard failure, reopens the issue instead of letting the close
    # stand.
    assert 'gh issue reopen' in workflow
    assert 'empty Merged PR number' in workflow
    assert 'empty required-run list' in workflow
    assert 'if: failure()\n' in workflow


def test_slice_on_issue_workflow_contract() -> None:
    workflow = ON_ISSUE_WORKFLOW.read_text(encoding='utf-8')

    # Rerun-first delivery: heal the canonical pull_request run instead
    # of stacking parallel check-runs; the API POST stays only as the
    # fail-closed fallback.
    assert 'actions/runs/$RUN_ID/rerun' in workflow
    assert 'falling back to check-run POST' in workflow
    assert 'actions: write' in workflow
    assert 'check-runs?check_name=pr_checks_slice' in workflow
    # API-posted reconciliation: an API-posted check-run outranks the
    # workflow-run entries of the same name and only a newer POST
    # supersedes it (Vaquum/Limen PR #757), so a disagreeing latest
    # API-posted check-run always draws a superseding POST.
    assert 'check_suite_id' in workflow
    assert 'LATEST_POSTED' in workflow
    # Rule 9 staleness: the affected set includes the parent PRD's
    # slice sub-issue siblings, and a parent-lookup failure fails loud
    # instead of silently shrinking the set.
    assert 'sub_issues' in workflow
    assert 'refusing to compute a sibling set that may be incomplete' in workflow
    # SIGPIPE-safe truncation: parameter expansion, never a pipe.
    assert 'SUMMARY=${SUMMARY:0:60000}' in workflow
    assert '"$GATE_OUT" | head -c' not in workflow


def test_slice_sweep_workflow_contract() -> None:
    workflow = SWEEP_WORKFLOW.read_text(encoding='utf-8')

    assert "- cron: '17 */6 * * *'" in workflow
    assert 'workflow_dispatch:' in workflow
    # A dispatch from another ref must still execute main's gate.
    assert 'ref: main' in workflow
    # Killing a sweep mid-posting would strand some PRs a full interval.
    assert 'cancel-in-progress: false' in workflow
    # Main-only authority: the sweep publishes its own verdict against
    # the latest check-run and never reruns the pull_request workflow —
    # a rerun would re-execute the gate from the judged PR's merge ref
    # and let the PR self-attest.
    assert 'check-runs?check_name=pr_checks_slice' in workflow
    assert 'self-attest' in workflow
    assert 'actions/runs/$RUN_ID/rerun' not in workflow
    assert 'actions: write' not in workflow
    # API-posted reconciliation (Vaquum/Limen PR #757): a disagreeing
    # latest API-posted check-run always draws a superseding POST, and
    # classifying check-runs by canonical check suite needs actions:
    # read on the token.
    assert 'check_suite_id' in workflow
    assert 'LATEST_POSTED' in workflow
    assert 'actions: read' in workflow
    assert 'file enumeration incomplete' in workflow
    assert 'SUMMARY=${SUMMARY:0:60000}' in workflow
    assert '--require-hashes -r requirements/ci/gate-tools.txt' in workflow


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
    # Informational only: never itself a required context, so it cannot
    # deadlock the merge it reports on.
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


def test_slice_workflows_validate_the_captured_head() -> None:
    for path in (SLICE_WORKFLOW, ON_ISSUE_WORKFLOW, SWEEP_WORKFLOW):
        workflow = path.read_text(encoding='utf-8')
        assert '--json title,body,changedFiles,headRefOid' in workflow
        assert '--head-sha ' in workflow
        assert 'PR head changed during capture' in workflow or 'no stale verdict posted' in workflow
    # Privileged rechecks keep validator execution on the default branch.
    assert 'ref: main' in SWEEP_WORKFLOW.read_text(encoding='utf-8')


def test_parent_edit_rechecks_child_only_pr(tmp_path) -> None:
    from governance.interface_contract import interface_contract_failures

    workflow = yaml.safe_load(ON_ISSUE_WORKFLOW.read_text(encoding='utf-8'))
    steps = workflow['jobs']['rerun_slice_gate_for_linked_prs']['steps']
    script = next(step['run'] for step in steps if step.get('name') == 'Find linked PRs')
    script = script.replace('${{ github.repository }}', 'Vaquum/Limen')
    script = script.replace('${{ github.event.issue.number }}', '1006')
    script = script.replace('${{ github.event.action }}', 'edited')
    (tmp_path / 'governance').mkdir()
    shutil.copyfile(REPO_ROOT / 'governance/_common.py', tmp_path / 'governance/_common.py')
    # Model the published native parent/child graph; gh --jq returns child ids.
    cli = tmp_path / 'gh'
    cli.write_text(
        f'#!{sys.executable}\n'
        'import json, sys\n'
        'args = sys.argv[1:]\n'
        'if args[:2] == ["api", "graphql"]: print("null")\n'
        'elif args[0] == "api" and args[1].endswith("/1006/sub_issues"): print("1007")\n'
        'elif args[:2] == ["pr", "list"]: print(json.dumps(['
        '{"number": 1010, "body": "Closes: #1007", "headRefOid": "a" * 40},'
        '{"number": 1011, "body": "Closes #1008", "headRefOid": "b" * 40}]))\n'
        'else: raise SystemExit("Unexpected gh input: " + repr(args))\n',
        encoding='utf-8',
    )
    cli.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ['PATH'],
               GITHUB_OUTPUT=str(tmp_path / 'output'))
    result = subprocess.run(['bash', '-c', script], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / 'linked_prs.json').read_text()) == [
        {'number': 1010, 'headRefOid': 'a' * 40},
    ]

    def contract(interface):
        return ('## Interface Contract\n```yaml\ndelivery: public\n'
                f'interfaces: [{interface}]\nscope: "One complete capability."\nproof:\n'
                f'  {interface}: [test:tests/test_capability.py::test_capability]\n```\n')

    child = contract('python')
    options = {'closing_parent': False, 'read_source': lambda _: 'def test_capability():\n    assert True\n',
               'required_checks': set()}
    assert not interface_contract_failures(child, child, parent_body=child, **options)
    assert interface_contract_failures(child, child, parent_body=contract('compiled'), **options)

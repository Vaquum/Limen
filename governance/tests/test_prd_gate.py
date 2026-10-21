from __future__ import annotations

import pytest

from governance import prd_gate


def _issue(
    labels: list[str] | None = None,
    state: str = 'open',
    is_pull_request: bool = False,
) -> dict[str, object]:
    return {
        'state': state,
        'labels': ['bug'] if labels is None else labels,
        'is_pull_request': is_pull_request,
    }


def test_gate_accepts_feat_closing_open_prd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        prd_gate, 'fetch_issue', lambda _repo, _number: _issue(labels=['prd'])
    )

    failures = prd_gate.gate(
        'feat: add template listing to Limen CLI',
        'Closes #9',
        'Vaquum/Limen',
    )
    assert failures == []


def test_gate_accepts_feat_scoped_and_breaking_closing_prd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prd_gate, 'fetch_issue', lambda _repo, _number: _issue(labels=['prd'])
    )

    assert prd_gate.gate('feat(cli): add template listing', 'Closes #9', 'Vaquum/Limen') == []
    assert prd_gate.gate('feat!: drop the legacy manifest', 'Closes #9', 'Vaquum/Limen') == []


def test_gate_accepts_non_feat_closing_plain_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prd_gate, 'fetch_issue', lambda _repo, _number: _issue())

    for title in (
        'fix: bound the version read',
        'test: cover the empty manifest',
        'chore: rotate the signing key',
        'docs: clarify the rollout runbook',
        'refactor: split the manifest parser',
    ):
        assert prd_gate.gate(title, 'Closes #9', 'Vaquum/Limen') == [], title


def test_gate_accepts_non_feat_closing_prd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        prd_gate, 'fetch_issue', lambda _repo, _number: _issue(labels=['prd'])
    )

    assert prd_gate.gate('fix: bound the version read', 'Closes #9', 'Vaquum/Limen') == []


def test_gate_rejects_feat_closing_non_prd_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prd_gate, 'fetch_issue', lambda _repo, _number: _issue())

    failures = prd_gate.gate(
        'feat: add template listing to Limen CLI',
        'Closes #9',
        'Vaquum/Limen',
    )
    assert failures == [
        "PR title 'feat: add template listing to Limen CLI' is `feat` but issue "
        "#9 does not carry the `prd` label (labels: ['bug']). A `feat` PR must "
        'close a PRD filed via `.github/ISSUE_TEMPLATE/prd.yml` (rule 4).'
    ]


def test_gate_rejects_missing_closing_reference() -> None:
    failures = prd_gate.gate(
        'feat: add template listing to Limen CLI',
        'A body with no closing reference at all.',
        'Vaquum/Limen',
    )
    assert failures == [
        'PR body has no closing reference. Every PR must close exactly one OPEN '
        'issue (`Closes #N`, or Fixes/Resolves). A `feat` PR closes a PRD; any '
        'other type closes any OPEN issue or a PRD (rule 1).'
    ]


def test_find_closing_references_deduplicates_in_order() -> None:
    assert prd_gate.find_closing_references('Closes #9\nFixes #12\nCloses #9') == [9, 12]


def test_gate_rejects_multiple_closing_references() -> None:
    failures = prd_gate.gate(
        'fix: bound the version read',
        'Closes #9\nFixes #10',
        'Vaquum/Limen',
    )
    assert failures == [
        'PR body has 2 closing references (#9, #10). Every PR must close exactly '
        'one OPEN issue (rule 1).'
    ]


def test_gate_rejects_unknown_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prd_gate, 'fetch_issue', lambda _repo, _number: None)

    failures = prd_gate.gate('fix: bound the version read', 'Closes #9', 'Vaquum/Limen')
    assert failures == [
        'issue #9 does not exist in Vaquum/Limen. The closing reference must '
        'point at a real OPEN issue.'
    ]


def test_gate_rejects_pr_number_used_as_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        prd_gate, 'fetch_issue', lambda _repo, _number: _issue(is_pull_request=True)
    )

    failures = prd_gate.gate('fix: bound the version read', 'Closes #8', 'Vaquum/Limen')
    assert failures == [
        '#8 is a pull request, not an issue. The closing reference must point '
        'at an OPEN issue.'
    ]


def test_gate_rejects_closed_issue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        prd_gate, 'fetch_issue', lambda _repo, _number: _issue(state='closed')
    )

    failures = prd_gate.gate('fix: bound the version read', 'Closes #9', 'Vaquum/Limen')
    assert failures == [
        "issue #9 state is 'closed'; must be OPEN for a PR to close it."
    ]


def test_gate_rejects_qualified_and_url_closing_references() -> None:
    failures = prd_gate.gate(
        'fix: bound the version read',
        'Closes #9\nCloses Vaquum/Limen#12',
        'Vaquum/Limen',
    )
    assert failures == [
        'PR body contains 1 qualified or URL closing reference(s) '
        "('Closes Vaquum/Limen#12'). GitHub honors these on merge but the gate "
        'cannot fold them into the validated closing set; use the bare '
        '`Closes #N` form (rule 1).'
    ]

    url_failures = prd_gate.gate(
        'fix: bound the version read',
        'Closes #9\nFixes https://github.com/Vaquum/Limen/issues/12',
        'Vaquum/Limen',
    )
    assert len(url_failures) == 1
    assert 'qualified or URL closing reference' in url_failures[0]


def test_gate_skips_prd_rule_when_title_is_not_conventional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Title format is cc_gate's concern. An unparseable title cannot be
    # classified as `feat`, so rule 4 is skipped here -- and the merge
    # stays blocked because cc_gate fails the same title.
    monkeypatch.setattr(prd_gate, 'fetch_issue', lambda _repo, _number: _issue())

    assert prd_gate.gate('not a conventional title', 'Closes #9', 'Vaquum/Limen') == []


def test_pr_type_is_feat_classifies_titles() -> None:
    assert prd_gate.pr_type_is_feat('feat: add listing')
    assert prd_gate.pr_type_is_feat('feat(cli): add listing')
    assert prd_gate.pr_type_is_feat('feat!: drop legacy')
    assert prd_gate.pr_type_is_feat('feat(cli)!: drop legacy')
    assert not prd_gate.pr_type_is_feat('fix: bound the read')
    assert not prd_gate.pr_type_is_feat('feature: add listing')
    assert not prd_gate.pr_type_is_feat('not conventional')

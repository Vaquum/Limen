from __future__ import annotations

from pathlib import Path

import pytest

from governance import slice_gate

SIGNIFICANCE_BLOCK = (
    '> **Significance.** This is the exact slice contract.\n'
    '> It must be preserved byte-for-byte.'
)

DONE_MEANS_COMPLETE = (
    '## Done Means\n'
    '- [x] Capability complete\n'
    '- [x] Tests complete\n\n'
    'Merge SHA:\n'
    'Merged PR number:\n'
    'Required CI runs (workflow name : run id):\n'
    '-\n\n'
)

AUTHOR_CHECKS = '## Author Checks\n- [x] Sections intact.\n'


def _template(tmp_path: Path) -> Path:
    path = tmp_path / 'slice.yml'
    path.write_text(
        'name: Slice\n'
        'body:\n'
        '  - type: textarea\n'
        '    attributes:\n'
        '      value: |\n'
        '        > **Significance.** This is the exact slice contract.\n'
        '        > It must be preserved byte-for-byte.\n',
        encoding='utf-8',
    )
    return path


def _issue(body: str, labels: list[str] | None = None) -> dict[str, object]:
    return {
        'title': 'feat: add law template',
        'state': 'open',
        'labels': ['slice'] if labels is None else labels,
        'body': body,
        'is_pull_request': False,
    }


def _body(
    out_of_scope: str = '- (none)',
    done_means: str = DONE_MEANS_COMPLETE,
) -> str:
    return (
        f'{SIGNIFICANCE_BLOCK}\n\n'
        '## Surfaces\n'
        '- `governance/**`\n'
        '- `.github/workflows/**`\n\n'
        '## Out of Scope\n'
        f'{out_of_scope}\n\n'
        f'{done_means}'
        f'{AUTHOR_CHECKS}'
    )


def _patch_graph(
    monkeypatch: pytest.MonkeyPatch,
    parent: int | None = None,
    open_slices: list[int] | None = None,
) -> None:
    monkeypatch.setattr(
        slice_gate, 'fetch_parent_issue_number', lambda _repo, _number: parent
    )
    monkeypatch.setattr(
        slice_gate,
        'fetch_open_slice_sub_issue_numbers',
        lambda _repo, _number: list(open_slices or []),
    )


def test_gate_accepts_open_slice_issue_with_matching_scope(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(slice_gate, 'fetch_issue', lambda _repo, _number: _issue(_body()))
    _patch_graph(monkeypatch)

    failures = slice_gate.gate(
        'feat: add law template',
        'Closes #9',
        ['governance/version_gate.py', '.github/workflows/pr_checks_version.yml'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
    )
    assert failures == []


def test_gate_rejects_pr_number_used_as_slice_issue(
    tmp_path: Path,
    monkeypatch,
) -> None:
    issue = _issue(_body())
    issue['is_pull_request'] = True
    monkeypatch.setattr(slice_gate, 'fetch_issue', lambda _repo, _number: issue)

    failures = slice_gate.gate(
        'feat: add law template',
        'Closes #8',
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
    )
    assert failures == [
        '#8 is a pull request, not an issue. The closing reference must point at an OPEN '
        'slice issue filed via the slice template at `.github/ISSUE_TEMPLATE/slice.yml`, '
        'not another PR.'
    ]


def test_gate_requires_significance_blockquote(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(slice_gate, 'fetch_issue', lambda _repo, _number: _issue(_body().replace(
        SIGNIFICANCE_BLOCK,
        '> **Significance.** Different words.',
    )))
    _patch_graph(monkeypatch)

    failures = slice_gate.gate(
        'feat: add law template',
        'Closes #9',
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
    )
    assert any('missing 1 of 1 full Significance blockquotes' in item for item in failures)


def test_gate_blocks_files_outside_surfaces(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(slice_gate, 'fetch_issue', lambda _repo, _number: _issue(_body()))
    _patch_graph(monkeypatch)

    failures = slice_gate.gate(
        'feat: add law template',
        'Closes #9',
        ['README.md'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
    )
    assert any('not listed in issue #9 Surfaces' in item for item in failures)


def test_gate_blocks_out_of_scope_even_when_surface_allows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        slice_gate,
        'fetch_issue',
        lambda _repo, _number: _issue(_body('- `governance/experimental.py`')),
    )
    _patch_graph(monkeypatch)

    failures = slice_gate.gate(
        'feat: add law template',
        'Closes #9',
        ['governance/experimental.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
    )
    assert any('listed in issue #9 Out of Scope' in item for item in failures)


def test_no_closing_reference_fails_with_closing_set_message(tmp_path: Path) -> None:
    failures = slice_gate.gate(
        'feat: add law template',
        'A body with no closing reference at all.',
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
    )
    assert failures == [
        'PR body has no closing reference. The closing set must be exactly the slice '
        'issue (`Closes #N`, or Fixes/Resolves, with N an OPEN slice-labelled issue), '
        'plus its parent PRD only when the slice is the parent\'s last open slice '
        'sub-issue (rule 9).'
    ]


def test_multiple_closing_references_fail_before_api_call(tmp_path: Path) -> None:
    failures = slice_gate.gate(
        'feat: add law template',
        'Closes #9\nFixes #10\nResolves #11',
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
    )
    assert failures == [
        'PR body has 3 closing references (#9, #10, #11). The closing set must be exactly '
        'the slice issue, plus its parent PRD only when the slice is the parent\'s last '
        'open slice sub-issue (rule 9).'
    ]


def test_rule_9_rejects_prd_close_with_open_siblings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    issues = {
        9: _issue(_body()),
        12: _issue(_body(), labels=['planning']),
    }
    monkeypatch.setattr(slice_gate, 'fetch_issue', lambda _repo, number: issues[number])
    _patch_graph(monkeypatch, parent=12, open_slices=[9, 10])

    failures = slice_gate.gate(
        'feat: add law template',
        'Closes #9\nCloses #12',
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
    )
    assert failures == [
        'closing set {#9, #12} must be exactly {#9} because parent PRD #12 still has '
        'other open slice sub-issues (#10) (rule 9).'
    ]


def test_rule_9_requires_prd_close_on_last_slice(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(slice_gate, 'fetch_issue', lambda _repo, _number: _issue(_body()))
    _patch_graph(monkeypatch, parent=12, open_slices=[9])

    failures = slice_gate.gate(
        'feat: add law template',
        'Closes #9',
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
    )
    assert failures == [
        'closing set {#9} must be exactly {#9, #12} because slice #9 is parent PRD '
        '#12\'s last open slice sub-issue (rule 9).'
    ]


def test_rule_9_accepts_correct_closing_sets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    issues = {
        9: _issue(_body()),
        12: _issue(_body(), labels=['planning']),
    }
    monkeypatch.setattr(slice_gate, 'fetch_issue', lambda _repo, number: issues[number])

    _patch_graph(monkeypatch, parent=12, open_slices=[9, 10])
    assert slice_gate.gate(
        'feat: add law template',
        'Closes #9',
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
    ) == []

    _patch_graph(monkeypatch, parent=12, open_slices=[9])
    assert slice_gate.gate(
        'feat: add law template',
        'Closes #9\nCloses #12',
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
    ) == []


def test_rule_1_rejects_qualified_and_url_closing_references(tmp_path: Path) -> None:
    failures = slice_gate.gate(
        'feat: add law template',
        'Closes #9\nCloses Vaquum/new-repository-template#12',
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
    )
    assert failures == [
        'PR body contains 1 qualified or URL closing reference(s) '
        "('Closes Vaquum/new-repository-template#12'). GitHub honors these on merge "
        'but the gate cannot fold them into the validated closing set; use the bare '
        '`Closes #N` form (rule 1).'
    ]

    url_failures = slice_gate.gate(
        'feat: add law template',
        'Closes #9\nFixes https://github.com/Vaquum/new-repository-template/issues/12',
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
    )
    assert len(url_failures) == 1
    assert 'qualified or URL closing reference' in url_failures[0]


def test_rule_9_rejects_two_slice_labelled_references(
    tmp_path: Path,
    monkeypatch,
) -> None:
    issues = {
        9: _issue(_body()),
        12: _issue(_body()),
    }
    monkeypatch.setattr(slice_gate, 'fetch_issue', lambda _repo, number: issues[number])

    failures = slice_gate.gate(
        'feat: add law template',
        'Closes #9\nCloses #12',
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
    )
    assert failures == [
        'closing references (#9, #12) contain 2 slice-labelled issues; exactly one must '
        'be the slice, and the other reference may only be its parent PRD (rule 9).'
    ]


def test_rule_9_rejects_closed_parent_prd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prd = _issue(_body(), labels=['planning'])
    prd['state'] = 'closed'
    issues = {
        9: _issue(_body()),
        12: prd,
    }
    monkeypatch.setattr(slice_gate, 'fetch_issue', lambda _repo, number: issues[number])
    _patch_graph(monkeypatch, parent=12, open_slices=[9])

    failures = slice_gate.gate(
        'feat: add law template',
        'Closes #9\nCloses #12',
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
    )
    assert failures == [
        "parent PRD #12 state is 'closed'; must be OPEN for the PR to close it (rule 9)."
    ]


def test_rule_10_rejects_unchecked_checkbox(
    tmp_path: Path,
    monkeypatch,
) -> None:
    body = _body(done_means=(
        '## Done Means\n'
        '- [x] Capability complete\n'
        '- [ ] Tests complete\n\n'
        'Merge SHA:\n\n'
    ))
    monkeypatch.setattr(slice_gate, 'fetch_issue', lambda _repo, _number: _issue(body))
    _patch_graph(monkeypatch)

    failures = slice_gate.gate(
        'feat: add law template',
        'Closes #9',
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
    )
    assert failures == [
        'issue #9 Done Means has 1 checkbox(es) neither checked nor overruled: '
        '\'Tests complete\'. Every box must be `- [x]` or carry '
        '`OVERRULED: <reason>` before merge (rule 10).'
    ]


def test_rule_10_accepts_checked_and_overruled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    body = _body(done_means=(
        '## Done Means\n'
        '- [x] Capability complete\n'
        '- [ ] Docs updated OVERRULED: docs land in the follow-up slice\n\n'
        'Merge SHA:\n\n'
    ))
    monkeypatch.setattr(slice_gate, 'fetch_issue', lambda _repo, _number: _issue(body))
    _patch_graph(monkeypatch)

    failures = slice_gate.gate(
        'feat: add law template',
        'Closes #9',
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
    )
    assert failures == []


def test_rule_10_rejects_duplicate_done_means_sections(
    tmp_path: Path,
    monkeypatch,
) -> None:
    body = _body(done_means=(
        '## Done Means\n'
        '- [ ] Tests complete\n\n'
        'Merge SHA:\n\n'
    )) + '\n## Done Means\n- [x] shadow\n\n## Author Checks\n- [x] ok\n'
    monkeypatch.setattr(slice_gate, 'fetch_issue', lambda _repo, _number: _issue(body))
    _patch_graph(monkeypatch)

    failures = slice_gate.gate(
        'feat: add law template',
        'Closes #9',
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
    )
    assert failures == [
        'issue #9 body has 2 Done Means sections (## Done Means ... ## Author Checks); '
        'exactly one is required so checkbox completion and closeout evidence are '
        'unambiguous (rule 10).'
    ]


def test_rule_10_sees_all_gfm_checkbox_forms(
    tmp_path: Path,
    monkeypatch,
) -> None:
    body = _body(done_means=(
        '## Done Means\n'
        '* [ ] Star box\n'
        '+ [ ] Plus box\n'
        '-  [ ] Wide box\n\n'
        'Merge SHA:\n\n'
    ))
    monkeypatch.setattr(slice_gate, 'fetch_issue', lambda _repo, _number: _issue(body))
    _patch_graph(monkeypatch)

    failures = slice_gate.gate(
        'feat: add law template',
        'Closes #9',
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
    )
    assert failures == [
        'issue #9 Done Means has 3 checkbox(es) neither checked nor overruled: '
        "'Star box'; 'Plus box'; 'Wide box'. Every box must be `- [x]` or carry "
        '`OVERRULED: <reason>` before merge (rule 10).'
    ]


def test_rule_10_rejects_placeholder_and_unbounded_overrules(
    tmp_path: Path,
    monkeypatch,
) -> None:
    body = _body(done_means=(
        '## Done Means\n'
        '- [ ] Docs updated OVERRULED: <reason>\n'
        '- [ ] Tests complete NOTOVERRULED: not a real overrule\n\n'
        'Merge SHA:\n\n'
    ))
    monkeypatch.setattr(slice_gate, 'fetch_issue', lambda _repo, _number: _issue(body))
    _patch_graph(monkeypatch)

    failures = slice_gate.gate(
        'feat: add law template',
        'Closes #9',
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
    )
    assert failures == [
        'issue #9 Done Means has 2 checkbox(es) neither checked nor overruled: '
        "'Docs updated OVERRULED: <reason>'; 'Tests complete NOTOVERRULED: not a real "
        "overrule'. Every box must be `- [x]` or carry `OVERRULED: <reason>` before "
        'merge (rule 10).'
    ]


def test_rule_9_is_skipped_when_issue_metadata_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    issue = _issue(_body())
    issue['state'] = 'closed'
    monkeypatch.setattr(slice_gate, 'fetch_issue', lambda _repo, _number: issue)
    monkeypatch.setattr(
        slice_gate,
        'fetch_parent_issue_number',
        lambda _repo, _number: pytest.fail('rule 9 must not run graph lookups here'),
    )

    failures = slice_gate.gate(
        'feat: add law template',
        'Closes #9',
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
    )
    assert failures == [
        "issue #9 state is 'closed'; must be OPEN for a PR to close it."
    ]


def test_rule_10_requires_done_means_section(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        slice_gate,
        'fetch_issue',
        lambda _repo, _number: _issue(_body(done_means='')),
    )
    _patch_graph(monkeypatch)

    failures = slice_gate.gate(
        'feat: add law template',
        'Closes #9',
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
    )
    assert failures == [
        'issue #9 body has no parseable Done Means section (## Done Means ... '
        '## Author Checks); rule 10 cannot verify checkbox completion.'
    ]

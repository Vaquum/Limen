from __future__ import annotations

import json
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
HEAD_SHA = 'a' * 40


def _contract(delivery='maintenance', interfaces=(), proof=None) -> str:
    value = {'delivery': delivery, 'interfaces': list(interfaces),
             'scope': 'Preserve declared behavior.', 'proof': proof or {}}
    return '\n## Interface Contract\n```yaml\n' + json.dumps(value) + '\n```\n'


def _pr(body: str, contract: str | None = None) -> str:
    return body + (_contract() if contract is None else contract)


@pytest.fixture(autouse=True)
def _candidate_evidence(monkeypatch):
    class Evidence:
        def __init__(self, repo, head_sha):
            assert head_sha == HEAD_SHA
            self.required_checks = {'pr_checks_slice'}

        def __call__(self, path):
            if path != 'tests/test_contract_examples.py':
                raise ValueError(f'proof file {path!r} is absent from candidate head')
            return 'def test_python():\n    assert True\n\ndef test_compiled():\n    assert True\n'

    monkeypatch.setattr(slice_gate, 'CandidateEvidence', Evidence)


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
    contract: str | None = None,
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
        f'{_contract() if contract is None else contract}'
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
        _pr('Closes #9'),
        ['governance/version_gate.py', '.github/workflows/pr_checks_version.yml'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
        HEAD_SHA,
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
        _pr('Closes #8'),
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
        HEAD_SHA,
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
        _pr('Closes #9'),
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
        HEAD_SHA,
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
        _pr('Closes #9'),
        ['README.md'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
        HEAD_SHA,
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
        _pr('Closes #9'),
        ['governance/experimental.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
        HEAD_SHA,
    )
    assert any('listed in issue #9 Out of Scope' in item for item in failures)


def test_no_closing_reference_fails_with_closing_set_message(tmp_path: Path) -> None:
    failures = slice_gate.gate(
        'feat: add law template',
        _pr('A body with no closing reference at all.'),
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
        HEAD_SHA,
    )
    assert failures == [
        'PR body has no closing reference. The closing set must be exactly the slice '
        'issue (`Closes #N`, or Fixes/Resolves, with N an OPEN slice-labelled issue), '
        'plus its parent PRD only when explicitly completing the parent (rule 9).'
    ]


def test_multiple_closing_references_fail_before_api_call(tmp_path: Path) -> None:
    failures = slice_gate.gate(
        'feat: add law template',
        _pr('Closes #9\nFixes #10\nResolves #11'),
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
        HEAD_SHA,
    )
    assert failures == [
        'PR body has 3 closing references (#9, #10, #11). The closing set must be exactly '
        'the slice issue, plus its parent PRD only when explicitly completing the parent '
        '(rule 9).'
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
        _pr('Closes #9\nCloses #12'),
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
        HEAD_SHA,
    )
    assert failures == [
        'closing set {#9, #12} must be exactly {#9} because parent PRD #12 still has '
        'other open slice sub-issues (#10) (rule 9).'
    ]


def test_parent_can_remain_open_after_last_slice(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(slice_gate, 'fetch_issue', lambda _repo, _number: _issue(_body()))
    _patch_graph(monkeypatch, parent=12, open_slices=[9])

    failures = slice_gate.gate(
        'feat: add law template',
        _pr('Closes #9\nParent remains open: complete the compiled outcome in a new slice'),
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
        HEAD_SHA,
    )
    assert failures == []


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
        _pr('Closes #9'),
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
        HEAD_SHA,
    ) == []

    _patch_graph(monkeypatch, parent=12, open_slices=[9])
    assert slice_gate.gate(
        'feat: add law template',
        _pr('Closes #9\nCloses #12'),
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
        HEAD_SHA,
    ) == []


def test_rule_1_rejects_qualified_and_url_closing_references(tmp_path: Path) -> None:
    failures = slice_gate.gate(
        'feat: add law template',
        _pr('Closes #9\nCloses Vaquum/new-repository-template#12'),
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
        HEAD_SHA,
    )
    assert failures == [
        'PR body contains 1 qualified or URL closing reference(s) '
        "('Closes Vaquum/new-repository-template#12'). GitHub honors these on merge "
        'but the gate cannot fold them into the validated closing set; use the bare '
        '`Closes #N` form (rule 1).'
    ]

    url_failures = slice_gate.gate(
        'feat: add law template',
        _pr('Closes #9\nFixes https://github.com/Vaquum/new-repository-template/issues/12'),
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
        HEAD_SHA,
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
        _pr('Closes #9\nCloses #12'),
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
        HEAD_SHA,
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
        _pr('Closes #9\nCloses #12'),
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
        HEAD_SHA,
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
        _pr('Closes #9'),
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
        HEAD_SHA,
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
        _pr('Closes #9'),
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
        HEAD_SHA,
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
        _pr('Closes #9'),
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
        HEAD_SHA,
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
        _pr('Closes #9'),
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
        HEAD_SHA,
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
        _pr('Closes #9'),
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
        HEAD_SHA,
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
        _pr('Closes #9'),
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
        HEAD_SHA,
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
        _pr('Closes #9'),
        ['governance/version_gate.py'],
        _template(tmp_path),
        'Vaquum/new-repository-template',
        HEAD_SHA,
    )
    assert failures == [
        'issue #9 body has no parseable Done Means section (## Done Means ... '
        '## Author Checks); rule 10 cannot verify checkbox completion.'
    ]


@pytest.mark.parametrize('remaining', [
    '', 'Parent remains open:', 'Parent remains open: <remaining outcome or next slice>',
    'Parent remains open: (none)',
    'Parent remains open: next outcome\nParent remains open: another outcome',
])
def test_last_child_requires_one_remaining_outcome(tmp_path, monkeypatch, remaining):
    monkeypatch.setattr(slice_gate, 'fetch_issue', lambda *_: _issue(_body()))
    _patch_graph(monkeypatch, parent=12, open_slices=[9])
    failures = slice_gate.gate('feat: add law template', _pr('Closes #9\n' + remaining),
                               [], _template(tmp_path), 'Vaquum/Limen', HEAD_SHA)
    assert len(failures) == 1
    assert 'add `Parent remains open:' in failures[0]


def test_parent_closure_requires_complete_evidence(tmp_path, monkeypatch):
    child = _contract('public', ['python'], {
        'python': ['test:tests/test_contract_examples.py::test_python']})
    parent = _contract('public', ['python', 'compiled'], {
        'python': ['test:tests/test_contract_examples.py::test_python'],
        'compiled': ['test:tests/test_not_delivered.py::test_compiled']})
    issues = {9: _issue(_body(contract=child)),
              12: _issue(_body(contract=parent), labels=['planning'])}
    monkeypatch.setattr(slice_gate, 'fetch_issue', lambda _repo, number: issues[number])
    _patch_graph(monkeypatch, parent=12, open_slices=[9])
    child_only = _pr('Closes #9\nParent remains open: deliver compiled support', child)
    assert slice_gate.gate('feat: add law template', child_only, [], _template(tmp_path),
                           'Vaquum/Limen', HEAD_SHA) == []
    failures = slice_gate.gate('feat: add law template', _pr('Closes #9\nCloses: #12', child),
                               [], _template(tmp_path), 'Vaquum/Limen', HEAD_SHA)
    assert failures
    assert any('test_not_delivered.py' in item for item in failures)


def test_child_only_pr_checks_authoritative_parent_targets(tmp_path, monkeypatch):
    child = _contract('public', ['python'], {
        'python': ['test:tests/test_contract_examples.py::test_python']})
    parent = _contract('public', ['compiled'], {
        'compiled': ['test:tests/test_contract_examples.py::test_compiled']})
    issues = {9: _issue(_body(contract=child)),
              12: _issue(_body(contract=parent), labels=['planning'])}
    monkeypatch.setattr(slice_gate, 'fetch_issue', lambda _repo, number: issues[number])
    _patch_graph(monkeypatch, parent=12, open_slices=[9, 10])
    failures = slice_gate.gate('feat: add law template', _pr('Closes #9', child),
                               [], _template(tmp_path), 'Vaquum/Limen', HEAD_SHA)
    assert failures
    assert any('parent' in item.lower() for item in failures)


def test_pr_cannot_supply_missing_parent_contract(tmp_path, monkeypatch):
    issues = {9: _issue(_body()), 12: _issue(_body(contract=''), labels=['planning'])}
    monkeypatch.setattr(slice_gate, 'fetch_issue', lambda _repo, number: issues[number])
    _patch_graph(monkeypatch, parent=12, open_slices=[9, 10])
    body = _pr('Closes #9') + _contract().replace('## Interface Contract', '## Parent Interface Contract')
    failures = slice_gate.gate('feat: add law template', body, [], _template(tmp_path),
                               'Vaquum/Limen', HEAD_SHA)
    assert failures
    assert any('parent' in item.lower() for item in failures)


def test_missing_slice_contract_adopts_its_own_pr(tmp_path, monkeypatch):
    monkeypatch.setattr(slice_gate, 'fetch_issue', lambda *_: _issue(_body(contract='')))
    _patch_graph(monkeypatch)
    assert slice_gate.gate('feat: add law template', _pr('Closes #9'), [],
                           _template(tmp_path), 'Vaquum/Limen', HEAD_SHA) == []


def test_native_parent_api_failure_does_not_become_parentless(tmp_path, monkeypatch):
    monkeypatch.setattr(slice_gate, 'fetch_issue',
                        lambda _repo, number: _issue(_body()) if number == 9 else None)
    _patch_graph(monkeypatch, parent=12, open_slices=[9])
    with pytest.raises(SystemExit) as error:
        slice_gate.gate('feat: add law template', _pr('Closes #9'), [],
                         _template(tmp_path), 'Vaquum/Limen', HEAD_SHA)
    assert error.value.code == 2


def test_head_sha_is_required_on_cli(monkeypatch, tmp_path):
    monkeypatch.setattr('sys.argv', ['slice_gate', '--pr-title', 'feat: example',
                                   '--pr-body-file', 'body', '--pr-files-file', 'files',
                                   '--template', str(_template(tmp_path)), '--repo', 'Vaquum/Limen'])
    with pytest.raises(SystemExit) as error:
        slice_gate.main()
    assert error.value.code == 2


@pytest.mark.parametrize('keyword', ['Closes:', 'Fixes :', 'Resolves:', 'closed'])
def test_optional_colon_closures_reach_parent_validation(tmp_path, monkeypatch, keyword):
    issues = {9: _issue(_body()), 12: _issue(_body(), labels=['planning'])}
    monkeypatch.setattr(slice_gate, 'fetch_issue', lambda _repo, number: issues[number])
    _patch_graph(monkeypatch, parent=12, open_slices=[9, 10])
    failures = slice_gate.gate('feat: add law template', _pr(f'Closes #9\n{keyword} #12'),
                               [], _template(tmp_path), 'Vaquum/Limen', HEAD_SHA)
    assert failures
    assert 'other open slice' in failures[0]


@pytest.mark.parametrize('target', ['Vaquum/Limen#12', 'https://github.com/Vaquum/Limen/issues/12'])
def test_qualified_colon_reference_cannot_hide_parent(tmp_path, target):
    failures = slice_gate.gate('feat: add law template', _pr(f'Closes #9\nCloses: {target}'),
                               [], _template(tmp_path), 'Vaquum/Limen', HEAD_SHA)
    assert 'qualified or URL closing reference' in failures[0]

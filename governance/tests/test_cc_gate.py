from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_gate_has_no_issue_fetching_machinery() -> None:
    source = (REPO_ROOT / 'governance/cc_gate.py').read_text(encoding='utf-8')

    assert 'fetch_issue' not in source
    assert 'planning' not in source
    assert 'linked issue title' not in source
    # Closing-reference parsing stays: commit closures must be declared
    # in the PR body.
    assert 'def find_closing_references(body: str) -> list[int]:' in source


def _load_cc_gate():
    import importlib.util
    spec = importlib.util.spec_from_file_location('cc_gate', REPO_ROOT / 'governance/cc_gate.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_attribution_hit_flags_ai_attribution() -> None:
    cc = _load_cc_gate()
    assert cc.attribution_hit('Co-Authored-By: Claude <noreply@anthropic.com>') is not None
    assert cc.attribution_hit('feat: generated with Copilot assistance') is not None
    assert cc.attribution_hit('chore: 🤖 generated with Claude') is not None
    assert cc.attribution_hit('refactor: address anthropic feedback') is not None
    # AI-qualified forms of the narrowed tokens stay flagged.
    assert cc.attribution_hit('docs: notes from a google-gemini session') is not None
    assert cc.attribution_hit('docs: gemini-pro prompt notes') is not None
    assert cc.attribution_hit('feat: cursor-ai integration pass') is not None
    assert cc.attribution_hit('chore: llm-generated summary') is not None
    assert cc.attribution_hit('Co-authored-by: GPT <bot@example.com>') is not None


def test_attribution_hit_passes_clean_messages() -> None:
    cc = _load_cc_gate()
    assert cc.attribution_hit('feat: add the maker-fill evaluator') is None
    assert cc.attribution_hit('docs: clarify the rollout runbook') is None
    assert cc.attribution_hit('fix: bound the version read') is None
    # The repo's own governance files and the Copilot review feature are
    # exempt -- they are filenames and a feature, not authorship.
    assert cc.attribution_hit('docs: retitle CLAUDE.md and point AGENTS.md at it') is None
    assert cc.attribution_hit('feat: update copilot-instructions.md and require Copilot review') is None
    # Domain vocabulary that names no AI assistant must pass: the bare
    # tokens are narrowed to AI-qualified forms, `api`/`code` are not
    # Gemini qualifiers, and the co-author alternation is word-bounded.
    assert cc.attribution_hit('feat: add Gemini exchange connector') is None
    assert cc.attribution_hit('feat: add Gemini API client for the Gemini exchange') is None
    assert cc.attribution_hit('fix: reuse the database cursor between batches') is None
    assert cc.attribution_hit('docs: describe artifacts generated with the manifest runner') is None
    assert cc.attribution_hit('Co-authored-by: Bill Hillman <hillman@example.com>') is None


def test_find_closing_references_deduplicates_in_order() -> None:
    cc = _load_cc_gate()
    assert cc.find_closing_references('Closes #9\nFixes: #12\nResolves:#9') == [9, 12]


def test_list_commits_fails_closed_on_unparseable_log_line(monkeypatch) -> None:
    import subprocess

    cc = _load_cc_gate()

    class FakeResult:
        returncode = 0
        stdout = 'deadbeef only-two-fields\n'
        stderr = ''

    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: FakeResult())
    with pytest.raises(SystemExit) as excinfo:
        cc.list_commits('base', 'head')
    assert excinfo.value.code == 2


def test_gate_checks_title_commits_closures_and_attribution_only(monkeypatch) -> None:
    cc = _load_cc_gate()
    monkeypatch.setattr(cc, 'list_commits', lambda _base, _head: [])
    monkeypatch.setattr(cc, 'list_commit_messages', lambda _base, _head: [])

    assert cc.gate('feat: valid title', 'Closes #9', 'base', 'head', 'Vaquum/Limen') == []

    failures = cc.gate('not a conventional title', 'Closes #9', 'base', 'head', 'Vaquum/Limen')
    assert len(failures) == 1
    assert 'PR title' in failures[0]


@pytest.mark.parametrize('closing', [
    'Closes #12', 'Closes: #12', 'Fixes:#12',
    'Resolves: Vaquum/Limen#12',
    'Fixes: https://github.com/Vaquum/Limen/issues/12',
])
def test_commit_closures_are_declared_in_pr(monkeypatch, closing):
    cc = _load_cc_gate()
    monkeypatch.setattr(cc, 'list_commits', lambda *_: [])
    monkeypatch.setattr(cc, 'list_commit_messages', lambda *_: [('a' * 40, 'fix: example\n\n' + closing)])
    failures = cc.gate('fix: example', 'Closes #9', 'base', 'head', 'Vaquum/Limen')
    assert len(failures) == 1
    assert 'add `Closes #12` to the PR body' in failures[0]
    assert cc.gate('fix: example', 'Closes #9\nCloses: #12', 'base', 'head', 'Vaquum/Limen') == []


@pytest.mark.parametrize('closing', [
    'Fixes: Other/Repository#12',
    'Fixes https://github.com/Other/Repository/issues/12',
    'Fixes https://another-host.example/Vaquum/Limen/issues/12',
])
def test_cross_repository_commit_closures_fail_even_when_number_matches(monkeypatch, closing):
    cc = _load_cc_gate()
    monkeypatch.setattr(cc, 'list_commits', lambda *_: [])
    monkeypatch.setattr(cc, 'list_commit_messages', lambda *_: [('a' * 40, 'fix: example\n\n' + closing)])
    failures = cc.gate('fix: example', 'Closes #12', 'base', 'head', 'Vaquum/Limen')
    assert len(failures) == 1
    assert 'cross-repository commit closure' in failures[0]


def test_merge_commit_closures_are_checked_without_attribution_regression(monkeypatch):
    cc = _load_cc_gate()
    monkeypatch.setattr(cc, 'list_commits', lambda *_: [
        {'sha': 'merge', 'subject': "Merge branch 'codex/example'", 'is_merge': True}])
    monkeypatch.setattr(cc, 'list_commit_messages', lambda *_: [
        ('merge', "Merge branch 'codex/example'\n\nCloses: #12")])
    failures = cc.gate('fix: example', 'Closes #9', 'base', 'head', 'Vaquum/Limen')
    assert len(failures) == 1
    assert 'add `Closes #12`' in failures[0]
    assert cc.gate('fix: example', 'Closes #9\nCloses #12', 'base', 'head', 'Vaquum/Limen') == []


def test_list_commit_messages_keeps_merge_messages(monkeypatch):
    cc = _load_cc_gate()
    calls = []

    def git(args):
        calls.append(args)
        return 'merge\x1fMerge branch\n\nCloses: #12\n\x1e'

    monkeypatch.setattr(cc, 'run_git', git)
    assert cc.list_commit_messages('base', 'head') == [('merge', 'Merge branch\n\nCloses: #12')]
    assert '--no-merges' not in calls[0]

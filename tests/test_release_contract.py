from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def _load_create_release() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        'create_release',
        ROOT / 'scripts' / 'create_release.py',
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_wait_for_checks() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        'wait_for_commit_checks',
        ROOT / 'scripts' / 'wait_for_commit_checks.py',
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['wait_for_commit_checks'] = module
    spec.loader.exec_module(module)
    return module


def _run(name, status, conclusion=None):
    payload = {'name': name, 'status': status}
    if conclusion is not None:
        payload['conclusion'] = conclusion
    return payload


def test_wait_partition_sorts_runs_by_state():
    module = _load_wait_for_checks()
    verdict = module.partition_runs([
        _run('PR Checks Tests', 'completed', 'success'),
        _run('PR Checks Coverage', 'completed', 'skipped'),
        _run('PR Checks Runtime', 'in_progress'),
        _run('PR Checks Docs Site', 'completed', 'failure'),
    ], [])
    assert verdict.passed == ('PR Checks Tests', 'PR Checks Coverage')
    assert verdict.pending == ('PR Checks Runtime',)
    assert verdict.failed == ('PR Checks Docs Site (failure)',)
    assert verdict.excluded == ()


def test_wait_partition_excludes_release_machinery_case_insensitively():
    module = _load_wait_for_checks()
    verdict = module.partition_runs([
        _run('Create Automated Release', 'completed', 'success'),
        _run('Publish Wheel', 'in_progress'),
        _run('PR Checks Tests', 'completed', 'success'),
    ], ['automated release', 'PUBLISH'])
    assert verdict.excluded == ('Create Automated Release', 'Publish Wheel')
    assert verdict.passed == ('PR Checks Tests',)
    assert verdict.pending == ()
    assert verdict.failed == ()


def test_wait_loop_returns_fast_paths_without_sleeping(monkeypatch):
    module = _load_wait_for_checks()
    sleeps = []
    monkeypatch.setattr(module.time, 'sleep', lambda s: sleeps.append(s))
    monkeypatch.setattr(
        module, 'fetch_check_runs',
        lambda repo, sha: [_run('PR Checks Tests', 'completed', 'success')],
    )
    verdict, outcome = module.wait_for_green('o/r', 'sha', [], 60.0, 5.0, 1)
    assert outcome == 'passed'
    assert verdict.passed == ('PR Checks Tests',)
    assert sleeps == []
    monkeypatch.setattr(
        module, 'fetch_check_runs',
        lambda repo, sha: [_run('PR Checks Tests', 'completed', 'failure')],
    )
    verdict, outcome = module.wait_for_green('o/r', 'sha', [], 60.0, 5.0)
    assert outcome == 'failed'
    assert verdict.failed == ('PR Checks Tests (failure)',)
    assert sleeps == []


def test_wait_loop_waits_then_settles(monkeypatch):
    module = _load_wait_for_checks()
    now = [1000.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(
        module.time, 'sleep',
        lambda s: now.__setitem__(0, now[0] + s),
    )
    green = [_run('PR Checks Tests', 'completed', 'success')]
    payloads = [
        [_run('PR Checks Tests', 'in_progress')],
        green, green, green,
    ]
    monkeypatch.setattr(
        module, 'fetch_check_runs', lambda repo, sha: payloads.pop(0)
    )
    verdict, outcome = module.wait_for_green('o/r', 'sha', [], 60.0, 5.0)
    assert outcome == 'passed'
    assert verdict.passed == ('PR Checks Tests',)
    assert verdict.pending == ()


def test_wait_loop_times_out_on_pending(monkeypatch):
    module = _load_wait_for_checks()
    now = [1000.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(
        module.time, 'sleep',
        lambda s: now.__setitem__(0, now[0] + s),
    )
    monkeypatch.setattr(
        module, 'fetch_check_runs',
        lambda repo, sha: [_run('PR Checks Tests', 'in_progress')],
    )
    verdict, outcome = module.wait_for_green('o/r', 'sha', [], 10.0, 30.0)
    assert outcome == 'timeout'
    assert verdict.pending == ('PR Checks Tests',)


def test_wait_loop_deadline_green_is_a_timeout(monkeypatch):
    module = _load_wait_for_checks()
    now = [1000.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(
        module.time, 'sleep',
        lambda s: now.__setitem__(0, now[0] + s),
    )
    payloads = [
        [_run('PR Checks Tests', 'in_progress')],
        [_run('PR Checks Tests', 'completed', 'success')],
    ]
    monkeypatch.setattr(
        module, 'fetch_check_runs', lambda repo, sha: payloads.pop(0)
    )
    verdict, outcome = module.wait_for_green('o/r', 'sha', [], 10.0, 30.0)
    assert outcome == 'timeout'
    assert verdict.passed == ('PR Checks Tests',)


def test_wait_main_exit_codes(monkeypatch, capsys):
    module = _load_wait_for_checks()
    green = module.Verdict(
        pending=(), failed=(), passed=('PR Checks Tests',), excluded=()
    )
    monkeypatch.setattr(module, 'wait_for_green', lambda *a: (green, 'passed'))
    argv = ['--repo', 'o/r', '--sha', 'abc', '--exclude-substring', 'x']
    assert module.main(argv) == 0
    assert 'PASS' in capsys.readouterr().out
    bad = module.Verdict(
        pending=(), failed=('PR Checks Tests (failure)',), passed=(), excluded=()
    )
    monkeypatch.setattr(module, 'wait_for_green', lambda *a: (bad, 'failed'))
    assert module.main(argv) == 1
    assert 'FAILING CHECKS' in capsys.readouterr().err
    empty = module.Verdict(pending=(), failed=(), passed=(), excluded=())
    monkeypatch.setattr(module, 'wait_for_green', lambda *a: (empty, 'timeout'))
    assert module.main(argv) == 1
    assert 'refusing to ship' in capsys.readouterr().err
    stuck = module.Verdict(
        pending=('PR Checks Tests',), failed=(), passed=(), excluded=()
    )
    monkeypatch.setattr(module, 'wait_for_green', lambda *a: (stuck, 'timeout'))
    assert module.main(argv) == 1
    err = capsys.readouterr().err
    assert 'timed out' in err
    assert 'PR Checks Tests' in err


def test_wait_fetch_maps_failures_to_exit_2(monkeypatch):
    module = _load_wait_for_checks()

    def missing(*args, **kwargs):
        raise FileNotFoundError('no gh')

    monkeypatch.setattr(module.subprocess, 'run', missing)
    with pytest.raises(SystemExit, match='gh not found'):
        module.fetch_check_runs('o/r', 'sha')


def test_wait_fetch_parses_paginated_line_delimited_runs(monkeypatch):
    import json

    module = _load_wait_for_checks()

    class Completed:
        returncode = 0
        stdout = '\n'.join([
            json.dumps(_run('PR Checks Tests', 'completed', 'success')),
            '',
            json.dumps(_run('PR Checks Coverage', 'in_progress')),
        ])
        stderr = ''

    captured = {}

    def fake_run(cmd, **kwargs):
        captured['cmd'] = cmd
        return Completed()

    monkeypatch.setattr(module.subprocess, 'run', fake_run)
    runs = module.fetch_check_runs('o/r', 'sha')
    assert [run['name'] for run in runs] == ['PR Checks Tests', 'PR Checks Coverage']
    assert captured['cmd'][-2:] == ['--jq', '.check_runs[]']


def test_wait_fetch_rejects_malformed_lines(monkeypatch):
    module = _load_wait_for_checks()

    class BadJson:
        returncode = 0
        stdout = '{"name": "broken",\n'
        stderr = ''

    class BadShape:
        returncode = 0
        stdout = '[1, 2]\n'
        stderr = ''

    monkeypatch.setattr(
        module.subprocess, 'run', lambda *a, **k: BadJson()
    )
    with pytest.raises(SystemExit, match='bad gh output on line 1'):
        module.fetch_check_runs('o/r', 'sha')
    monkeypatch.setattr(
        module.subprocess, 'run', lambda *a, **k: BadShape()
    )
    with pytest.raises(SystemExit, match='unexpected gh output shape on line 1'):
        module.fetch_check_runs('o/r', 'sha')


def test_wait_loop_holds_for_late_registering_runs(monkeypatch):
    module = _load_wait_for_checks()
    now = [1000.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(
        module.time, 'sleep',
        lambda s: now.__setitem__(0, now[0] + s),
    )
    green_tests = [_run('PR Checks Tests', 'completed', 'success')]
    green_both = [*green_tests, _run('PR Checks Coverage', 'completed', 'success')]
    payloads = [green_tests, green_tests, green_both, green_both, green_both]
    calls = []
    monkeypatch.setattr(
        module, 'fetch_check_runs',
        lambda repo, sha: (calls.append(1), payloads.pop(0))[1],
    )
    verdict, outcome = module.wait_for_green('o/r', 'sha', [], 60.0, 5.0)
    assert outcome == 'passed'
    assert verdict.passed == ('PR Checks Tests', 'PR Checks Coverage')
    assert len(calls) == 5


def test_release_tag_authority_and_traceability() -> None:
    module = _load_create_release()

    assert module.TAG_RE.pattern == r'^v\d+\.\d+\.\d+$'
    assert module.TAG_RE.match('v5.1.0')
    for rejected in ('5.1.0', 'v5.1', 'v5.1.0rc1', 'V5.1.0', 'v5.1.0 '):
        assert module.TAG_RE.match(rejected) is None, rejected

    version = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    assert module.compute_tag(version, {}) == f'v{version}'
    assert module.compute_tag(version, {'tag': f'v{version}', 'version': version}) == f'v{version}'
    with pytest.raises(ValueError, match='model may not choose identifiers'):
        module.compute_tag(version, {'tag': 'v99.0.0'})
    with pytest.raises(ValueError, match='model may not choose identifiers'):
        module.compute_tag(version, {'version': '99.0.0'})
    with pytest.raises(ValueError, match='model may not choose identifiers'):
        module.compute_tag('5.1.0rc1', {})

    module.validate_release_prose({'release_name': 'Fire Horse', 'release_notes': '## Summary'})
    for malformed in ({}, {'release_name': 'x'}, {'release_name': '', 'release_notes': 'y'}, {'release_name': 'x', 'release_notes': 3}):
        with pytest.raises(ValueError, match='non-empty string release_'):
            module.validate_release_prose(malformed)

    changelog = (ROOT / 'CHANGELOG.md').read_text(encoding='utf-8')
    anchor = module.changelog_anchor(version, changelog)
    assert anchor.startswith(''.join(version.split('.')))
    assert module.changelog_anchor('9.9.9', '## [9.9.9] - 2026-01-02\n') == '999---2026-01-02'
    with pytest.raises(ValueError, match=r'create_release CHANGELOG\.md has no entry'):
        module.changelog_anchor('0.0.0', changelog)

    assert module.extract_pr_numbers([
        'Merge pull request #687 from Vaquum/feat/687-release-supply-chain-hardening',
        'test: decouple docs inventory from formatting (#689)',
        'docs: mention #123 in passing',
        'chore: no reference',
    ]) == [687, 689]

    appendix = module.build_traceability('v5.1.0', 'v5.0.0', [687], '510---2026-07-14')
    assert appendix.startswith('## Traceability')
    assert '- Merged pull requests: #687' in appendix
    assert '- Compare: https://github.com/Vaquum/Limen/compare/v5.0.0...v5.1.0' in appendix
    assert '- Changelog: https://github.com/Vaquum/Limen/blob/v5.1.0/CHANGELOG.md#510---2026-07-14' in appendix
    assert 'SHA-256' in appendix

    first_release = module.build_traceability('v0.1.0', None, [], 'anchor')
    assert 'Compare:' not in first_release
    assert 'Merged pull requests:' not in first_release


def test_manual_release_requires_the_exact_merged_source(tmp_path, monkeypatch):
    import json
    import subprocess

    trigger = _load_create_release()

    def git(*arguments):
        return subprocess.check_output(['git', *arguments], cwd=tmp_path, text=True).strip()

    git('init', '-q')
    git('config', 'user.name', 'Release contract')
    git('config', 'user.email', 'release@example.test')
    (tmp_path / 'pyproject.toml').write_text('[project]\nversion = "5.46.2"\n')
    git('add', '.')
    git('commit', '-qm', 'Release predecessor')
    git('tag', 'v5.46.1')
    predecessor = git('rev-parse', 'HEAD')
    (tmp_path / 'release.txt').write_text('Reviewed release source\n')
    git('add', '.')
    git('commit', '-qm', 'Release candidate')
    head = git('rev-parse', 'HEAD')
    git('update-ref', 'refs/remotes/origin/main', head)
    tag = f'v5.46.2-dispatch-{head}'
    git('tag', tag)
    run = {'path': '.github/workflows/pr_post_release.yml', 'event': 'workflow_dispatch',
           'conclusion': 'success', 'head_repository': {'full_name': 'Vaquum/Limen'},
           'head_sha': head, 'head_branch': tag}
    trigger.verify_release_trigger(tmp_path, run)
    trigger.verify_release_trigger(tmp_path, dict(run, head_branch='main'))
    record = tmp_path / 'run.json'
    record.write_text(json.dumps(run))
    # Exercise the same file-input entry path used by the publisher.
    subprocess.run([sys.executable, str(ROOT / 'scripts/create_release.py'), '--verify-trigger', str(record)], cwd=tmp_path, check=True)
    for field, bad in [('path', '.github/workflows/other.yml'), ('event', 'push'),
                       ('conclusion', 'failure'), ('head_repository', {'full_name': 'other/Limen'}),
                       ('head_sha', predecessor), ('head_branch', 'codex/other'),
                       ('head_branch', f'v5.46.3-dispatch-{head}'),
                       ('head_branch', f'v5.46.2-dispatch-{predecessor}')]:
        with pytest.raises(ValueError):
            trigger.verify_release_trigger(tmp_path, dict(run, **{field: bad}))
    git('tag', '-f', tag, predecessor)
    with pytest.raises(ValueError, match='another commit'):
        trigger.verify_release_trigger(tmp_path, run)
    git('tag', '-f', tag, head)
    git('update-ref', 'refs/remotes/origin/main', predecessor)
    with pytest.raises(subprocess.CalledProcessError):
        trigger.verify_release_trigger(tmp_path, run)
    monkeypatch.chdir(tmp_path)
    assert _load_create_release().get_previous_tag() == 'v5.46.1'

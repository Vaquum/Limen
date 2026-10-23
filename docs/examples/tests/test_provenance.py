import copy
import json
from pathlib import Path
import subprocess

import pytest

from provenance import acquire, base_revision, git, identity, latest_change, main, merged_pull, pull_associations, tree_objects, validate_provenance

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def recorded_history(tmp_path):
    tmp_path = tmp_path / 'checkout'
    tmp_path.mkdir()
    def run(*args):
        subprocess.run(['git', '-c', 'user.name=Research book test', '-c', 'user.email=research-book@example.invalid', *args], cwd=tmp_path, check=True, capture_output=True)
    original = (ROOT / 'docs/Research-Book.md').read_text()
    source = tmp_path / 'page.md'
    run('init', '-q', '-b', 'main')
    source.write_text(original);run('add', 'page.md');run('commit', '-qm', 'docs: retain recorded chapter')
    base = git(tmp_path, 'rev-parse', 'HEAD')
    run('update-ref', 'refs/remotes/origin/main', base)
    run('checkout', '-qb', 'chapter')
    source.write_text(original + (ROOT / 'docs/examples/python-research/README.md').read_text())
    run('commit', '-qam', 'docs: add project continuation')
    return tmp_path, run, base, git(tmp_path, 'rev-parse', 'HEAD')


@pytest.mark.parametrize('fast_forward', [False, True])
def test_original_head_uses_shared_first_parent_main_boundary(recorded_history, fast_forward):
    root, run, base, candidate = recorded_history
    assert base_revision(root) == base
    assert identity(root, base, 'page.md', tree_objects(root, candidate), tree_objects(root, base)) == (candidate, 'candidate')
    run('checkout', '-q', 'main')
    run('merge', '--ff-only' if fast_forward else '--no-ff', 'chapter', '-qm', 'merge: integrate chapter')
    merged = git(root, 'rev-parse', 'HEAD')
    run('update-ref', 'refs/remotes/origin/main', merged)
    assert base_revision(root) == merged
    current = tree_objects(root, merged)
    assert identity(root, merged, 'page.md', current, current) == (merged, 'merged')
    (root / 'later.md').write_text((ROOT / 'docs/examples/compiled-research/README.md').read_text())
    run('add', 'later.md');run('commit', '-qm', 'docs: retain later project source')
    run('update-ref', 'refs/remotes/origin/main', 'HEAD')
    run('checkout', '-q', '--detach', candidate)
    assert git(root, 'merge-base', 'HEAD', 'origin/main') == candidate
    expected = candidate if fast_forward else base
    assert base_revision(root) == expected
    assert git(root, 'rev-parse', 'HEAD') == candidate
    expected_status = 'merged' if fast_forward else 'candidate'
    assert identity(root, expected, 'page.md', tree_objects(root, candidate), tree_objects(root, expected)) == (candidate, expected_status)
    run('checkout', '-q', '--detach', base)
    assert base_revision(root) == base


def test_provenance_cli_inspects_explicit_checkout_without_changing_it(recorded_history, tmp_path, monkeypatch):
    root, _, base, candidate = recorded_history
    output = tmp_path / 'provenance.json'
    monkeypatch.setattr('provenance.source_paths', lambda selected: ['page.md'])
    monkeypatch.setattr('sys.argv', ['provenance.py', '--root', str(root), '--output', str(output)])
    before = git(root, 'status', '--porcelain', '--untracked-files=normal')
    main()
    record = json.loads(output.read_text())
    assert record['base'] == base
    assert [(item['path'], item['commit'], item['status'], item['merged_pr']) for item in record['files']] == [('page.md', candidate, 'candidate', None)]
    assert git(root, 'rev-parse', 'HEAD') == candidate
    assert git(root, 'status', '--porcelain', '--untracked-files=normal') == before


def test_restored_pre_pr_bytes_remain_candidate_against_original_ancestry(recorded_history, monkeypatch):
    root, run, base, candidate = recorded_history
    run('checkout', '-q', 'main');run('merge', '--no-ff', 'chapter', '-qm', 'merge: integrate chapter')
    run('update-ref', 'refs/remotes/origin/main', 'HEAD')
    run('checkout', '-q', 'chapter')
    (root / 'page.md').write_text(git(root, 'show', base + ':page.md') + '\n')
    run('commit', '-qam', 'docs: restore the recorded chapter')
    restored = git(root, 'rev-parse', 'HEAD')
    assert git(root, 'merge-base', 'HEAD', 'origin/main') == candidate
    assert tree_objects(root, base)['page.md'] == tree_objects(root, restored)['page.md']
    monkeypatch.setattr('provenance.source_paths', lambda selected: ['page.md'])
    record = acquire(root, root.parent / 'restored-provenance.json')
    assert (record['files'][0]['commit'], record['files'][0]['status']) == (restored, 'candidate')
    validate_provenance(root, record)
    comparison = subprocess.run(['git', 'diff', '--quiet', candidate, 'HEAD', '--', 'page.md'], cwd=root, check=False)
    assert comparison.returncode == 1
    record['files'][0]['status'] = 'merged'
    with pytest.raises(ValueError, match='Source-changing commit or source bytes differ'):
        validate_provenance(root, record)


def test_first_parent_change_includes_merge_and_later_revert(tmp_path):
    def run(*args):
        subprocess.run(['git', '-c', 'user.name=Research book test', '-c', 'user.email=research-book@example.invalid', *args], cwd=tmp_path, check=True, capture_output=True)
    source = tmp_path / 'page.md'
    original = (ROOT / 'docs/Research-Book.md').read_text()
    run('init', '-q', '-b', 'main')
    source.write_text(original);run('add', 'page.md');run('commit', '-qm', 'docs: retain recorded chapter')
    run('checkout', '-qb', 'chapter')
    source.write_text(original + (ROOT / 'docs/examples/python-research/README.md').read_text())
    run('commit', '-qam', 'docs: add project continuation')
    run('checkout', '-q', 'main');run('merge', '--no-ff', 'chapter', '-qm', 'merge: integrate chapter')
    merged = git(tmp_path, 'rev-parse', 'HEAD')
    assert latest_change(tmp_path, 'HEAD', 'page.md') == merged
    before = tree_objects(tmp_path, merged)
    assert before['page.md'] == git(tmp_path, 'rev-parse', merged + ':page.md')
    assert identity(tmp_path, merged, 'page.md', before, before) == (merged, 'merged')
    source.write_text(original);run('commit', '-qam', 'docs: restore chapter boundary')
    restored = git(tmp_path, 'rev-parse', 'HEAD')
    current = tree_objects(tmp_path, 'HEAD')
    assert latest_change(tmp_path, 'HEAD', 'page.md') == restored
    assert latest_change(tmp_path, 'HEAD', 'page.md') != merged
    assert identity(tmp_path, merged, 'page.md', current, before) == (restored, 'candidate')
    assert identity(tmp_path, restored, 'page.md', current, current) == (restored, 'merged')
    named = tmp_path / 'project with spaces.md'
    named.write_text((ROOT / 'docs/examples/python-research/README.md').read_text())
    run('add', named.name);run('commit', '-qm', 'docs: retain project source')
    current = tree_objects(tmp_path, 'HEAD')
    assert current[named.name] == git(tmp_path, 'rev-parse', 'HEAD:' + named.name)
    assert identity(tmp_path, merged, named.name, current, before) == (git(tmp_path, 'rev-parse', 'HEAD'), 'candidate')


def test_source_association_requires_its_exact_merged_main_pr():
    responses = json.loads((ROOT / 'docs/examples/tests/fixtures/merged-source-pr.json').read_text())
    commit = '56b71e68c08fce5a1edbdb002bd0b23e91a6c527'
    assert merged_pull(commit, responses) == 'https://github.com/Vaquum/Limen/pull/1055'
    for field, value in [('merged_at', None), ('merge_commit_sha', '0' * 40)]:
        changed = copy.deepcopy(responses);changed[0][field] = value
        with pytest.raises(ValueError, match='unique merged PR'):
            merged_pull(commit, changed)
    with pytest.raises(ValueError, match='unique merged PR'):
        merged_pull(commit, responses + responses)


def test_incomplete_github_response_retries_and_retains_exact_association(monkeypatch, capsys):
    payload = (ROOT / 'docs/examples/tests/fixtures/merged-source-pr.json').read_text()
    commit = '56b71e68c08fce5a1edbdb002bd0b23e91a6c527'
    responses = iter([(1, '', 'unexpected end of JSON input\n'), (0, payload, '')])
    calls, delays = [], []

    def run(command, **kwargs):
        assert kwargs['timeout'] == 30
        calls.append(command)
        status, output, error = next(responses)
        return subprocess.CompletedProcess(command, status, output, error)

    monkeypatch.setattr('provenance.subprocess.run', run)
    monkeypatch.setattr('provenance.time.sleep', delays.append)
    assert merged_pull(commit, pull_associations(commit)) == 'https://github.com/Vaquum/Limen/pull/1055'
    assert calls == [['gh', 'api', f'repos/Vaquum/Limen/commits/{commit}/pulls']] * 2
    assert delays == [1]
    assert 'retry 1/5: unexpected end of JSON input' in capsys.readouterr().err


@pytest.mark.parametrize('error,attempts,delays', [('unexpected end of JSON input\n', 6, [1, 2, 3, 4, 5]), ('HTTP 401: Bad credentials\n', 1, [])])
def test_source_association_errors_remain_fatal(monkeypatch, error, attempts, delays):
    calls, sleeps = [], []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, '', error)

    monkeypatch.setattr('provenance.subprocess.run', run)
    monkeypatch.setattr('provenance.time.sleep', sleeps.append)
    with pytest.raises(subprocess.CalledProcessError) as failure:
        pull_associations('56b71e68c08fce5a1edbdb002bd0b23e91a6c527')
    assert failure.value.stderr == error
    assert len(calls) == attempts
    assert sleeps == delays


@pytest.mark.parametrize('payload,error', [('{', json.JSONDecodeError), ('{}', ValueError)])
def test_successful_response_requires_a_complete_json_array(monkeypatch, payload, error):
    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, payload, '')

    monkeypatch.setattr('provenance.subprocess.run', run)
    with pytest.raises(error):
        pull_associations('56b71e68c08fce5a1edbdb002bd0b23e91a6c527')


def test_candidate_source_status_uses_one_exact_git_inventory(recorded_history, monkeypatch):
    from check_evidence import validate_source_status
    root, run, _, _ = recorded_history
    chapter = (ROOT / 'docs/Research-Book.md').read_text()
    (root / 'unchanged.md').write_text(chapter)
    run('add', 'unchanged.md');run('commit', '-qm', 'docs: retain unchanged source')
    base = git(root, 'rev-parse', 'HEAD')
    names = ['page.md', 'project with spaces.md', 'project\nwith\ttabs.md', 'project\rwithCR.md']
    continuation = (ROOT / 'docs/examples/python-research/README.md').read_text()
    for name in names:
        (root / name).write_text(chapter if name == 'page.md' else chapter + continuation)
    run('add', '.');run('commit', '-qm', 'docs: retain candidate sources')
    sources = [{'path': 'unchanged.md', 'status': 'merged', 'merged_pr': None}]
    sources += [{'path': name, 'status': 'candidate', 'merged_pr': None} for name in names]
    sources.append(sources[1] | {'region': 'selection'})
    original = subprocess.check_output
    calls = []

    def observe(command, **kwargs):
        if command[:2] == ['git', 'diff']:
            calls.append(command)
        return original(command, **kwargs)

    monkeypatch.setattr('check_evidence.subprocess.check_output', observe)
    validate_source_status(root, sources, base)
    assert len(calls) == 1
    assert '--name-only' in calls[0] and '-z' in calls[0] and '--no-renames' in calls[0]
    for name in names:
        changed = copy.deepcopy(sources)
        next(item for item in changed if item['path'] == name)['status'] = 'merged'
        with pytest.raises(ValueError, match='Unmerged source presented as merged'):
            validate_source_status(root, changed, base)
    for comparison in (base, None):
        changed = copy.deepcopy(sources)
        changed[1]['merged_pr'] = 'https://github.com/Vaquum/Limen/pull/1055'
        count = len(calls)
        with pytest.raises(ValueError, match='Candidate source cannot claim a merged PR'):
            validate_source_status(root, changed, comparison)
        assert len(calls) == count + int(comparison is not None)
    (root / 'unchanged.md').write_text(continuation)
    run('commit', '-qam', 'docs: change previously merged source')
    with pytest.raises(ValueError, match='Unmerged source presented as merged'):
        validate_source_status(root, sources, base)


def test_candidate_source_status_does_not_relabel_git_errors(recorded_history):
    from check_evidence import validate_source_status
    root, _, _, _ = recorded_history
    sources = [{'path': 'page.md', 'status': 'candidate', 'merged_pr': None}]
    with pytest.raises(subprocess.CalledProcessError) as failure:
        validate_source_status(root, sources, 'refs/heads/missing-research-source')
    assert failure.value.returncode == 128

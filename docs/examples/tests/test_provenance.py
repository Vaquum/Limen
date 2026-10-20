import copy
import json
from pathlib import Path
import subprocess

import pytest

from provenance import git, identity, latest_change, merged_pull, pull_associations, tree_objects

ROOT = Path(__file__).resolve().parents[3]


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
        calls.append(command)
        status, output, error = next(responses)
        return subprocess.CompletedProcess(command, status, output, error)

    monkeypatch.setattr('provenance.subprocess.run', run)
    monkeypatch.setattr('provenance.time.sleep', delays.append)
    assert merged_pull(commit, pull_associations(commit)) == 'https://github.com/Vaquum/Limen/pull/1055'
    assert calls == [['gh', 'api', f'repos/Vaquum/Limen/commits/{commit}/pulls']] * 2
    assert delays == [1]
    assert 'retry 1/2: unexpected end of JSON input' in capsys.readouterr().err


@pytest.mark.parametrize('error,attempts,delays', [('unexpected end of JSON input\n', 3, [1, 2]), ('HTTP 401: Bad credentials\n', 1, [])])
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

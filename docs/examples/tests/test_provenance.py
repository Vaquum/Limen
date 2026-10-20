import copy
import json
from pathlib import Path
import subprocess

import pytest

from provenance import git, latest_change, merged_pull

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
    source.write_text(original);run('commit', '-qam', 'docs: restore chapter boundary')
    assert latest_change(tmp_path, 'HEAD', 'page.md') == git(tmp_path, 'rev-parse', 'HEAD')
    assert latest_change(tmp_path, 'HEAD', 'page.md') != merged


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

import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

from limen.cli.git_utils import git_add_and_commit
from limen.cli.git_utils import git_executable
from limen.cli.git_utils import git_snapshot


def _init_repo(repo: Path, with_identity: bool = True) -> None:
    git = git_executable()
    subprocess.run([git, 'init'], cwd=repo, capture_output=True, check=False)
    if with_identity:
        subprocess.run([git, 'config', 'user.email', 't@t'], cwd=repo, capture_output=True, check=False)
        subprocess.run([git, 'config', 'user.name', 't'], cwd=repo, capture_output=True, check=False)


def test_git_executable_returns_path() -> None:
    path = git_executable()
    assert 'git' in path


def test_git_add_and_commit_returns_false_when_git_not_found() -> None:
    with tempfile.TemporaryDirectory() as d, \
         patch('limen.cli.git_utils.git_executable', side_effect=FileNotFoundError):
        result = git_add_and_commit(Path(d), Path('file.txt'), 'msg')
    assert result is False


def test_git_add_and_commit_creates_commit() -> None:
    git = git_executable()
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        subprocess.run([git, 'init'], cwd=repo, capture_output=True, check=False)
        subprocess.run([git, 'config', 'user.email', 'test@test.com'], cwd=repo, capture_output=True, check=False)
        subprocess.run([git, 'config', 'user.name', 'Test'], cwd=repo, capture_output=True, check=False)
        (repo / 'file.txt').write_text('hello')
        git_add_and_commit(repo, Path('file.txt'), 'test commit')
        result = subprocess.run([git, 'log', '--oneline'], cwd=repo, capture_output=True, text=True, check=False)
        assert 'test commit' in result.stdout


def test_git_snapshot_commits_dirty_tree() -> None:
    git = git_executable()
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        _init_repo(repo)
        (repo / 'a.txt').write_text('one')
        ok, _ = git_snapshot(repo, 'snap')
        assert ok
        log = subprocess.run([git, 'log', '--oneline'], cwd=repo, capture_output=True, text=True, check=False)
        assert 'snap' in log.stdout


def test_git_snapshot_noop_on_clean_tree() -> None:
    git = git_executable()
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        _init_repo(repo)
        (repo / 'a.txt').write_text('one')
        subprocess.run([git, 'add', '-A'], cwd=repo, capture_output=True, check=False)
        subprocess.run([git, 'commit', '-m', 'init'], cwd=repo, capture_output=True, check=False)
        before = subprocess.run([git, 'rev-parse', 'HEAD'], cwd=repo, capture_output=True, text=True, check=False)
        ok, _ = git_snapshot(repo, 'snap')
        after = subprocess.run([git, 'rev-parse', 'HEAD'], cwd=repo, capture_output=True, text=True, check=False)
        assert ok
        assert before.stdout == after.stdout  # no new commit


def test_git_snapshot_falls_back_to_limen_identity() -> None:
    git = git_executable()
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as home:
        repo = Path(d)
        _init_repo(repo, with_identity=False)
        (repo / 'a.txt').write_text('one')
        # Hide any global/system identity so the first commit attempt has none.
        isolated = {
            **os.environ,
            'HOME': home,
            'GIT_CONFIG_GLOBAL': str(Path(home) / 'nonexistent'),
            'GIT_CONFIG_SYSTEM': str(Path(home) / 'nonexistent'),
        }
        with patch.dict(os.environ, isolated, clear=True):
            ok, err = git_snapshot(repo, 'snap')
        assert ok, err
        author = subprocess.run(
            [git, 'log', '-1', '--format=%an'], cwd=repo, capture_output=True, text=True, check=False,
        )
        assert author.stdout.strip() == 'limen'

import subprocess
import tempfile
from pathlib import Path

from limen.cli.git_utils import git_add_and_commit
from limen.cli.git_utils import git_executable


def test_git_executable_returns_path() -> None:
    path = git_executable()
    assert 'git' in path


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

import shutil
import subprocess
from pathlib import Path


def git_executable() -> str:

    '''
    Locate the git binary on PATH.

    Returns:
        str: Absolute path to the git executable

    Raises:
        FileNotFoundError: If git is not found on PATH

    '''

    git = shutil.which('git')
    if git is None:
        raise FileNotFoundError('git not found on PATH')
    return git


def git_add_and_commit(repo_root: Path, path: Path, message: str) -> bool:

    '''
    Stage a path and create a git commit inside a repository.

    Args:
        repo_root (Path): Root directory of the git repository
        path (Path): Path to stage (relative to repo_root)
        message (str): Commit message

    Returns:
        bool: True if both git add and git commit succeeded

    '''

    git = git_executable()
    add = subprocess.run([git, 'add', str(path)], cwd=repo_root, capture_output=True, check=False)
    if add.returncode != 0:
        return False
    commit = subprocess.run([git, 'commit', '-m', message], cwd=repo_root, capture_output=True, check=False)
    return commit.returncode == 0

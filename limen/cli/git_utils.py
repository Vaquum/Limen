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


def git_add_and_commit(repo_root: Path, path: Path, message: str) -> None:

    '''
    Stage a path and create a git commit inside a repository.

    Args:
        repo_root (Path): Root directory of the git repository
        path (Path): Path to stage (relative to repo_root)
        message (str): Commit message

    '''

    git = git_executable()
    subprocess.run([git, 'add', str(path)], cwd=repo_root, capture_output=True, check=False)
    subprocess.run([git, 'commit', '-m', message], cwd=repo_root, capture_output=True, check=False)

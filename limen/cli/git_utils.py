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

    try:
        git = git_executable()
    except FileNotFoundError:
        return False
    add = subprocess.run([git, 'add', str(path)], cwd=repo_root, capture_output=True, check=False)
    if add.returncode != 0:
        return False
    commit = subprocess.run([git, 'commit', '-m', message], cwd=repo_root, capture_output=True, check=False)
    return commit.returncode == 0


def git_push(repo_root: Path, remote_url: str) -> tuple[bool, str]:

    '''
    Push the current branch of a repository to a remote URL.

    Args:
        repo_root (Path): Root directory of the git repository
        remote_url (str): Destination git remote URL

    Returns:
        tuple[bool, str]: (succeeded, stderr) where stderr carries git's
            error output when the push fails

    Raises:
        FileNotFoundError: If git is not found on PATH

    '''

    git = git_executable()
    result = subprocess.run(
        [git, 'push', remote_url, 'HEAD'],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0, result.stderr.strip()


def git_clone(remote_url: str, dest: Path) -> tuple[bool, str]:

    '''
    Clone a git repository into a destination directory, preserving history.

    Args:
        remote_url (str): Source git remote URL
        dest (Path): Destination directory for the clone

    Returns:
        tuple[bool, str]: (succeeded, stderr) where stderr carries git's
            error output when the clone fails

    Raises:
        FileNotFoundError: If git is not found on PATH

    '''

    git = git_executable()
    result = subprocess.run(
        [git, 'clone', remote_url, str(dest)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0, result.stderr.strip()

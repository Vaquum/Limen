from pathlib import Path

import click

from limen.cli.git_utils import git_push
from limen.yaml.config import find_project_root
from limen.yaml.config import read_limen_toml


def run_backup(start: Path) -> bool:

    '''
    Push the project's committed history to the configured backup remote.

    Args:
        start (Path): Directory to start searching for the project root

    Returns:
        bool: True on success, False on failure

    '''

    project_root = find_project_root(start)
    if project_root is None:
        click.secho('  ✗ No limen project found. Run this command from inside a Limen project.', fg='red')
        return False

    try:
        config = read_limen_toml(project_root)
    except (FileNotFoundError, ValueError) as exc:
        click.secho(f"  ✗ Cannot read limen.toml: {exc}", fg='red')
        return False

    backup_remote = config.get('store', {}).get('backup_remote', '')
    if not isinstance(backup_remote, str) or not backup_remote:
        click.secho(
            '  ✗ No backup remote configured.\n'
            '    Set backup_remote in limen.toml, e.g. '
            'backup_remote = "git@github.com:user/my-project.git"',
            fg='red',
        )
        return False

    click.echo(f"Backing up to {backup_remote} ...")
    try:
        ok, error = git_push(project_root, backup_remote)
    except FileNotFoundError:
        click.secho('  ✗ git not found on PATH — install git and try again.', fg='red')
        return False

    if not ok:
        click.secho(f"  ✗ Backup failed: {error}", fg='red')
        return False

    click.secho('  ✓ Backed up', fg='green')
    return True

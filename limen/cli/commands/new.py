import shutil
import subprocess
from pathlib import Path

import click

from limen.cli.git_utils import git_executable


_TEMPLATE_REPO = 'https://github.com/Vaquum/limen-project-template.git'


def run_new(project_name: str, backup_remote: str | None) -> bool:

    '''
    Create a new Limen project from the official project template.

    Args:
        project_name (str): Name of the new project directory to create
        backup_remote (str | None): Git remote URL for manifest store backup

    Returns:
        bool: True on success, False on failure

    '''

    project_path = Path(project_name)

    if project_path.exists():
        click.secho(f"  ✗ '{project_name}' already exists.", fg='red')
        return False

    click.echo(f"Creating project '{project_name}' ...")

    if not _clone_template(project_path):
        return False

    if backup_remote is None:
        backup_remote = click.prompt(
            '  Backup remote URL (leave blank to skip)',
            default='',
            show_default=False,
        ).strip()

    if backup_remote:
        _write_backup_remote(project_path, backup_remote)

    click.secho(f"\n  ✓ Project '{project_name}' created.", fg='green')
    click.echo("\n  Next steps:")
    click.echo(f"    cd {project_name}")
    click.echo("    limen validate manifests/examples/logreg_binary.yaml")
    return True


def _clone_template(project_path: Path) -> bool:

    git = git_executable()
    click.echo('  Cloning template ...')
    result = subprocess.run(
        [git, 'clone', '--depth=1', _TEMPLATE_REPO, str(project_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        click.secho(f"  ✗ Clone failed: {result.stderr.strip()}", fg='red')
        return False

    shutil.rmtree(project_path / '.git')
    init = subprocess.run([git, 'init'], cwd=project_path, capture_output=True, check=False)
    if init.returncode != 0:
        shutil.rmtree(project_path)
        click.secho(f"  ✗ git init failed: {init.stderr.strip()}", fg='red')
        return False
    add = subprocess.run([git, 'add', '.'], cwd=project_path, capture_output=True, check=False)
    if add.returncode != 0:
        shutil.rmtree(project_path)
        click.secho(f"  ✗ git add failed: {add.stderr.strip()}", fg='red')
        return False
    commit = subprocess.run(
        [git, 'commit', '-m', 'feat: initial project from limen-project-template'],
        cwd=project_path, capture_output=True, check=False,
    )
    if commit.returncode != 0:
        click.secho(
            '  ⚠ Initial git commit failed — project created but not version-controlled.',
            fg='yellow',
        )
    return True


def _write_backup_remote(project_path: Path, remote_url: str) -> None:

    toml_path = project_path / 'limen.toml'
    text = toml_path.read_text(encoding='utf-8')
    updated = text.replace('backup_remote = ""', f'backup_remote = "{remote_url}"')
    if updated == text:
        click.secho('  ⚠ Could not set backup remote — limen.toml format unexpected.', fg='yellow')
        return
    toml_path.write_text(updated, encoding='utf-8')
    click.echo(f"  Backup remote set to: {remote_url}")

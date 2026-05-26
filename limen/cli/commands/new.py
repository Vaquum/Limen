import shutil
import subprocess
import sys
from pathlib import Path

import click


_TEMPLATE_REPO = 'https://github.com/Vaquum/limen-project-template.git'
_MIN_PYTHON = (3, 10)


def _git_executable() -> str:

    git = shutil.which('git')
    if git is None:
        raise FileNotFoundError('git not found on PATH')
    return git


def run_new(project_name: str, backup_remote: str | None) -> bool:

    '''
    Create a new Limen project from the official project template.

    Args:
        project_name (str): Name of the new project directory to create
        backup_remote (str | None): Git remote URL for manifest store backup

    Returns:
        bool: True on success, False on failure

    '''

    if not _check_python_version():
        return False

    project_path = Path(project_name)

    if project_path.exists():
        click.secho(f"  ✗ '{project_name}' already exists.", fg='red')
        return False

    click.echo(f"Creating project '{project_name}' ...")

    if not _clone_template(project_path):
        return False

    if not _create_venv(project_path):
        return False

    if not _install_limen(project_path):
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
    click.echo("    source venv/bin/activate")
    return True


def _check_python_version() -> bool:

    major, minor = sys.version_info[:2]
    min_major, min_minor = _MIN_PYTHON

    if (major, minor) < (min_major, min_minor):
        click.secho(
            f"  ✗ Python {min_major}.{min_minor}+ required. Found {major}.{minor}.",
            fg='red',
        )
        click.echo(
            f"  Install via pyenv: pyenv install {min_major}.{min_minor} "
            f"&& pyenv global {min_major}.{min_minor}"
        )
        return False
    return True


def _clone_template(project_path: Path) -> bool:

    git = _git_executable()
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
    subprocess.run([git, 'init'], cwd=project_path, capture_output=True, check=False)
    subprocess.run([git, 'add', '.'], cwd=project_path, capture_output=True, check=False)
    subprocess.run(
        [git, 'commit', '-m', 'feat: initial project from limen-project-template'],
        cwd=project_path,
        capture_output=True,
        check=False,
    )
    return True


def _create_venv(project_path: Path) -> bool:

    click.echo('  Creating virtual environment ...')
    result = subprocess.run(
        [sys.executable, '-m', 'venv', 'venv'],
        cwd=project_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        click.secho(f"  ✗ Failed to create venv: {result.stderr.strip()}", fg='red')
        return False
    return True


def _install_limen(project_path: Path) -> bool:

    click.echo('  Installing limen ...')
    venv_pip = project_path / 'venv' / 'bin' / 'pip'
    result = subprocess.run(
        [str(venv_pip), 'install', 'vaquum-limen', '--quiet'],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        click.secho(f"  ✗ Failed to install limen: {result.stderr.strip()}", fg='red')
        return False
    return True


def _write_backup_remote(project_path: Path, remote_url: str) -> None:

    toml_path = project_path / 'limen.toml'
    text = toml_path.read_text(encoding='utf-8')
    text = text.replace('backup_remote = ""', f'backup_remote = "{remote_url}"')
    toml_path.write_text(text, encoding='utf-8')
    click.echo(f"  Backup remote set to: {remote_url}")

import shutil
import subprocess
from pathlib import Path

import click


_TEMPLATE_REPO = 'https://github.com/Vaquum/limen-project-template.git'


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


def _write_backup_remote(project_path: Path, remote_url: str) -> None:

    toml_path = project_path / 'limen.toml'
    text = toml_path.read_text(encoding='utf-8')
    text = text.replace('backup_remote = ""', f'backup_remote = "{remote_url}"')
    toml_path.write_text(text, encoding='utf-8')
    click.echo(f"  Backup remote set to: {remote_url}")

import shutil
import subprocess
from pathlib import Path

import click

from limen.cli.git_utils import git_clone
from limen.cli.git_utils import git_executable


_TEMPLATE_REPO = 'https://github.com/Vaquum/limen-project-template.git'


def run_new(project_name: str,
            backup_remote: str | None,
            from_remote: str | None = None) -> bool:

    '''
    Create a new Limen project from the template or restore one from backup.

    Args:
        project_name (str): Name of the new project directory to create
        backup_remote (str | None): Git remote URL for manifest store backup
        from_remote (str | None): Backup remote to restore from; when set, the
            project is cloned from it with history intact instead of scaffolded
            from the template

    Returns:
        bool: True on success, False on failure

    '''

    project_path = Path(project_name)

    if project_path.exists():
        click.secho(f"  ✗ '{project_name}' already exists.", fg='red')
        return False

    if from_remote is not None:
        return _restore_from_backup(project_path, project_name, from_remote)

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
    click.echo("    limen list-templates")
    click.echo("    limen init logreg-first.yaml --template logreg_binary")
    click.echo("    limen validate logreg-first.yaml")
    return True


def _restore_from_backup(project_path: Path, project_name: str, from_remote: str) -> bool:

    click.echo(f"Restoring project '{project_name}' from {from_remote} ...")
    try:
        ok, error = git_clone(from_remote, project_path)
    except FileNotFoundError:
        click.secho('  ✗ git not found on PATH — install git and try again.', fg='red')
        return False
    if not ok:
        click.secho(f"  ✗ Restore failed: {error}", fg='red')
        return False

    if not (project_path / 'limen.toml').exists():
        click.secho(
            f"  ⚠ Restored '{project_name}', but no limen.toml found — "
            'this may not be a Limen project backup.',
            fg='yellow',
        )

    click.secho(f"\n  ✓ Project '{project_name}' restored.", fg='green')
    click.echo("\n  Next steps:")
    click.echo(f"    cd {project_name}")
    click.echo("    limen ls")
    return True


def _clone_template(project_path: Path) -> bool:

    try:
        git = git_executable()
    except FileNotFoundError:
        click.secho('  ✗ git not found on PATH — install git and try again.', fg='red')
        return False
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
    init = subprocess.run([git, 'init'], cwd=project_path, capture_output=True, text=True, check=False)
    if init.returncode != 0:
        shutil.rmtree(project_path)
        click.secho(f"  ✗ git init failed: {init.stderr.strip()}", fg='red')
        return False
    # Start on 'main' to match GitHub's default branch, so backup/restore line up.
    subprocess.run(
        [git, 'symbolic-ref', 'HEAD', 'refs/heads/main'],
        cwd=project_path, capture_output=True, check=False,
    )
    add = subprocess.run([git, 'add', '.'], cwd=project_path, capture_output=True, text=True, check=False)
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

    if '"' in remote_url or '\n' in remote_url:
        click.secho('  ⚠ Backup remote URL contains invalid characters — skipping.', fg='yellow')
        return
    toml_path = project_path / 'limen.toml'
    text = toml_path.read_text(encoding='utf-8')
    updated = text.replace('backup_remote = ""', f'backup_remote = "{remote_url}"')
    if updated == text:
        click.secho('  ⚠ Could not set backup remote — limen.toml format unexpected.', fg='yellow')
        return
    toml_path.write_text(updated, encoding='utf-8')
    click.echo(f"  Backup remote set to: {remote_url}")

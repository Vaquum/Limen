from pathlib import Path

import click

from limen.yaml.config import find_project_root
from limen.yaml.store import normalize_manifest_ref
from limen.yaml.store import resolve_manifest_uri


def run_show(ref: str, start: Path) -> bool:

    '''
    Print the stored YAML of a committed manifest.

    Args:
        ref (str): Manifest reference (bare hash, sha256:<hash>, or manifest:// URI)
        start (Path): Directory to start searching for the project root

    Returns:
        bool: True on success, False on failure

    '''

    project_root = find_project_root(start)
    if project_root is None:
        click.secho('  ✗ No limen project found. Run this command from inside a Limen project.', fg='red')
        return False

    try:
        candidate, _ = resolve_manifest_uri(normalize_manifest_ref(ref), project_root)
    except ValueError as exc:
        click.secho(f"  ✗ {exc}", fg='red')
        return False

    click.echo(candidate.read_text(encoding='utf-8'), nl=False)
    return True

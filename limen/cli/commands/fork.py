import re
from pathlib import Path

import click
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from limen.yaml.config import find_project_root
from limen.yaml.store import _SHA256_PREFIX
from limen.yaml.store import fork_manifest
from limen.yaml.store import normalize_manifest_ref
from limen.yaml.store import resolve_manifest_uri

_NAME_SLUG_RE = re.compile(r'^[A-Za-z0-9_-]+$')
_VERSION_SUFFIX_RE = re.compile(r'_v\d+$')


def run_fork(ref: str, name: str | None, start: Path) -> bool:

    '''
    Fork a committed manifest into a new development-mode working file.

    Args:
        ref (str): Manifest reference (bare hash, sha256:<hash>, or manifest:// URI)
        name (str | None): Name for the fork; prompted with a default if None
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
        click.secho(f'  ✗ {exc}', fg='red')
        return False

    manifests_dir = project_root / 'manifests'
    parent_name = _read_name(candidate, fallback=candidate.stem)

    if name is None:
        default = _default_fork_name(parent_name, manifests_dir)
        name = click.prompt('Name for the fork', default=default)

    if not _NAME_SLUG_RE.match(name):
        click.secho(
            f"  ✗ '{name}' is not a valid name. "
            'Use only letters, digits, underscores, or hyphens.',
            fg='red',
        )
        return False

    dest = manifests_dir / f'{name}.yaml'
    try:
        parent_id = fork_manifest(candidate, dest, name)
    except FileExistsError:
        click.secho(f"  ✗ '{dest}' already exists — choose another name.", fg='red')
        return False
    except ValueError as exc:
        click.secho(f'  ✗ {exc}', fg='red')
        return False

    short = parent_id[len(_SHA256_PREFIX):len(_SHA256_PREFIX) + 8]
    click.secho(f"  ✓ Forked from {_SHA256_PREFIX}{short} → {dest}", fg='green')
    click.echo('  Lineage to parent recorded. Next:')
    click.echo(f"    1. Edit {dest} (set metadata.mode: production when ready)")
    click.echo(f"    2. limen commit {dest}")
    return True


def _read_name(path: Path, fallback: str) -> str:

    yaml = YAML()
    try:
        data = yaml.load(path.read_text(encoding='utf-8'))
    except (OSError, YAMLError):
        return fallback
    if isinstance(data, dict):
        metadata = data.get('metadata')
        if isinstance(metadata, dict) and isinstance(metadata.get('name'), str):
            return metadata['name']
    return fallback


def _default_fork_name(base: str, manifests_dir: Path) -> str:

    root = _VERSION_SUFFIX_RE.sub('', base)
    n = 2
    while (manifests_dir / f'{root}_v{n}.yaml').exists():
        n += 1
    return f'{root}_v{n}'

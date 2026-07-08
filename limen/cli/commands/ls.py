from pathlib import Path

import click

from limen.yaml.config import find_project_root
from limen.yaml.store import MANIFEST_URI_SCHEME
from limen.yaml.store import SHA256_PREFIX
from limen.yaml.store import is_full_manifest_id
from limen.yaml.store import load_index
from limen.yaml.store import short_id


def run_ls(start: Path) -> bool:

    '''
    List all committed manifests in the project store.

    Args:
        start (Path): Directory to start searching for the project root

    Returns:
        bool: True on success, False if no project found

    '''

    project_root = find_project_root(start)
    if project_root is None:
        click.secho('  ✗ No limen project found. Run this command from inside a Limen project.', fg='red')
        return False

    try:
        index = load_index(project_root)
    except ValueError as exc:
        click.secho(f"  ⚠ {exc} — run limen reindex to rebuild it.", fg='yellow')
        return True

    manifests = index['manifests']

    if not manifests:
        click.echo('  No committed manifests yet. Use limen commit to add one.')
        return True

    _REQUIRED = {'id', 'name', 'committed_at'}
    click.echo(f"Committed manifests ({len(manifests)}):\n")
    for entry in manifests:
        entry_id = entry.get('id') if isinstance(entry, dict) else None
        if (not isinstance(entry, dict)
                or not _REQUIRED.issubset(entry)
                or not is_full_manifest_id(entry_id)
                or not isinstance(entry.get('name'), str)
                or not isinstance(entry.get('committed_at'), str)):
            click.secho('  ⚠ Skipping malformed entry in index.json.', fg='yellow')
            continue
        uri = f'{MANIFEST_URI_SCHEME}{SHA256_PREFIX}{short_id(entry_id)}'
        parent_id_val = entry.get('parent_id')
        parent = (f"  parent: {short_id(parent_id_val)}"
                  if is_full_manifest_id(parent_id_val) else '')
        click.echo(f"  {uri}  {entry['name']:<30}  {entry['committed_at']}{parent}")

    return True

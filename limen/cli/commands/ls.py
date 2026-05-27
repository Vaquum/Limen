import json
from pathlib import Path

import click

from limen.yaml.config import find_project_root
from limen.yaml.config import _STORE_RELATIVE
from limen.yaml.store import _MANIFEST_URI_SCHEME
from limen.yaml.store import _SHA256_PREFIX


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

    index_path = project_root / _STORE_RELATIVE / 'index.json'
    if not index_path.exists():
        click.echo('  No committed manifests yet. Use limen commit to add one.')
        return True

    try:
        index = json.loads(index_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        click.secho('  ⚠ index.json is corrupted — run limen commit to repair it.', fg='yellow')
        return True

    if not isinstance(index, dict) or not isinstance(index.get('manifests'), list):
        click.secho('  ⚠ index.json has invalid structure — run limen commit to repair it.', fg='yellow')
        return True

    manifests = index['manifests']

    if not manifests:
        click.echo('  No committed manifests yet. Use limen commit to add one.')
        return True

    _REQUIRED = {'id', 'name', 'committed_at'}
    click.echo(f"Committed manifests ({len(manifests)}):\n")
    p = len(_SHA256_PREFIX)
    for entry in manifests:
        entry_id = entry.get('id') if isinstance(entry, dict) else None
        if (not isinstance(entry, dict)
                or not _REQUIRED.issubset(entry)
                or not isinstance(entry_id, str)
                or not entry_id.startswith(_SHA256_PREFIX)
                or not isinstance(entry.get('name'), str)
                or not isinstance(entry.get('committed_at'), str)):
            click.secho('  ⚠ Skipping malformed entry in index.json.', fg='yellow')
            continue
        short_id = entry_id[p:p + 8]
        uri = f'{_MANIFEST_URI_SCHEME}{_SHA256_PREFIX}{short_id}'
        parent_id_val = entry.get('parent_id')
        parent = (f"  parent: {parent_id_val[p:p + 8]}"
                  if isinstance(parent_id_val, str) and parent_id_val.startswith(_SHA256_PREFIX)
                  else '')
        click.echo(f"  {uri}  {entry['name']:<30}  {entry['committed_at']}{parent}")

    return True

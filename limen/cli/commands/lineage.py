from pathlib import Path
from typing import Any

import click

from limen.yaml.config import find_project_root
from limen.yaml.store import _SHA256_PREFIX
from limen.yaml.store import load_index
from limen.yaml.store import normalize_manifest_ref
from limen.yaml.store import resolve_manifest_uri


def run_lineage(ref: str, start: Path) -> bool:

    '''
    Display the lineage chain for a committed manifest.

    Walks the parent_id chain from the target up to the root and prints it
    root-first, with the target marked.

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
        click.secho(f'  ✗ {exc}', fg='red')
        return False

    target_id = f'{_SHA256_PREFIX}{candidate.stem}'

    try:
        index = load_index(project_root)
    except ValueError as exc:
        click.secho(f'  ✗ {exc}', fg='red')
        return False

    by_id = {
        m['id']: m for m in index['manifests']
        if isinstance(m, dict) and isinstance(m.get('id'), str)
    }

    chain: list[tuple[str, dict[str, Any] | None]] = []
    seen: set[str] = set()
    current: str | None = target_id
    while current is not None and current not in seen:
        seen.add(current)
        entry = by_id.get(current)
        chain.append((current, entry))
        current = entry.get('parent_id') if isinstance(entry, dict) else None

    chain.reverse()

    target_name = _name_of(by_id.get(target_id), fallback=candidate.stem)
    short_target = target_id[len(_SHA256_PREFIX):len(_SHA256_PREFIX) + 8]
    click.echo(f"Lineage for {_SHA256_PREFIX}{short_target} ({target_name}):\n")

    for depth, (manifest_id, entry) in enumerate(chain):
        short = manifest_id[len(_SHA256_PREFIX):len(_SHA256_PREFIX) + 8]
        name = _name_of(entry, fallback='(not in store)')
        committed_at = entry.get('committed_at', '') if isinstance(entry, dict) else ''
        connector = '' if depth == 0 else '└─ '
        indent = '  ' + '   ' * depth
        marker = '  ← target' if manifest_id == target_id else ''
        click.echo(f"{indent}{connector}{_SHA256_PREFIX}{short}  {name:<28}  {committed_at}{marker}")

    return True


def _name_of(entry: dict[str, Any] | None, fallback: str) -> str:

    if isinstance(entry, dict) and isinstance(entry.get('name'), str):
        return entry['name']
    return fallback

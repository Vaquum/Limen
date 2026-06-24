from pathlib import Path

import click

from limen.cli.commands._load_yaml import load_and_validate
from limen.cli.git_utils import git_add_and_commit
from limen.yaml.config import find_project_root
from limen.yaml.store import _SHA256_PREFIX
from limen.yaml.store import _lineage_block
from limen.yaml.store import commit_manifest
from limen.yaml.store import is_full_manifest_id
from limen.yaml.store import manifest_name
from limen.yaml.store import short_id


def run_commit(yaml_path: Path, parent_id: str | None, message: str | None) -> bool:

    '''
    Validate, store, and git-commit a YAML manifest.

    Args:
        yaml_path (Path): Path to the source YAML file
        parent_id (str | None): Parent manifest ID for lineage tracking
        message (str | None): Custom git commit message; auto-generated if None

    Returns:
        bool: True on success, False on failure

    '''

    project_root = find_project_root(yaml_path.parent)
    if project_root is None:
        click.secho(
            '  ✗ No limen project found. '
            'The YAML file must be inside a Limen project directory (containing limen.toml).',
            fg='red',
        )
        return False

    click.echo(f"Validating {yaml_path.name} ...")
    yaml_dict, valid = load_and_validate(yaml_path)
    if not valid:
        return False

    if parent_id is None:
        lineage_parent = _lineage_block(yaml_dict).get('parent_id')
        if isinstance(lineage_parent, str):
            parent_id = lineage_parent

    if parent_id is not None and not is_full_manifest_id(parent_id):
        click.secho(
            f"  ✗ Invalid parent ID: '{parent_id}'\n"
            '    Expected sha256:<64-hex-chars>.',
            fg='red',
        )
        return False

    mode = yaml_dict.get('metadata', {}).get('mode', 'development')
    if mode != 'production':
        click.secho(
            '  ✗ Cannot commit a development-mode manifest.\n'
            '    Set metadata.mode: production before committing.',
            fg='red',
        )
        return False

    manifest_id, already_existed = commit_manifest(yaml_path, project_root, parent_id)
    short = f'{_SHA256_PREFIX}{short_id(manifest_id)}'

    if already_existed:
        repair_msg = f'repair: restore index for {short}'
        if git_add_and_commit(project_root, Path('manifests') / 'committed', repair_msg):
            click.secho(f"\n  ✓ Repaired and committed {manifest_id}", fg='green')
        else:
            click.secho(f"\n  Already in store: {manifest_id}", fg='yellow')
        click.echo(f"  Run with: limen run {manifest_id}")
        return True

    name = manifest_name(yaml_dict, fallback=yaml_path.stem)
    commit_msg = message or f'commit: {name} ({short})'
    git_ok = git_add_and_commit(project_root, Path('manifests') / 'committed', commit_msg)

    if not git_ok:
        click.secho(
            f"\n  ✓ Stored {manifest_id}\n"
            '  ⚠ Git commit failed — manifest is stored but not version-controlled.',
            fg='yellow',
        )
    else:
        click.secho(f"\n  ✓ Committed {manifest_id}", fg='green')
    click.echo(f"  Run with: limen run {manifest_id}")
    return True

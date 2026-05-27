import hashlib
import json
import re
import warnings
from datetime import datetime
from datetime import timezone
from io import StringIO
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from limen.yaml.config import _STORE_RELATIVE
from limen.yaml.config import find_project_root


_SHA256_PREFIX = 'sha256:'
_MANIFEST_URI_SCHEME = 'manifest://'
_SHA256_HEX_LENGTH = 64


def commit_manifest(yaml_path: Path,
                    project_root: Path,
                    parent_id: str | None = None) -> tuple[str, bool]:

    '''
    Content-address and store a YAML manifest in the committed store.

    Reads the source YAML, computes its SHA256, injects a lineage block,
    writes to manifests/committed/<hex>.yaml, and updates index.json.

    Args:
        yaml_path (Path): Path to the source YAML file
        project_root (Path): Root directory of the limen project
        parent_id (str | None): Parent manifest ID for lineage tracking

    Returns:
        tuple[str, bool]: (manifest_id, already_existed). manifest_id is
            "sha256:<hex>". already_existed is True if the manifest was
            already in the store (idempotent).

    '''

    content = yaml_path.read_text(encoding='utf-8')
    hex_hash = hashlib.sha256(content.encode()).hexdigest()
    manifest_id = f'{_SHA256_PREFIX}{hex_hash}'

    store_path = project_root / _STORE_RELATIVE
    dest = store_path / f'{hex_hash}.yaml'

    already_existed = dest.exists()

    yaml = YAML()
    yaml.preserve_quotes = True

    if not already_existed:
        data = yaml.load(content)
        name = str(data.get('metadata', {}).get('name', yaml_path.stem))
        committed_at = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        lineage: dict = {'id': manifest_id, 'committed_at': committed_at}
        if parent_id is not None:
            lineage['parent_id'] = parent_id
        data['lineage'] = lineage
        stream = StringIO()
        yaml.dump(data, stream)
        store_path.mkdir(parents=True, exist_ok=True)
        dest.write_text(stream.getvalue(), encoding='utf-8')
        stored_parent_id = parent_id
    else:
        existing = yaml.load(dest.read_text(encoding='utf-8'))
        name = str(existing.get('metadata', {}).get('name', dest.stem))
        committed_at = existing.get('lineage', {}).get('committed_at', '')
        stored_parent_id = existing.get('lineage', {}).get('parent_id')

    entry: dict[str, Any] = {
        'id': manifest_id,
        'name': name,
        'committed_at': committed_at,
        'parent_id': stored_parent_id,
        'file': f'{hex_hash}.yaml',
    }
    _update_index(store_path, entry)

    return manifest_id, already_existed


def resolve_manifest_uri(uri: str, start: Path) -> tuple[Path, Path]:

    '''
    Resolve a manifest:// URI to a filesystem path.

    Verifies that lineage.id in the committed file matches the URI hash,
    catching cases where the wrong file is at the expected path.

    Args:
        uri (str): URI of the form manifest://sha256:<hex>
        start (Path): Directory to start searching for the project root

    Returns:
        tuple[Path, Path]: (manifest_path, project_root) — resolved path to the
            committed manifest file and the limen project root

    Raises:
        ValueError: If the URI is malformed, the project root is not found,
            the manifest is not in the store, or lineage.id does not match

    '''

    if not uri.startswith(_MANIFEST_URI_SCHEME):
        raise ValueError(f"Not a manifest URI: '{uri}'")

    ref = uri[len(_MANIFEST_URI_SCHEME):]
    if not ref.startswith(_SHA256_PREFIX):
        raise ValueError(f"Manifest URI must use sha256 scheme: '{uri}'")

    hex_hash = ref[len(_SHA256_PREFIX):]

    if not re.fullmatch(r'[0-9a-f]{1,64}', hex_hash):
        raise ValueError(f"Malformed hash in manifest URI: '{uri}'")

    project_root = find_project_root(start)
    if project_root is None:
        raise ValueError('No limen project found. Run from inside a project directory.')

    store_path = project_root / _STORE_RELATIVE

    if len(hex_hash) == _SHA256_HEX_LENGTH:
        candidate = store_path / f'{hex_hash}.yaml'
        if not candidate.exists():
            raise ValueError(f"Manifest not found in store: '{uri}'")
    else:
        matches = list(store_path.glob(f'{hex_hash}*.yaml'))
        if not matches:
            raise ValueError(f"Manifest not found in store: '{uri}'")
        if len(matches) > 1:
            shorts = ', '.join(p.stem[:12] for p in matches)
            raise ValueError(f"Ambiguous short hash '{hex_hash}' matches multiple manifests: {shorts}")
        candidate = matches[0]

    full_hex = candidate.stem

    yaml_obj = YAML()
    data = yaml_obj.load(candidate.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest format in '{candidate.name}'")
    expected_id = f'{_SHA256_PREFIX}{full_hex}'
    actual_id = data.get('lineage', {}).get('id')
    if actual_id != expected_id:
        raise ValueError(
            f"Integrity check failed: '{candidate.name}' lineage.id "
            f"does not match expected '{expected_id}'"
        )

    return candidate, project_root


def _update_index(store_path: Path, entry: dict[str, Any]) -> None:

    index_path = store_path / 'index.json'
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            warnings.warn(f"index.json is corrupted — reinitializing: {index_path}", stacklevel=2)
            index = {'version': 1, 'manifests': []}
    else:
        index = {'version': 1, 'manifests': []}

    if not any(m['id'] == entry['id'] for m in index['manifests']):
        index['manifests'].append(entry)
    index_path.write_text(json.dumps(index, indent=2), encoding='utf-8')

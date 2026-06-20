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
from ruamel.yaml.error import YAMLError

from limen.yaml.config import _STORE_RELATIVE
from limen.yaml.config import find_project_root


_SHA256_PREFIX = 'sha256:'
_MANIFEST_URI_SCHEME = 'manifest://'
_SHA256_HEX_LENGTH = 64
_YAML_DUMP_WIDTH = 4096


def canonical_manifest_id(yaml_dict: dict[str, Any]) -> str:

    '''
    Compute the content-addressed manifest ID for a YAML dict.

    Strips any ``lineage`` key before hashing so commit-time and run-time
    identifiers remain identical regardless of whether a lineage block has
    been injected.

    Args:
        yaml_dict (dict[str, Any]): Parsed YAML manifest as a plain dict

    Returns:
        str: ``sha256:<64-hex>`` identifier

    '''

    data_no_lineage = {k: v for k, v in yaml_dict.items() if k != 'lineage'}
    canonical = json.dumps(data_no_lineage, sort_keys=True, default=str)
    return f'{_SHA256_PREFIX}{hashlib.sha256(canonical.encode()).hexdigest()}'


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

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = _YAML_DUMP_WIDTH

    try:
        data = yaml.load(content)
    except YAMLError as exc:
        raise ValueError(f"Cannot parse YAML '{yaml_path.name}': {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML format in '{yaml_path.name}': expected a mapping")

    manifest_id = canonical_manifest_id(data)
    hex_hash = manifest_id[len(_SHA256_PREFIX):]

    store_path = project_root / _STORE_RELATIVE
    dest = store_path / f'{hex_hash}.yaml'

    already_existed = dest.exists()

    if not already_existed:
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
        try:
            existing = yaml.load(dest.read_text(encoding='utf-8'))
        except (OSError, YAMLError) as exc:
            raise ValueError(f"Cannot read committed manifest '{dest.name}': {exc}") from exc
        if not isinstance(existing, dict):
            raise ValueError(f"Invalid committed manifest format in '{dest.name}': expected a mapping")
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
    try:
        data = yaml_obj.load(candidate.read_text(encoding='utf-8'))
    except (OSError, YAMLError) as exc:
        raise ValueError(f"Cannot read manifest '{candidate.name}': {exc}") from exc
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


def is_full_manifest_id(value: Any) -> bool:

    '''
    Check whether a value is a well-formed sha256:<64-hex> manifest ID.

    Args:
        value (Any): Candidate manifest ID

    Returns:
        bool: True if value is a full sha256 manifest ID

    '''

    if not isinstance(value, str) or not value.startswith(_SHA256_PREFIX):
        return False
    hex_part = value[len(_SHA256_PREFIX):]
    return len(hex_part) == _SHA256_HEX_LENGTH and all(c in '0123456789abcdef' for c in hex_part)


def normalize_manifest_ref(ref: str) -> str:

    '''
    Normalize a user-supplied manifest reference to a manifest:// URI.

    Accepts a bare hash, a sha256:<hash> reference, or a full
    manifest://sha256:<hash> URI and returns the canonical URI form.

    Args:
        ref (str): User-supplied manifest reference

    Returns:
        str: A manifest://sha256:<hash> URI

    '''

    ref = ref.strip()
    if ref.startswith(_MANIFEST_URI_SCHEME):
        return ref
    if ref.startswith(_SHA256_PREFIX):
        return f'{_MANIFEST_URI_SCHEME}{ref}'
    return f'{_MANIFEST_URI_SCHEME}{_SHA256_PREFIX}{ref}'


def load_index(project_root: Path) -> dict[str, Any]:

    '''
    Read the manifest store index, returning an empty index when absent.

    Args:
        project_root (Path): Root directory of the limen project

    Returns:
        dict[str, Any]: Parsed index with 'version' and 'manifests' keys

    Raises:
        ValueError: If index.json exists but is corrupted or malformed

    '''

    index_path = project_root / _STORE_RELATIVE / 'index.json'
    if not index_path.exists():
        return {'version': 1, 'manifests': []}
    try:
        index = json.loads(index_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(f"index.json is corrupted: {index_path}") from exc
    if not isinstance(index, dict) or not isinstance(index.get('manifests'), list):
        raise ValueError(f"index.json has invalid structure: {index_path}")
    return index


def fork_manifest(committed_path: Path, dest: Path, new_name: str) -> str:

    '''
    Create a development-mode working copy of a committed manifest.

    Copies the committed manifest to ``dest``, sets metadata.name to
    ``new_name``, switches metadata.mode to development, and records the
    source manifest as lineage.parent_id so the lineage is preserved on the
    next commit.

    Args:
        committed_path (Path): Path to the committed manifest file
        dest (Path): Destination path for the working copy
        new_name (str): Value to set for metadata.name

    Returns:
        str: The parent manifest ID recorded in the working copy

    Raises:
        FileExistsError: If dest already exists
        ValueError: If the committed manifest has no metadata mapping

    '''

    if dest.exists():
        raise FileExistsError(str(dest))

    parent_id = f'{_SHA256_PREFIX}{committed_path.stem}'

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = _YAML_DUMP_WIDTH
    data = yaml.load(committed_path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest format in '{committed_path.name}'")

    metadata = data.get('metadata')
    if not isinstance(metadata, dict):
        raise ValueError(f"Committed manifest '{committed_path.name}' has no metadata block")

    metadata['name'] = new_name
    metadata['mode'] = 'development'
    data['lineage'] = {'parent_id': parent_id}

    stream = StringIO()
    yaml.dump(data, stream)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(stream.getvalue(), encoding='utf-8')

    return parent_id


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

    if not isinstance(index, dict) or not isinstance(index.get('manifests'), list):
        warnings.warn(f"index.json has invalid structure — reinitializing: {index_path}", stacklevel=2)
        index = {'version': 1, 'manifests': []}

    index['manifests'] = [m for m in index['manifests'] if not (isinstance(m, dict) and m.get('id') == entry['id'])]
    index['manifests'].append(entry)
    index_path.write_text(json.dumps(index, indent=2), encoding='utf-8')

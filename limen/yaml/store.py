import hashlib
import json
from datetime import datetime
from io import StringIO
from pathlib import Path

from ruamel.yaml import YAML


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
    manifest_id = f'sha256:{hex_hash}'

    store_path = project_root / 'manifests' / 'committed'
    dest = store_path / f'{hex_hash}.yaml'

    if dest.exists():
        return manifest_id, True

    yaml = YAML()
    yaml.preserve_quotes = True
    data = yaml.load(content)
    name = str(data.get('metadata', {}).get('name', yaml_path.stem))
    committed_at = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')

    lineage: dict = {'id': manifest_id, 'committed_at': committed_at}
    if parent_id is not None:
        lineage['parent_id'] = parent_id
    data['lineage'] = lineage

    stream = StringIO()
    yaml.dump(data, stream)

    store_path.mkdir(parents=True, exist_ok=True)
    dest.write_text(stream.getvalue(), encoding='utf-8')
    _update_index(store_path, manifest_id, name, hex_hash, committed_at, parent_id)

    return manifest_id, False


def _update_index(store_path: Path,
                  manifest_id: str,
                  name: str,
                  hex_hash: str,
                  committed_at: str,
                  parent_id: str | None) -> None:

    index_path = store_path / 'index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding='utf-8'))
    else:
        index = {'version': 1, 'manifests': []}

    index['manifests'].append({
        'id': manifest_id,
        'name': name,
        'committed_at': committed_at,
        'parent_id': parent_id,
        'file': f'{hex_hash}.yaml',
    })
    index_path.write_text(json.dumps(index, indent=2), encoding='utf-8')

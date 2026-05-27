import json
import tempfile
from pathlib import Path

import pytest

from limen.yaml.store import _SHA256_PREFIX
from limen.yaml.store import commit_manifest
from limen.yaml.store import resolve_manifest_uri


_SAMPLE_YAML = '''\
schema_version: "1.0"
metadata:
  name: test_experiment
  mode: development
sfd:
  manifest:
    type: ml
    data_source:
      method: limen.data.get_spot_klines
    reference_architecture: limen.sfd.reference_architecture.logreg_binary.logreg_binary
    target:
      name: target
      class: limen.targets.QuantileBinaryTarget
  params:
    lookback: [12, 24]
uel:
  n_permutations: 4
'''


def _make_project(tmp: Path) -> tuple[Path, Path]:
    (tmp / 'limen.toml').write_text('[store]\nbackup_remote = ""\n')
    (tmp / 'manifests' / 'committed').mkdir(parents=True)
    yaml_path = tmp / 'manifests' / 'examples' / 'test.yaml'
    yaml_path.parent.mkdir(parents=True)
    yaml_path.write_text(_SAMPLE_YAML)
    return tmp, yaml_path


def test_commit_stores_file_in_committed_dir() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, yaml_path = _make_project(Path(d))
        manifest_id, _ = commit_manifest(yaml_path, project_root)
        hex_hash = manifest_id[len(_SHA256_PREFIX):]
        assert (project_root / 'manifests' / 'committed' / f'{hex_hash}.yaml').exists()


def test_commit_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, yaml_path = _make_project(Path(d))
        id1, existed1 = commit_manifest(yaml_path, project_root)
        id2, existed2 = commit_manifest(yaml_path, project_root)
        assert id1 == id2
        assert not existed1
        assert existed2


def test_commit_injects_lineage_section() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, yaml_path = _make_project(Path(d))
        manifest_id, _ = commit_manifest(yaml_path, project_root)
        hex_hash = manifest_id[len(_SHA256_PREFIX):]
        committed = (project_root / 'manifests' / 'committed' / f'{hex_hash}.yaml').read_text()
        assert 'lineage:' in committed
        assert manifest_id in committed


def test_commit_updates_index_json() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, yaml_path = _make_project(Path(d))
        manifest_id, _ = commit_manifest(yaml_path, project_root)
        index = json.loads((project_root / 'manifests' / 'committed' / 'index.json').read_text())
        assert index['version'] == 1
        assert index['manifests'][0]['id'] == manifest_id
        assert index['manifests'][0]['name'] == 'test_experiment'


def test_commit_recovers_from_corrupt_index() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, yaml_path = _make_project(Path(d))
        commit_manifest(yaml_path, project_root)
        (project_root / 'manifests' / 'committed' / 'index.json').write_text('not valid json')
        yaml_path2 = project_root / 'manifests' / 'examples' / 'test2.yaml'
        yaml_path2.write_text(_SAMPLE_YAML + '\n# extra\n')
        manifest_id2, _ = commit_manifest(yaml_path2, project_root)
        index = json.loads((project_root / 'manifests' / 'committed' / 'index.json').read_text())
        assert any(m['id'] == manifest_id2 for m in index['manifests'])


def test_commit_repairs_index_when_missing() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, yaml_path = _make_project(Path(d))
        manifest_id, _ = commit_manifest(yaml_path, project_root)
        (project_root / 'manifests' / 'committed' / 'index.json').unlink()
        _, already_existed = commit_manifest(yaml_path, project_root)
        assert already_existed
        index = json.loads((project_root / 'manifests' / 'committed' / 'index.json').read_text())
        assert any(m['id'] == manifest_id for m in index['manifests'])


def test_commit_timestamps_are_utc() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, yaml_path = _make_project(Path(d))
        commit_manifest(yaml_path, project_root)
        index = json.loads((project_root / 'manifests' / 'committed' / 'index.json').read_text())
        assert index['manifests'][0]['committed_at'].endswith('Z')


def test_resolve_manifest_uri_returns_path_and_project_root() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, yaml_path = _make_project(Path(d))
        manifest_id, _ = commit_manifest(yaml_path, project_root)
        uri = f'manifest://{manifest_id}'
        resolved, returned_root = resolve_manifest_uri(uri, project_root)
        assert resolved.exists()
        assert returned_root == project_root


def test_commit_recovers_from_structurally_invalid_index() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, yaml_path = _make_project(Path(d))
        index_path = project_root / 'manifests' / 'committed' / 'index.json'
        index_path.write_text(json.dumps([1, 2, 3]))
        manifest_id, _ = commit_manifest(yaml_path, project_root)
        index = json.loads(index_path.read_text())
        assert any(m['id'] == manifest_id for m in index['manifests'])


def test_resolve_manifest_uri_raises_on_corrupt_manifest_file() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, yaml_path = _make_project(Path(d))
        manifest_id, _ = commit_manifest(yaml_path, project_root)
        hex_hash = manifest_id[len(_SHA256_PREFIX):]
        committed_path = project_root / 'manifests' / 'committed' / f'{hex_hash}.yaml'
        committed_path.write_text(': invalid: yaml: {{{')
        with pytest.raises(ValueError, match='Cannot read manifest'):
            resolve_manifest_uri(f'manifest://{manifest_id}', project_root)


def test_resolve_manifest_uri_rejects_malformed_hash() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, _ = _make_project(Path(d))
        with pytest.raises(ValueError, match='Malformed hash'):
            resolve_manifest_uri('manifest://sha256:*', project_root)
        with pytest.raises(ValueError, match='Malformed hash'):
            resolve_manifest_uri('manifest://sha256:', project_root)
        with pytest.raises(ValueError, match='Malformed hash'):
            resolve_manifest_uri('manifest://sha256:' + 'a' * 65, project_root)


def test_resolve_manifest_uri_rejects_missing_manifest() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, _ = _make_project(Path(d))
        with pytest.raises(ValueError, match='not found in store'):
            resolve_manifest_uri('manifest://sha256:' + 'a' * 64, project_root)


def test_resolve_manifest_uri_rejects_tampered_lineage() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, yaml_path = _make_project(Path(d))
        manifest_id, _ = commit_manifest(yaml_path, project_root)
        hex_hash = manifest_id[len(_SHA256_PREFIX):]
        committed_path = project_root / 'manifests' / 'committed' / f'{hex_hash}.yaml'
        tampered = committed_path.read_text().replace(manifest_id, 'sha256:' + 'b' * 64)
        committed_path.write_text(tampered)
        with pytest.raises(ValueError, match='Integrity check failed'):
            resolve_manifest_uri(f'manifest://{manifest_id}', project_root)


def test_resolve_manifest_uri_accepts_short_hash() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, yaml_path = _make_project(Path(d))
        manifest_id, _ = commit_manifest(yaml_path, project_root)
        short = manifest_id[len(_SHA256_PREFIX):len(_SHA256_PREFIX) + 8]
        resolved, _ = resolve_manifest_uri(f'manifest://sha256:{short}', project_root)
        assert resolved.exists()


def test_resolve_manifest_uri_rejects_ambiguous_short_hash() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, yaml_path = _make_project(Path(d))
        commit_manifest(yaml_path, project_root)
        store = project_root / 'manifests' / 'committed'
        existing = next(store.glob('*.yaml'))
        first_char = existing.stem[0]
        # plant a second file starting with the same char but a different full hash
        clone_stem = first_char + ('0' * 63 if existing.stem != first_char + '0' * 63 else 'f' * 63)
        (store / f'{clone_stem}.yaml').write_text(existing.read_text())
        with pytest.raises(ValueError, match='Ambiguous'):
            resolve_manifest_uri(f'manifest://sha256:{first_char}', project_root)

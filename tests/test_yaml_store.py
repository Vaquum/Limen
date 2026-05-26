import json
import tempfile
from pathlib import Path

from limen.yaml.store import commit_manifest


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
        hex_hash = manifest_id.replace('sha256:', '')
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
        hex_hash = manifest_id.replace('sha256:', '')
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

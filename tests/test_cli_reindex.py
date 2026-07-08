import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from limen.cli.commands.reindex import run_reindex
from limen.yaml.config import STORE_RELATIVE
from limen.yaml.store import commit_manifest
from limen.yaml.store import load_index
from limen.yaml.store import rebuild_index


_SAMPLE_YAML = '''\
schema_version: "1.0"
metadata:
  name: indexed_experiment
  mode: production
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


def _make_committed(root: Path) -> str:
    root = root.resolve()
    (root / 'limen.toml').write_text('')
    (root / 'manifests' / 'committed').mkdir(parents=True)
    yaml_path = root / 'manifests' / 'src.yaml'
    yaml_path.write_text(_SAMPLE_YAML)
    manifest_id, _ = commit_manifest(yaml_path, root)
    return manifest_id


def test_run_reindex_fails_when_no_project() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        assert run_reindex(Path(d)) is False


def test_rebuild_index_recovers_deleted_index() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d).resolve()
        manifest_id = _make_committed(root)
        index_path = root / STORE_RELATIVE / 'index.json'
        index_path.unlink()

        count, warnings = rebuild_index(root)

        assert count == 1
        assert warnings == []
        index = load_index(root)
        assert index['manifests'][0]['id'] == manifest_id
        assert index['manifests'][0]['name'] == 'indexed_experiment'


def test_rebuild_index_recovers_corrupt_index() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d).resolve()
        _make_committed(root)
        index_path = root / STORE_RELATIVE / 'index.json'
        index_path.write_text('{ broken json', encoding='utf-8')

        count, _ = rebuild_index(root)

        assert count == 1
        assert load_index(root)['manifests'][0]['name'] == 'indexed_experiment'


def test_rebuild_index_preserves_parent_id() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d).resolve()
        parent_id = _make_committed(root)
        # Commit a child carrying lineage.parent_id.
        child = root / 'manifests' / 'child.yaml'
        child_text = _SAMPLE_YAML.replace('name: indexed_experiment', 'name: child_experiment')
        child_text = child_text.replace('lookback: [12, 24]', 'lookback: [36, 48]')
        child_text += f'lineage:\n  parent_id: {parent_id}\n'
        child.write_text(child_text)
        child_id, _ = commit_manifest(child, root)

        (root / STORE_RELATIVE / 'index.json').unlink()
        rebuild_index(root)

        index = load_index(root)
        by_id = {m['id']: m for m in index['manifests']}
        assert by_id[child_id]['parent_id'] == parent_id
        assert by_id[parent_id]['parent_id'] is None


def test_rebuild_index_skips_file_with_mismatched_id() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d).resolve()
        _make_committed(root)
        store = root / STORE_RELATIVE
        # A committed file whose lineage.id does not match its filename.
        bogus = store / ('a' * 64 + '.yaml')
        bogus.write_text(_SAMPLE_YAML + 'lineage:\n  id: sha256:' + 'b' * 64 + '\n')

        count, warnings = rebuild_index(root)

        assert count == 1
        assert any('does not match filename' in w for w in warnings)


def test_rebuild_index_skips_file_without_lineage() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d).resolve()
        _make_committed(root)
        store = root / STORE_RELATIVE
        (store / ('c' * 64 + '.yaml')).write_text(_SAMPLE_YAML)

        count, warnings = rebuild_index(root)

        assert count == 1
        assert any('missing lineage block' in w for w in warnings)


def test_run_reindex_returns_true_and_writes_index() -> None:
    with tempfile.TemporaryDirectory() as d, patch('click.echo'), patch('click.secho'):
        root = Path(d).resolve()
        _make_committed(root)
        (root / STORE_RELATIVE / 'index.json').unlink()
        assert run_reindex(root) is True
        index = json.loads((root / STORE_RELATIVE / 'index.json').read_text())
        assert len(index['manifests']) == 1

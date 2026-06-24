import json
import tempfile
from pathlib import Path

import pytest

from limen.yaml.store import _SHA256_PREFIX
from limen.yaml.store import commit_manifest
from limen.yaml.store import fork_manifest
from limen.yaml.store import is_full_manifest_id
from limen.yaml.store import load_index
from limen.yaml.store import normalize_manifest_ref
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
    tmp = tmp.resolve()
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
        with pytest.warns(UserWarning, match='index.json is corrupted'):
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
        with pytest.warns(UserWarning, match='index.json has invalid structure'):
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


def test_commit_manifest_raises_on_invalid_yaml_content() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, _ = _make_project(Path(d))
        bad_yaml = project_root / 'bad.yaml'
        bad_yaml.write_text(': invalid: yaml: {', encoding='utf-8')
        with pytest.raises(ValueError, match='Cannot parse YAML'):
            commit_manifest(bad_yaml, project_root)


def test_commit_manifest_raises_when_yaml_is_not_a_dict() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, _ = _make_project(Path(d))
        list_yaml = project_root / 'list.yaml'
        list_yaml.write_text('- item1\n- item2\n', encoding='utf-8')
        with pytest.raises(ValueError, match='expected a mapping'):
            commit_manifest(list_yaml, project_root)


def test_commit_manifest_raises_when_existing_committed_file_is_not_a_dict() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, yaml_path = _make_project(Path(d))
        manifest_id, _ = commit_manifest(yaml_path, project_root)
        hex_hash = manifest_id[len(_SHA256_PREFIX):]
        store = project_root / 'manifests' / 'committed'
        (store / f'{hex_hash}.yaml').write_text('- corrupted\n- list\n', encoding='utf-8')
        with pytest.raises(ValueError, match='expected a mapping'):
            commit_manifest(yaml_path, project_root)


def test_commit_manifest_tolerates_scalar_lineage() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, yaml_path = _make_project(Path(d))
        # A hand-edited manifest with a scalar (non-mapping) lineage block passes
        # validation; committing must degrade gracefully, not raise AttributeError.
        yaml_path.write_text(_SAMPLE_YAML + 'lineage: foo\n')
        manifest_id, _ = commit_manifest(yaml_path, project_root)
        assert manifest_id.startswith(_SHA256_PREFIX)
        entry = next(m for m in load_index(project_root)['manifests'] if m['id'] == manifest_id)
        assert entry['parent_id'] is None


def test_update_index_deduplicates_pre_existing_entries() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, yaml_path = _make_project(Path(d))
        manifest_id, _ = commit_manifest(yaml_path, project_root)
        store = project_root / 'manifests' / 'committed'
        index_path = store / 'index.json'
        index = json.loads(index_path.read_text())
        # manually inject a duplicate entry
        index['manifests'].append(index['manifests'][0])
        index_path.write_text(json.dumps(index))
        assert len(json.loads(index_path.read_text())['manifests']) == 2
        # re-committing the same YAML should deduplicate
        commit_manifest(yaml_path, project_root)
        result = json.loads(index_path.read_text())
        ids = [m['id'] for m in result['manifests']]
        assert ids.count(manifest_id) == 1


def test_is_full_manifest_id_accepts_valid_and_rejects_invalid() -> None:
    valid = _SHA256_PREFIX + 'a' * 64
    assert is_full_manifest_id(valid)
    assert not is_full_manifest_id('a' * 64)
    assert not is_full_manifest_id(_SHA256_PREFIX + 'a' * 8)
    assert not is_full_manifest_id(_SHA256_PREFIX + 'z' * 64)
    assert not is_full_manifest_id(None)
    assert not is_full_manifest_id(123)


def test_normalize_manifest_ref_handles_all_forms() -> None:
    hex_ref = 'd3a5d334'
    assert normalize_manifest_ref(hex_ref) == f'manifest://{_SHA256_PREFIX}{hex_ref}'
    assert normalize_manifest_ref(f'{_SHA256_PREFIX}{hex_ref}') == f'manifest://{_SHA256_PREFIX}{hex_ref}'
    assert normalize_manifest_ref(f'manifest://{_SHA256_PREFIX}{hex_ref}') == f'manifest://{_SHA256_PREFIX}{hex_ref}'
    assert normalize_manifest_ref(f'  {hex_ref}  ') == f'manifest://{_SHA256_PREFIX}{hex_ref}'


def test_load_index_returns_empty_when_absent() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, _ = _make_project(Path(d))
        index = load_index(project_root)
        assert index == {'version': 1, 'manifests': []}


def test_load_index_raises_on_corrupt_index() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, yaml_path = _make_project(Path(d))
        commit_manifest(yaml_path, project_root)
        index_path = project_root / 'manifests' / 'committed' / 'index.json'
        index_path.write_text('{not json', encoding='utf-8')
        with pytest.raises(ValueError, match='corrupted'):
            load_index(project_root)


def test_fork_manifest_creates_dev_copy_with_parent_lineage() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, yaml_path = _make_project(Path(d))
        manifest_id, _ = commit_manifest(yaml_path, project_root)
        committed = project_root / 'manifests' / 'committed' / f'{manifest_id[len(_SHA256_PREFIX):]}.yaml'
        dest = project_root / 'manifests' / 'forked.yaml'

        parent_id = fork_manifest(committed, dest, 'forked')

        assert parent_id == manifest_id
        assert dest.exists()
        from limen.yaml.parser import parse
        data, errors = parse(dest.read_text(encoding='utf-8'))
        assert errors == []
        assert data['metadata']['name'] == 'forked'
        assert data['metadata']['mode'] == 'development'
        assert data['lineage']['parent_id'] == manifest_id


def test_fork_manifest_refuses_to_overwrite() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, yaml_path = _make_project(Path(d))
        manifest_id, _ = commit_manifest(yaml_path, project_root)
        committed = project_root / 'manifests' / 'committed' / f'{manifest_id[len(_SHA256_PREFIX):]}.yaml'
        dest = project_root / 'manifests' / 'forked.yaml'
        dest.write_text('existing', encoding='utf-8')
        with pytest.raises(FileExistsError):
            fork_manifest(committed, dest, 'forked')
        assert dest.read_text(encoding='utf-8') == 'existing'


def test_commit_reads_parent_from_forked_lineage() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, yaml_path = _make_project(Path(d))
        parent_id, _ = commit_manifest(yaml_path, project_root)
        committed = project_root / 'manifests' / 'committed' / f'{parent_id[len(_SHA256_PREFIX):]}.yaml'
        fork_dest = project_root / 'manifests' / 'forked.yaml'
        fork_manifest(committed, fork_dest, 'forked')
        # Change a param so the fork is substantively distinct from its parent.
        text = fork_dest.read_text(encoding='utf-8').replace('lookback: [12, 24]', 'lookback: [36, 48]')
        fork_dest.write_text(text, encoding='utf-8')

        child_id, _ = commit_manifest(fork_dest, project_root)

        assert child_id != parent_id
        index = load_index(project_root)
        child_entry = next(m for m in index['manifests'] if m['id'] == child_id)
        assert child_entry['parent_id'] == parent_id

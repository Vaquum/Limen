import json
import os
import selectors
import subprocess
import sys
import tempfile
import time
from functools import partial
from pathlib import Path

import pytest
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

from limen.yaml import validate
from limen.yaml.config import round_trip_yaml
from limen.yaml.store import SHA256_PREFIX
from limen.yaml.store import canonical_manifest_id
from limen.yaml.store import commit_manifest
from limen.yaml.store import fork_manifest
from limen.yaml.store import is_full_manifest_id
from limen.yaml.store import load_index
from limen.yaml.store import normalize_manifest_ref
from limen.yaml.store import resolve_manifest_uri


_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / 'limen' / 'yaml' / 'templates'

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
        hex_hash = manifest_id[len(SHA256_PREFIX):]
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
        hex_hash = manifest_id[len(SHA256_PREFIX):]
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
        hex_hash = manifest_id[len(SHA256_PREFIX):]
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
        hex_hash = manifest_id[len(SHA256_PREFIX):]
        committed_path = project_root / 'manifests' / 'committed' / f'{hex_hash}.yaml'
        tampered = committed_path.read_text().replace(manifest_id, 'sha256:' + 'b' * 64)
        committed_path.write_text(tampered)
        with pytest.raises(ValueError, match='Integrity check failed'):
            resolve_manifest_uri(f'manifest://{manifest_id}', project_root)


def test_resolve_manifest_uri_accepts_short_hash() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, yaml_path = _make_project(Path(d))
        manifest_id, _ = commit_manifest(yaml_path, project_root)
        short = manifest_id[len(SHA256_PREFIX):len(SHA256_PREFIX) + 8]
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
        hex_hash = manifest_id[len(SHA256_PREFIX):]
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
        assert manifest_id.startswith(SHA256_PREFIX)
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
    valid = SHA256_PREFIX + 'a' * 64
    assert is_full_manifest_id(valid)
    assert not is_full_manifest_id('a' * 64)
    assert not is_full_manifest_id(SHA256_PREFIX + 'a' * 8)
    assert not is_full_manifest_id(SHA256_PREFIX + 'z' * 64)
    assert not is_full_manifest_id(None)
    assert not is_full_manifest_id(123)


def test_normalize_manifest_ref_handles_all_forms() -> None:
    hex_ref = 'd3a5d334'
    assert normalize_manifest_ref(hex_ref) == f'manifest://{SHA256_PREFIX}{hex_ref}'
    assert normalize_manifest_ref(f'{SHA256_PREFIX}{hex_ref}') == f'manifest://{SHA256_PREFIX}{hex_ref}'
    assert normalize_manifest_ref(f'manifest://{SHA256_PREFIX}{hex_ref}') == f'manifest://{SHA256_PREFIX}{hex_ref}'
    assert normalize_manifest_ref(f'  {hex_ref}  ') == f'manifest://{SHA256_PREFIX}{hex_ref}'


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
        committed = project_root / 'manifests' / 'committed' / f'{manifest_id[len(SHA256_PREFIX):]}.yaml'
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
        committed = project_root / 'manifests' / 'committed' / f'{manifest_id[len(SHA256_PREFIX):]}.yaml'
        dest = project_root / 'manifests' / 'forked.yaml'
        dest.write_text('existing', encoding='utf-8')
        with pytest.raises(FileExistsError):
            fork_manifest(committed, dest, 'forked')
        assert dest.read_text(encoding='utf-8') == 'existing'


def test_commit_reads_parent_from_forked_lineage() -> None:
    with tempfile.TemporaryDirectory() as d:
        project_root, yaml_path = _make_project(Path(d))
        parent_id, _ = commit_manifest(yaml_path, project_root)
        committed = project_root / 'manifests' / 'committed' / f'{parent_id[len(SHA256_PREFIX):]}.yaml'
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


@pytest.fixture
def committed_rule_manifest(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    root, source = _make_project(tmp_path)
    text = (_TEMPLATE_DIR / 'rule_based.yaml').read_text().replace('mode: development', 'mode: production')
    assert validate(round_trip_yaml().load(text)).valid
    source.write_text(text)
    manifest_id, _ = commit_manifest(source, root)
    destination = root / 'manifests' / 'committed' / f'{manifest_id[len(SHA256_PREFIX):]}.yaml'
    return root, source, destination, manifest_id


def _assert_integrity_rejected(root: Path,
                               source: Path,
                               destination: Path,
                               index_state: str,
                               cause_type: type[Exception] | None = None) -> None:
    index_path = destination.parent / 'index.json'
    if index_state == 'missing':
        index_path.unlink()
    elif index_state == 'corrupt':
        index_path.write_text('{invalid index')
    destination_before = destination.read_bytes()
    index_before = index_path.read_bytes() if index_path.exists() else None
    operations = [
        partial(resolve_manifest_uri, f'manifest://{SHA256_PREFIX}{destination.stem[:length]}', root)
        for length in (64, 1, 8, 63)
    ]
    operations.append(partial(commit_manifest, source, root))
    for operation in operations:
        with pytest.raises(ValueError, match='Integrity check failed') as failure:
            operation()
        if cause_type is not None:
            assert isinstance(failure.value.__cause__, cause_type)
        assert destination.read_bytes() == destination_before
        assert (index_path.read_bytes() if index_path.exists() else None) == index_before


@pytest.mark.parametrize('index_state', ['existing', 'missing', 'corrupt'])
def test_committed_manifest_rejects_content_tampering(
    committed_rule_manifest: tuple[Path, Path, Path, str], index_state: str,
) -> None:
    root, source, destination, manifest_id = committed_rule_manifest
    destination.write_text(destination.read_text().replace(
        'name: rule_based_experiment', 'name: rule_based_experiment_tampered',
    ))
    document = round_trip_yaml().load(destination.read_text())
    assert document['lineage']['id'] == manifest_id
    assert canonical_manifest_id(document) != manifest_id
    _assert_integrity_rejected(root, source, destination, index_state)


@pytest.mark.parametrize('index_state', ['existing', 'missing', 'corrupt'])
@pytest.mark.parametrize('lineage_state', ['wrong', 'missing', 'scalar', 'empty'])
def test_committed_manifest_rejects_invalid_stored_lineage(
    committed_rule_manifest: tuple[Path, Path, Path, str], index_state: str, lineage_state: str,
) -> None:
    root, source, destination, manifest_id = committed_rule_manifest
    yaml = round_trip_yaml()
    document = yaml.load(destination.read_text())
    if lineage_state == 'wrong':
        document['lineage']['id'] = manifest_id[:-1] + ('0' if manifest_id[-1] != '0' else '1')
    elif lineage_state == 'missing':
        del document['lineage']
    elif lineage_state == 'scalar':
        document['lineage'] = 'damaged'
    else:
        document['lineage'] = {}
    with destination.open('w', encoding='utf-8') as stream:
        yaml.dump(document, stream)
    assert canonical_manifest_id(yaml.load(destination.read_text())) == manifest_id
    _assert_integrity_rejected(root, source, destination, index_state)


@pytest.mark.parametrize('index_state', ['existing', 'missing', 'corrupt'])
@pytest.mark.parametrize('damage,cause_type', [('mixed_keys', TypeError), ('root_cycle', ValueError)])
def test_committed_manifest_rejects_uncanonicalizable_content(
    committed_rule_manifest: tuple[Path, Path, Path, str],
    index_state: str,
    damage: str,
    cause_type: type[Exception],
) -> None:
    root, source, destination, manifest_id = committed_rule_manifest
    text = destination.read_text()
    if damage == 'mixed_keys':
        text += '\nuncanonicalizable:\n  1: one\n  label: two\n'
    else:
        text = '&root\n' + text + '\ncycle: *root\n'
    destination.write_text(text)
    document = round_trip_yaml().load(text)
    assert document['lineage']['id'] == manifest_id
    if damage == 'root_cycle':
        assert document['cycle'] is document
    with pytest.raises(cause_type):
        canonical_manifest_id(document)
    _assert_integrity_rejected(root, source, destination, index_state, cause_type)


@pytest.mark.parametrize('template', sorted(_TEMPLATE_DIR.glob('*.yaml')), ids=lambda path: path.stem)
def test_committed_manifest_preserves_canonical_equivalence(tmp_path: Path, template: Path) -> None:
    root, source = _make_project(tmp_path)
    source.write_text(template.read_text().replace('mode: development', 'mode: production'))
    yaml = round_trip_yaml(preserve_quotes=True)
    expected_id = canonical_manifest_id(yaml.load(source.read_text()))
    manifest_id, already_existed = commit_manifest(source, root)
    assert (manifest_id, already_existed) == (expected_id, False)
    destination = root / 'manifests' / 'committed' / f'{manifest_id[len(SHA256_PREFIX):]}.yaml'
    original = destination.read_bytes()
    document = yaml.load(destination.read_text())
    assert canonical_manifest_id(document) == manifest_id
    document['metadata']['name'] = DoubleQuotedScalarString(document['metadata']['name'])
    document['metadata'] = dict(reversed(tuple(document['metadata'].items())))
    document['lineage']['committed_at'] = '2000-01-01T00:00:00Z'
    document['lineage']['parent_id'] = SHA256_PREFIX + 'a' * 64
    document = dict(reversed(tuple(document.items())))
    with destination.open('w', encoding='utf-8') as stream:
        yaml.dump(document, stream)
    destination.write_text('# Canonically equivalent committed manifest\n' + destination.read_text())
    equivalent = destination.read_bytes()
    assert equivalent != original
    assert canonical_manifest_id(yaml.load(destination.read_text())) == manifest_id
    for length in (64, 1, 8, 63):
        uri = f'manifest://{SHA256_PREFIX}{destination.stem[:length]}'
        assert resolve_manifest_uri(uri, root) == (destination, root)
    assert commit_manifest(source, root) == (manifest_id, True)
    assert destination.read_bytes() == equivalent
    entry = load_index(root)['manifests'][0]
    assert entry['committed_at'] == document['lineage']['committed_at']
    assert entry['parent_id'] == document['lineage']['parent_id']


def _change_fork_source(stored, manifest_id, damage):
    if damage == 'genuine':
        return
    text = stored.read_text()
    if damage in ('mixed-keys', 'root-cycle'):
        stored.write_text(text + '\nuncanonicalizable:\n  1: one\n  label: two\n' if damage == 'mixed-keys'
                          else '&root\n' + text + '\ncycle: *root\n')
        return
    yaml = round_trip_yaml(preserve_quotes=True)
    document = yaml.load(text)
    if damage in ('content', 'coherent-replacement'):
        document['metadata']['description'] = 'Altered after commit'
    elif damage == 'metadata-name':
        document['metadata']['name'] += '_altered'
    elif damage == 'metadata-mode':
        document['metadata']['mode'] = 'development'
    elif damage == 'missing-lineage':
        del document['lineage']
    elif damage == 'scalar-lineage':
        document['lineage'] = 'damaged'
    elif damage == 'empty-lineage':
        document['lineage'] = {}
    elif damage == 'short-lineage':
        document['lineage']['id'] = manifest_id[:-1]
    elif damage == 'lineage':
        document['lineage']['id'] = manifest_id[:-1] + ('0' if manifest_id[-1] != '0' else '1')
    elif damage == 'equivalent':
        document['metadata']['name'] = DoubleQuotedScalarString(document['metadata']['name'])
        document['metadata'] = dict(reversed(tuple(document['metadata'].items())))
        document['lineage']['committed_at'] = '2000-01-01T00:00:00Z'
        document['lineage']['parent_id'] = SHA256_PREFIX + 'a' * 64
        document = dict(reversed(tuple(document.items())))
    else:
        raise AssertionError(damage)
    if damage == 'coherent-replacement':
        document['lineage']['id'] = canonical_manifest_id(document)
        assert document['lineage']['id'] != manifest_id
    with stored.open('w') as stream:
        yaml.dump(document, stream)
    if damage == 'equivalent':
        stored.write_text('# Canonically equivalent source\n' + stored.read_text())
        assert canonical_manifest_id(document) == manifest_id


def _fork_source_snapshot(paths):
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) if path.exists() else None for path in paths}


@pytest.mark.parametrize('index_state', ('existing', 'missing', 'corrupt'))
@pytest.mark.parametrize('damage,cause_type', (
    ('content', None), ('metadata-name', None), ('metadata-mode', None),
    ('lineage', None), ('short-lineage', None), ('missing-lineage', None),
    ('scalar-lineage', None), ('empty-lineage', None), ('coherent-replacement', None),
    ('mixed-keys', TypeError), ('root-cycle', ValueError),
))
def test_fork_refuses_changed_committed_source(committed_rule_manifest, damage, cause_type, index_state):
    root, source, stored, manifest_id = committed_rule_manifest
    _change_fork_source(stored, manifest_id, damage)
    index = stored.parent / 'index.json'
    if index_state == 'missing':
        index.unlink()
    elif index_state == 'corrupt':
        index.write_text('{invalid index')
    before = _fork_source_snapshot((source, stored, index))
    destination = root / 'missing/child.yaml'
    with pytest.raises(ValueError, match='Integrity check failed') as failure:
        fork_manifest(stored, destination, 'child')
    if cause_type is not None:
        assert isinstance(failure.value.__cause__, cause_type)
    assert not destination.exists()
    assert not destination.parent.exists()
    assert _fork_source_snapshot(before) == before


@pytest.mark.parametrize('damage', ('genuine', 'equivalent'))
def test_fork_retains_verified_immediate_parent(committed_rule_manifest, damage):
    root, source, stored, manifest_id = committed_rule_manifest
    _change_fork_source(stored, manifest_id, damage)
    before = _fork_source_snapshot((source, stored, stored.parent / 'index.json'))
    destination = root / 'missing/child.yaml'
    assert fork_manifest(stored, destination, 'child') == manifest_id
    document = round_trip_yaml().load(destination.read_text())
    assert document['lineage'] == {'parent_id': manifest_id}
    assert document['metadata']['name'] == 'child'
    assert document['metadata']['mode'] == 'development'
    assert _fork_source_snapshot(before) == before


@pytest.mark.parametrize('name', ('working.yaml', 'last-digest-character'))
def test_fork_refuses_renamed_source(committed_rule_manifest, name):
    root, _, stored, _ = committed_rule_manifest
    if name == 'last-digest-character':
        name = stored.stem[:-1] + ('0' if stored.stem[-1] != '0' else '1') + '.yaml'
    renamed = stored.with_name(name)
    renamed.write_bytes(stored.read_bytes())
    before = _fork_source_snapshot((stored, renamed, stored.parent / 'index.json'))
    with pytest.raises(ValueError, match='Integrity check failed'):
        fork_manifest(renamed, root / 'missing/child.yaml', 'child')
    assert not (root / 'missing').exists()
    assert _fork_source_snapshot(before) == before


def test_fork_overwrite_precedes_integrity_refusal(committed_rule_manifest):
    root, _, stored, manifest_id = committed_rule_manifest
    _change_fork_source(stored, manifest_id, 'content')
    destination = root / 'existing.yaml'
    destination.write_text('existing working copy')
    before = _fork_source_snapshot((destination, stored, stored.parent / 'index.json'))
    with pytest.raises(FileExistsError):
        fork_manifest(stored, destination, 'child')
    assert _fork_source_snapshot(before) == before


def test_verified_fork_recommit_journey(committed_rule_manifest):
    root, _, stored, parent_id = committed_rule_manifest
    yaml = round_trip_yaml()
    for generation in range(1, 4):
        destination = root / 'manifests' / f'child_{generation}.yaml'
        assert fork_manifest(stored, destination, f'child_{generation}') == parent_id
        document = yaml.load(destination.read_text())
        assert document['lineage'] == {'parent_id': parent_id}
        document['metadata']['mode'] = 'production'
        with destination.open('w') as stream:
            yaml.dump(document, stream)
        assert validate(document).valid
        child_id, existed = commit_manifest(destination, root)
        assert not existed and child_id != parent_id
        stored = root / 'manifests/committed' / (child_id[len(SHA256_PREFIX):] + '.yaml')
        committed = yaml.load(stored.read_text())
        assert canonical_manifest_id(committed) == child_id == committed['lineage']['id']
        assert committed['lineage']['parent_id'] == parent_id
        entry = next(item for item in load_index(root)['manifests'] if item['id'] == child_id)
        assert entry['parent_id'] == parent_id
        parent_id = child_id


@pytest.mark.parametrize('damage,ref_kind', (
    ('content', 'full'), ('lineage', 'uri'), ('missing-lineage', 'short'),
    ('coherent-replacement', 'full'), ('mixed-keys', 'uri'), ('root-cycle', 'short'),
    ('genuine', 'full'), ('equivalent', 'uri'),
))
def test_cli_fork_verifies_the_read_after_its_real_name_prompt(committed_rule_manifest, damage, ref_kind):
    root, source, stored, manifest_id = committed_rule_manifest
    ref = {'full': manifest_id, 'uri': 'manifest://' + manifest_id,
           'short': manifest_id[:len(SHA256_PREFIX) + 8]}[ref_kind]
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    with subprocess.Popen(
        [sys.executable, '-m', 'limen', 'fork', ref], cwd=root, env=environment,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ) as process:
        try:
            assert process.stdout is not None
            output = b''
            deadline = time.monotonic() + 30
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while b'Name for the fork' not in output:
                    remaining = deadline - time.monotonic()
                    assert remaining > 0 and selector.select(remaining), output.decode()
                    chunk = os.read(process.stdout.fileno(), 4096)
                    assert chunk, output.decode()
                    output += chunk
            # Resolution and the name read have completed; the actual fork read follows input.
            _change_fork_source(stored, manifest_id, damage)
            before = _fork_source_snapshot((source, stored, stored.parent / 'index.json'))
            tail, _ = process.communicate(input=b'child\n', timeout=30)
            output = (output + tail).decode()
            destination = root / 'manifests/child.yaml'
            if damage in ('genuine', 'equivalent'):
                assert process.returncode == 0, output
                document = round_trip_yaml().load(destination.read_text())
                assert document['lineage'] == {'parent_id': manifest_id}
                assert document['metadata']['name'] == 'child'
                assert document['metadata']['mode'] == 'development'
            else:
                assert process.returncode == 1, output
                assert 'Integrity check failed' in output
                assert 'Traceback' not in output
                assert not destination.exists()
            assert _fork_source_snapshot(before) == before
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

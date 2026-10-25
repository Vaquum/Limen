import copy
import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from threadpoolctl import threadpool_limits

from limen.cli.main import cli
from limen.experiment.manifest_core import DataSourceResolver
from limen.yaml import CompiledSFD, parse, validate
from limen.yaml.config import round_trip_yaml
from limen.yaml.schema import METADATA_OPTIONAL
from limen.yaml.store import canonical_manifest_id, commit_manifest, resolve_manifest_uri

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/spot_15m_20250101_20250531.parquet'


def _config(declaration=''):
    config, errors = parse(ROOT / 'limen/yaml/templates/logreg_binary.yaml')
    assert not errors
    for name in METADATA_OPTIONAL:
        config['metadata'].pop(name, None)
    if declaration:
        declared, errors = parse(declaration)
        assert not errors
        config['metadata'].update(declared)
    return config


@pytest.mark.parametrize('declaration,path', [
    ('tags: {a: {b: 1}}', 'metadata.tags'),
    ('created_at: 2025-10-25', 'metadata.created_at'),
    ('created_at: 2025-10-25T01:02:03Z', 'metadata.created_at'),
    ('limen_version: 5.4', 'metadata.limen_version'),
    ('author: 42', 'metadata.author'), ('author: [a, b]', 'metadata.author'),
    ('author: ""', 'metadata.author'), ('author: "  "', 'metadata.author'),
    ('author: 2025-10-25', 'metadata.author'),
    ('description: {x: 1}', 'metadata.description'), ('description: 3', 'metadata.description'),
    ('description: [a]', 'metadata.description'),
    ('tags: momentum', 'metadata.tags'), ('tags: [1]', 'metadata.tags[0]'),
    ('tags: [true]', 'metadata.tags[0]'), ('tags: [2025-01-01]', 'metadata.tags[0]'),
    ('tags: [[a]]', 'metadata.tags[0]'), ('tags: [{a: 1}]', 'metadata.tags[0]'),
    ('tags: [a, " "]', 'metadata.tags[1]'),
    ('created_at: 20251025', 'metadata.created_at'), ('created_at: true', 'metadata.created_at'),
    ('created_at: {y: 2025}', 'metadata.created_at'), ('created_at: yesterday', 'metadata.created_at'),
    ('created_at: "2025-13-45"', 'metadata.created_at'),
    ('created_at: "2025-02-29"', 'metadata.created_at'),
    ('created_at: "20251025"', 'metadata.created_at'),
    ('created_at: "2025-W43-6"', 'metadata.created_at'),
    ('created_at: "2025-10-25T24:00:00Z"', 'metadata.created_at'),
    ('created_at: "2025-10-25T01:02:03+02:60"', 'metadata.created_at'),
    ('created_at: "2025-10-25T01:02:03.1234567Z"', 'metadata.created_at'),
    ('created_at: "2025-10-25 01:02:03"', 'metadata.created_at'),
    ('limen_version: 5', 'metadata.limen_version'), ('limen_version: [5, 46]', 'metadata.limen_version'),
    ('limen_version: {v: 1}', 'metadata.limen_version'), ('limen_version: banana', 'metadata.limen_version'),
    ('limen_version: "5.46.2garbage"', 'metadata.limen_version'),
    ('limen_version: " 5.46.2"', 'metadata.limen_version'),
])
def test_optional_metadata_shape_is_checked_after_real_yaml_parsing(declaration, path):
    config = _config(declaration)
    before = repr(config)
    result = validate(config)
    assert not result.valid and len(result.errors) == 1
    assert result.errors[0].path == path
    assert repr(config) == before
    if path in ('metadata.created_at', 'metadata.limen_version'):
        assert 'Quote' in result.errors[0].suggestion


@pytest.mark.parametrize('declaration', [
    'author: Vaquum', 'description: ""', 'description: " "',
    'tags: []', 'tags: [momentum, momentum]', 'tags: ["roc_{C}"]',
    'created_at: "2024-02-29"', 'created_at: "2025-10-25T01:02:03"',
    'created_at: "2025-10-25T01:02:03Z"', 'created_at: "2025-10-25T01:02:03+02:30"',
    *[f'created_at: "2025-10-25T01:02:03.{"123456"[:precision]}Z"' for precision in range(1, 7)],
    *[f'limen_version: "{version}"' for version in ('3.8.0', '99.0.0', '0+unknown', '5.4', '1!2.0',
        '5.46.2rc1', '5.46.2.post1', '5.46.2.dev1', '5.46.2+local.1', 'v5.46.2', '5.46.2-alpha1')],
    *[f'{name}: null' for name in sorted(METADATA_OPTIONAL)],
    '',
])
def test_valid_optional_shapes_preserve_values_and_do_not_infer_compatibility(declaration):
    config = _config(declaration)
    before = copy.deepcopy(config)
    result = validate(config)
    assert result.valid and not result.errors and not result.warnings
    assert config == before and json.loads(json.dumps(config)) == config


def _write(path, config):
    with path.open('w') as stream:
        round_trip_yaml().dump(config, stream)


def _snapshot(directory):
    return {str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob('*') if path.is_file()}


def _forbidden(*args, **kwargs):
    raise AssertionError('Malformed metadata reached compilation or source loading')


@pytest.mark.parametrize('declaration', ['created_at: 2025-10-25', 'tags: {a: 1}', 'author: false', 'limen_version: 5.4'])
@pytest.mark.parametrize('command', [('validate',), ('profile',), ('commit',), ('reserve',), ('assess',),
                                    ('run', '--dry-run'), ('run', '--no-progress-bar')])
def test_cli_shape_admission_precedes_partial_artifacts(tmp_path, monkeypatch, declaration, command):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'limen.toml').write_text('')
    config = _config(declaration)
    path = tmp_path / 'metadata.yaml'
    _write(path, config)
    before = _snapshot(tmp_path)
    monkeypatch.setattr(CompiledSFD, '__init__', _forbidden)
    monkeypatch.setattr(DataSourceResolver, 'resolve', staticmethod(_forbidden))
    args = [*command, str(path), *(['--results', str(tmp_path)] if command[0] == 'assess' else [])]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1 and 'ERROR  [metadata.' in result.output
    assert _snapshot(tmp_path) == before
    assert not (tmp_path / 'results').exists() and not (tmp_path / 'manifests').exists()


@pytest.mark.parametrize('created', ['2025-10-25', '2025-10-25T01:02:03Z', '2025-10-25T01:02:03.5+02:00'])
def test_fully_populated_metadata_survives_canonical_commit_and_verified_readback(tmp_path, created):
    (tmp_path / 'limen.toml').write_text('')
    config = _config(f'author: Vaquum\ndescription: "Recorded metadata admission"\ntags: [momentum, momentum]\n'
                     f'created_at: "{created}"\nlimen_version: "0+unknown"')
    assert validate(config).valid
    identity = canonical_manifest_id(config)
    path = tmp_path / 'source.yaml'
    _write(path, config)
    assert commit_manifest(path, tmp_path) == (identity, False)
    stored, root = resolve_manifest_uri('manifest://' + identity, tmp_path)
    restored, errors = parse(stored)
    assert not errors and root == tmp_path
    assert validate(restored).valid and canonical_manifest_id(restored) == identity
    assert restored['metadata'] == config['metadata']


def test_legacy_stored_malformed_metadata_is_refused_without_rewriting_the_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'limen.toml').write_text('')
    config = _config('tags: {a: 1}')
    path = tmp_path / 'legacy.yaml'
    _write(path, config)
    identity, _ = commit_manifest(path, tmp_path)
    before = _snapshot(tmp_path)
    monkeypatch.setattr(CompiledSFD, '__init__', _forbidden)
    result = CliRunner().invoke(cli, ['run', '--no-progress-bar', 'manifest://' + identity])
    assert result.exit_code == 1 and 'ERROR  [metadata.tags]' in result.output
    assert _snapshot(tmp_path) == before


def test_recorded_run_and_assessment_keep_the_exact_quoted_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config, errors = parse(ROOT / 'docs/examples/compiled-research/calibration_assessment.yaml')
    assert not errors
    config['sfd']['manifest']['data_source']['params']['file_path_or_url'] = str(FIXTURE)
    config['metadata'].update(_config('author: Vaquum\ndescription: "Recorded metadata admission"\n'
        'tags: [calibration, calibration]\ncreated_at: "2025-10-25T01:02:03.5+02:00"\n'
        'limen_version: "0+unknown"')['metadata'])
    config['metadata']['name'] = 'metadata_assessment'
    config['assessment']['predict'] = False
    path = tmp_path / 'assessment.yaml'
    _write(path, config)
    with threadpool_limits(limits=1):
        result = CliRunner().invoke(cli, ['run', '--no-progress-bar', str(path)])
        assert result.exit_code == 0, result.output
        directory = tmp_path / 'results/dev/compiled-calibration-assessment'
        saved = json.loads((directory / 'metadata.json').read_text())['yaml_reference']
        parsed, errors = parse(path)
        assert not errors and saved == parsed == config
        result = CliRunner().invoke(cli, ['assess', str(path), '--results', str(directory)])
        assert result.exit_code == 0, result.output
    report = json.loads((directory / 'assessment.json').read_text())
    assert report['members'] == 4 and len(report['selected_ids']) == 4

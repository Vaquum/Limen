import json
from pathlib import Path

import polars as pl
import pytest
from click.testing import CliRunner
from threadpoolctl import threadpool_limits

from limen.cli.main import cli
from limen.experiment import MLManifest, UniversalExperimentLoop, _round_storage
from limen.experiment.manifest_core import DataSourceResolver
from limen.inference import Trainer
from limen.yaml import CompiledSFD, build_manifest, parse, validate
from limen.yaml.config import round_trip_yaml
from limen.yaml.rules import SchemaVersion
from limen.yaml.store import commit_manifest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet'


def _config():
    config, errors = parse(ROOT / 'limen/yaml/templates/logreg_binary.yaml')
    assert not errors
    return config


@pytest.mark.parametrize('version', ['999.0', '0.9', '1.1', '1', '1.00', '1.0.0', ' 1.0', ''])
def test_unsupported_schema_is_refused_before_interpreting_its_body(version):
    config = _config()
    config['schema_version'] = version
    config['metadata']['mode'] = {'future': 'shape'}
    config['uel']['output_format'] = []
    result = validate(config)
    assert not result.valid and not result.warnings
    assert len(result.errors) == 1
    assert result.errors[0].path == 'schema_version'
    assert f"Unsupported schema_version '{version}'" in result.errors[0].message
    assert "supports '1.0'" in result.errors[0].message


@pytest.mark.parametrize('version', [1.0, 1, True, None, [], {}])
def test_schema_type_and_missing_value_rejections_are_preserved(version):
    config = _config()
    config['schema_version'] = version
    result = validate(config)
    assert not result.valid and len(result.errors) == 1 and not result.warnings
    assert result.errors[0].path == 'schema_version'
    assert ('Missing required field' if version is None else 'must be a str') in result.errors[0].message
    config.pop('schema_version')
    result = validate(config)
    assert len(result.errors) == 1 and result.errors[0].message == "Missing required field 'schema_version'"


@pytest.mark.parametrize('template', sorted((ROOT / 'limen/yaml/templates').glob('*.yaml')), ids=lambda path: path.stem)
def test_supported_templates_do_not_depend_on_package_release_metadata(template):
    config, errors = parse(template)
    assert not errors
    result = validate(config)
    assert result.valid and not result.errors and not result.warnings
    config['metadata']['limen_version'] = '3.8.0'
    result = validate(config)
    assert result.valid and not result.errors and not result.warnings


def _snapshot(directory):
    return {str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob('*') if path.is_file()}


def _write(path, config):
    with path.open('w') as stream:
        round_trip_yaml().dump(config, stream)


def _forbidden(*args, **kwargs):
    raise AssertionError('Unsupported schema reached reconstruction or source loading')


@pytest.mark.parametrize('version', ['999.0', '', 1.0, None])
@pytest.mark.parametrize('command', [('validate',), ('profile',), ('run', '--dry-run'),
                                    ('run', '--no-progress-bar'), ('commit',), ('reserve',), ('assess',)])
def test_cli_version_admission_precedes_body_compilation_and_writes(tmp_path, monkeypatch, version, command):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'limen.toml').write_text('')
    config = _config()
    config['schema_version'] = version
    config['metadata']['mode'] = {'future': 'shape'}
    path = tmp_path / 'unsupported.yaml'
    _write(path, config)
    before = _snapshot(tmp_path)
    monkeypatch.setattr(CompiledSFD, '__init__', _forbidden)
    monkeypatch.setattr(DataSourceResolver, 'resolve', staticmethod(_forbidden))
    args = [*command, str(path), *(['--results', str(tmp_path)] if command[0] == 'assess' else [])]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1, result.output
    assert 'ERROR  [schema_version]' in result.output
    assert _snapshot(tmp_path) == before
    assert not (tmp_path / 'results').exists() and not (tmp_path / 'manifests').exists()


def test_stored_unsupported_manifest_cannot_start_a_fresh_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'limen.toml').write_text('')
    config = _config()
    config['schema_version'] = '999.0'
    config['metadata']['mode'] = 'production'
    path = tmp_path / 'legacy.yaml'
    _write(path, config)
    identity, _ = commit_manifest(path, tmp_path)
    before = _snapshot(tmp_path)
    monkeypatch.setattr(CompiledSFD, '__init__', _forbidden)
    result = CliRunner().invoke(cli, ['run', '--no-progress-bar', 'manifest://' + identity])
    assert result.exit_code == 1 and 'ERROR  [schema_version]' in result.output
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize('version', ['1.0', '999.0'])
def test_real_saved_checkpoint_version_admission_preserves_artifacts_and_readback(tmp_path, monkeypatch, version):
    monkeypatch.chdir(tmp_path)
    config, errors = parse(ROOT / 'docs/examples/compiled-research/recovery.yaml')
    assert not errors
    config['schema_version'] = version
    config['metadata']['limen_version'] = '3.8.0'
    config['sfd']['manifest']['data_source']['params']['file_path_or_url'] = str(FIXTURE)
    path = tmp_path / 'recovery.yaml'
    _write(path, config)
    append = UniversalExperimentLoop._append_round_data
    check = SchemaVersion.check

    def stop_after_record(loop, *args, **kwargs):
        append(loop, *args, **kwargs)
        loop._shutdown_requested = True

    def prior_version_rule(rule, document, errors, warnings):
        diagnosed = []
        check(rule, document, diagnosed, warnings)
        warnings.extend(diagnosed)  # Previous version mismatches were warnings.

    with threadpool_limits(limits=1):
        with monkeypatch.context() as previous:
            previous.setattr(UniversalExperimentLoop, '_append_round_data', stop_after_record)
            if version != '1.0':
                previous.setattr(SchemaVersion, 'check', prior_version_rule)
            result = CliRunner().invoke(cli, ['run', '--no-progress-bar', str(path)])
        assert result.exit_code == 1 and 'Interrupted; checkpoint retained' in result.output
        directory = tmp_path / 'results/dev/compiled-recovery'
        records = [json.loads(line) for line in _round_storage.text(_round_storage.round_path(directory)).splitlines()]
        assert len(records) == 1
        saved = json.loads((directory / 'metadata.json').read_text())['yaml_reference']
        assert saved['schema_version'] == version
        before = _snapshot(directory)
        if version != '1.0':
            with monkeypatch.context() as guard:
                guard.setattr(CompiledSFD, '__init__', _forbidden)
                guard.setattr(DataSourceResolver, 'resolve', staticmethod(_forbidden))
                result = CliRunner().invoke(cli, ['run', '--no-progress-bar', '--resume', str(directory)])
            assert result.exit_code == 1 and "schema_version: Unsupported schema_version '999.0'" in result.output
            assert _snapshot(directory) == before
        else:
            result = CliRunner().invoke(cli, ['run', '--no-progress-bar', '--resume', str(directory)])
            assert result.exit_code == 0, result.output
            assert len(pl.read_csv(directory / 'results.csv')) == 4
        # Scientific reconstruction retains its prevalidated-input boundary for old evidence.
        sensor = Trainer(directory, data=pl.read_parquet(FIXTURE)).train([records[0]['round_id']])[0]
        assert sensor is not None


def test_direct_compilation_keeps_its_prevalidated_input_contract():
    config = _config()
    config['schema_version'] = '999.0'
    assert isinstance(build_manifest(config), MLManifest)
    assert isinstance(CompiledSFD(config).manifest(), MLManifest)

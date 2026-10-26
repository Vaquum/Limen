from pathlib import Path
import json
import subprocess

import polars as pl
import pytest
from click.testing import CliRunner
from ruamel.yaml import YAML

from limen.cli.main import cli
from limen.experiment import UniversalExperimentLoop
from limen.yaml import CompiledSFD, build_search_strategy
from limen.yaml import parse, validate
from limen.yaml.schema import UEL_OPTIONAL, UEL_YAML_REJECTIONS

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('template', ('logreg_binary', 'rule_based'))
@pytest.mark.parametrize('value', (
    'this.path.does.not.resolve', 'limen.indicators.roc',
    {'func': 'this.path.does.not.resolve'}, {'func': 'limen.indicators.roc'},
    None, False, True, '', {}, [], ['callback'], 42, lambda log, msq: None,
))
def test_yaml_callback_presence_has_one_actionable_owner(template, value):
    config, errors = parse(ROOT / 'limen/yaml/templates' / f'{template}.yaml')
    assert not errors
    assert validate(config).valid
    config['uel']['intra_callback'] = value
    result = validate(config)
    assert not result.valid
    errors = [error for error in result.errors if error.path == 'uel.intra_callback']
    assert len(errors) == 1
    assert 'UniversalExperimentLoop(intra_callback=<callable(log, msq)>)' in errors[0].message
    assert '#manual-intervention' in errors[0].message
    assert not any(warning.path == 'uel.intra_callback' for warning in result.warnings)
    assert 'intra_callback' not in UEL_OPTIONAL


def _project(root, kind='ml'):
    root.mkdir()
    (root / 'limen.toml').write_text('[project]\nname="callback-admission"\n')
    subprocess.run(['git', 'init', '--quiet', str(root)], check=True)
    if kind == 'ml':
        reference, errors = parse(ROOT / 'docs/examples/compiled-research/recovery.yaml')
    else:
        reference, errors = parse(ROOT / 'limen/yaml/templates/rule_based.yaml')
    assert not errors and validate(reference).valid
    reference['sfd']['manifest']['data_source'] = {
        'method': 'limen.data.HistoricalData.get_any_file',
        'params': {'file_path_or_url': str(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet')},
    }
    reference['uel']['output_path'] = 'callback-output'
    return reference


def _snapshot(root):
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns) if path.is_file() else None
            for path in sorted(root.rglob('*'))}


@pytest.mark.parametrize('kind', ('ml', 'rule_based'))
@pytest.mark.parametrize('value', ('this.path.does.not.resolve', None))
@pytest.mark.parametrize('command', ('validate', 'run', 'dry-run', 'profile', 'commit'))
def test_cli_callback_refuses_before_construction_and_project_writes(tmp_path, monkeypatch, kind, value, command):
    root = tmp_path / 'project'
    reference = _project(root, kind)
    reference['metadata']['mode'] = 'production'
    reference['uel']['intra_callback'] = value
    submitted = root / 'experiment.yaml'
    YAML().dump(reference, submitted)
    monkeypatch.chdir(root)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError('Rejected callback reached construction or source loading')

    for module in ('run', 'profile'):
        monkeypatch.setattr(f'limen.cli.commands.{module}.CompiledSFD', forbidden)
    monkeypatch.setattr('limen.experiment.manifest_core.DataSourceResolver.resolve', forbidden)
    before = _snapshot(root)
    args = ['run', '--dry-run', str(submitted)] if command == 'dry-run' else [command, str(submitted)]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1, result.output
    assert 'uel.intra_callback' in result.output
    assert result.output.count('YAML does not execute intra_callback') == 1
    assert 'UniversalExperimentLoop(intra_callback=' in result.output
    assert 'Unknown field' not in result.output
    assert calls == []
    assert _snapshot(root) == before
    assert not (root / 'results').exists()
    assert not (root / 'manifests').exists()


@pytest.mark.parametrize('value', ('limen.indicators.roc', None))
def test_genuine_stored_callback_uri_refuses_without_mutation(tmp_path, monkeypatch, value):
    from limen.yaml.store import commit_manifest

    root = tmp_path / 'project'
    reference = _project(root)
    reference['metadata']['mode'] = 'production'
    reference['uel']['intra_callback'] = value
    source = root / 'experiment.yaml'
    YAML().dump(reference, source)
    # The store's prevalidated package API retains a genuine legacy declaration.
    identity, _ = commit_manifest(source, root)
    monkeypatch.chdir(root)
    before = _snapshot(root)
    for dry in (False, True):
        args = ['run', 'manifest://' + identity]
        if dry:
            args.insert(1, '--dry-run')
        result = CliRunner().invoke(cli, args)
        assert result.exit_code == 1, result.output
        assert 'uel.intra_callback' in result.output
        assert _snapshot(root) == before
    assert not (root / 'results').exists()


@pytest.mark.parametrize('value', ('this.path.does.not.resolve', None))
@pytest.mark.parametrize('schema_version', ('1.0', '999.0'))
def test_legacy_cli_resume_refuses_before_checkpoint_or_scientific_writes(tmp_path, monkeypatch, value, schema_version):
    root = tmp_path / 'project'
    reference = _project(root)
    reference['schema_version'] = schema_version
    reference['uel']['intra_callback'] = value
    reference['uel']['feedback_interval'] = 1
    directory = root / 'saved-native-run'
    bars = pl.read_parquet(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet')

    def stop(log, _msq):
        if log.height == 2:
            loop._shutdown_requested = True

    loop = UniversalExperimentLoop(
        sfd=CompiledSFD(reference), data=bars, experiment_dir=directory,
        search_strategy=build_search_strategy(reference), yaml_reference=reference,
        feedback_interval=1, checkpoint_interval=1, intra_callback=stop,
    )
    loop.run('legacy-callback', n_permutations=4, prep_each_round=True, progress_bar=False,
             recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True))
    assert loop.interrupted
    assert pl.read_csv(directory / 'results.csv').height == 2
    saved = json.loads((directory / 'metadata.json').read_text())
    assert 'intra_callback' in saved['yaml_reference']['uel']
    before = _snapshot(root)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError('Rejected saved callback reached checkpoint loading or reconstruction')

    monkeypatch.setattr('limen.cli.commands.resume.CompiledSFD', forbidden)
    monkeypatch.setattr('limen.experiment.checkpoint_manager.CheckpointManager.load', forbidden)
    result = CliRunner().invoke(cli, ['run', '--resume', str(directory), '--no-progress-bar'])
    assert result.exit_code == 1, result.output
    if schema_version == '1.0':
        assert 'uel.intra_callback' in result.output
        assert 'UniversalExperimentLoop(intra_callback=' in result.output
    else:
        assert "schema_version: Unsupported schema_version '999.0'" in result.output
        assert 'intra_callback' not in result.output
    assert calls == []
    assert _snapshot(root) == before


@pytest.mark.parametrize('severity', ('warning', 'error'))
def test_callback_owner_preserves_unrelated_unknown_diagnostic(severity):
    from limen.yaml.rules import NoUnknownKeys
    from limen.yaml.schema import UEL_REQUIRED

    errors, warnings = [], []
    NoUnknownKeys('uel', UEL_REQUIRED | UEL_OPTIONAL, severity, rejected=UEL_YAML_REJECTIONS).check(
        {'uel': {'intra_callback': None, 'other_typo': True}}, errors, warnings,
    )
    assert len([error for error in errors if error.path == 'uel.intra_callback']) == 1
    unknown = errors if severity == 'error' else warnings
    assert len([error for error in unknown if error.path == 'uel.other_typo']) == 1
    assert not any(warning.path == 'uel.intra_callback' for warning in warnings)


def test_yaml_merge_cannot_hide_callback_presence(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    reference = _project(root)
    source = root / 'experiment.yaml'
    YAML().dump(reference, source)
    source.write_text(source.read_text().replace('uel:\n', 'uel:\n  <<: {intra_callback: null}\n'))
    parsed, errors = parse(source)
    assert not errors and 'intra_callback' in parsed['uel']
    result = validate(parsed)
    assert [error.path for error in result.errors] == ['uel.intra_callback']
    monkeypatch.chdir(root)
    before = _snapshot(root)
    invoked = CliRunner().invoke(cli, ['run', '--dry-run', str(source)])
    assert invoked.exit_code == 1 and 'uel.intra_callback' in invoked.output
    assert _snapshot(root) == before


def test_callback_omission_still_delivers_recorded_cli_results(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    reference = _project(root)
    reference['uel'].update(n_permutations=2, feedback_interval=1)
    source = root / 'experiment.yaml'
    YAML().dump(reference, source)
    monkeypatch.chdir(root)
    result = CliRunner().invoke(cli, ['run', '--no-progress-bar', str(source)])
    assert result.exit_code == 0, result.output
    output = root / 'results/dev/callback-output'
    frame = pl.read_csv(output / 'results.csv')
    assert frame.height == 2
    assert frame['accuracy'].is_finite().all() and frame['auc'].is_finite().all()
    assert json.loads((output / 'metadata.json').read_text())['yaml_reference'] == reference
    assert (output / source.name).read_bytes() == source.read_bytes()
    audit = [json.loads(line) for line in (output / 'audit.jsonl').read_text().splitlines()]
    assert audit and all(event['errors'] == [] and event['interventions'] == [] for event in audit)

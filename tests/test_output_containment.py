import json
import shutil
import subprocess
from pathlib import Path

import polars as pl
import pytest
from click.testing import CliRunner
from ruamel.yaml import YAML

from limen.cli.main import cli
from limen.cli.commands.run import run_experiment
from limen.yaml import parse, validate

ROOT = Path(__file__).resolve().parents[1]


def _project(path):
    path.mkdir()
    (path / 'limen.toml').write_text('[project]\nname = "output-containment"\n')
    subprocess.run(['git', 'init', '--quiet', str(path)], check=True)
    (path / 'inputs').mkdir()
    name = 'spot_1h_20240101_20241231.parquet'
    shutil.copyfile(ROOT / 'tests/fixtures' / name, path / 'inputs' / name)
    config, errors = parse(ROOT / 'docs/examples/compiled-research/recovery.yaml')
    assert not errors
    assert validate(config).valid
    config['uel']['n_permutations'] = 1
    return config


def _write(path, config):
    YAML().dump(config, path)
    assert validate(parse(path)[0]).valid


def _snapshot(root):
    return {
        str(path.relative_to(root)): (
            ('link', str(path.readlink())) if path.is_symlink() else
            (path.read_bytes(), path.stat().st_mtime_ns) if path.is_file() else ('directory',)
        )
        for path in sorted(root.rglob('*'))
    }


def _assert_artifacts(output, submitted, config, *, copied_name=None):
    assert (output / (copied_name or submitted.name)).read_bytes() == submitted.read_bytes()
    metadata = json.loads((output / 'metadata.json').read_text())
    assert metadata['yaml_reference'] == config
    results = pl.read_csv(output / 'results.csv')
    assert results.height == 1
    assert results['accuracy'].is_finite().all()
    assert results['auc'].is_finite().all()
    assert (output / 'round_data.jsonl.gz').is_file()
    return metadata


@pytest.mark.parametrize('dry_run', (False, True))
@pytest.mark.parametrize('mode', ('development', 'production'))
@pytest.mark.parametrize('kind', (
    'absolute-outside', 'absolute-existing', 'absolute-inside', 'parent', 'production', 'sibling',
    'symlink', 'symlink-parent', 'empty', 'dot', 'root-alias',
))
def test_direct_output_refuses_before_project_or_external_writes(tmp_path, monkeypatch, kind, dry_run, mode):
    project = tmp_path / 'project'
    config = _project(project)
    config['metadata']['mode'] = mode
    foreign = tmp_path / 'foreign'
    _project(foreign)
    (foreign / 'sentinel').write_text('existing external content')
    prefix = project / ('results/dev' if mode == 'development' else 'results')
    prefix.mkdir(parents=True)
    (prefix / 'external').symlink_to(foreign, target_is_directory=True)
    (prefix / 'root').symlink_to(prefix, target_is_directory=True)
    config['uel']['output_path'] = {
        'absolute-outside': str(foreign / 'output'),
        'absolute-existing': str(foreign),
        'absolute-inside': str(prefix / 'internal'),
        'parent': '../../../escaped',
        'production': '../prod_{name}',
        'sibling': '../dev-evil/out',
        'symlink': 'external/output',
        'symlink-parent': 'external/../escaped',
        'empty': '', 'dot': '.', 'root-alias': 'root',
    }[kind]
    submitted = project / 'experiment.yaml'
    _write(submitted, config)
    monkeypatch.chdir(project)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError('Invalid output reached construction, reservation or source access')

    monkeypatch.setattr('limen.cli.commands.run.CompiledSFD', forbidden)
    monkeypatch.setattr('limen.cli.commands.run.UniversalExperimentLoop', forbidden)
    monkeypatch.setattr('limen.cli.assessment.require_reservation', forbidden)
    monkeypatch.setattr('limen.experiment.manifest_core.DataSourceResolver.resolve', forbidden)
    before = _snapshot(tmp_path)
    args = ['run', '--no-progress-bar', str(submitted)]
    if dry_run:
        args.insert(1, '--dry-run')
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit)
    assert 'uel.output_path' in result.output
    assert 'Traceback' not in result.output
    assert calls == []
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize('mode', ('development', 'production'))
@pytest.mark.parametrize('template', (None, 'nested/{name}/{datetime}'))
def test_relative_output_delivers_recorded_results(tmp_path, monkeypatch, mode, template):
    project = tmp_path / 'project'
    config = _project(project)
    config['metadata']['mode'] = mode
    if template is None:
        config['uel'].pop('output_path')
    else:
        config['uel']['output_path'] = template
    submitted = project / 'experiment.yaml'
    _write(submitted, config)
    monkeypatch.chdir(project)
    result = CliRunner().invoke(cli, ['run', '--no-progress-bar', str(submitted)])
    assert result.exit_code == 0, result.output
    prefix = project / ('results/dev' if mode == 'development' else 'results')
    outputs = list(prefix.rglob('metadata.json'))
    assert len(outputs) == 1
    output = outputs[0].parent
    assert output.is_relative_to(prefix)
    _assert_artifacts(output, submitted, config)


@pytest.mark.parametrize('mode', ('development', 'production'))
@pytest.mark.parametrize('route', (
    'symlink-root', 'symlink-mode', 'internal-alias', 'explicit-base',
    'symlink-base', 'contained-parent', 'nested-yaml',
))
def test_authorized_output_routes_deliver_recorded_results(tmp_path, monkeypatch, mode, route):
    project = tmp_path / 'project'
    config = _project(project)
    config['metadata']['mode'] = mode
    config['uel']['output_path'] = 'destination'
    base = project
    mode_path = Path('results/dev' if mode == 'development' else 'results')
    output = project / mode_path / 'destination'
    if route in ('explicit-base', 'symlink-base'):
        base = tmp_path / 'authorized-base'
        base.mkdir()
        output = base / mode_path / 'destination'
        if route == 'symlink-base':
            alias = tmp_path / 'base-alias'
            alias.symlink_to(base, target_is_directory=True)
            base = alias
    elif route in ('symlink-root', 'symlink-mode'):
        external = tmp_path / 'authorized-results'
        external.mkdir()
        link = project / ('results' if route == 'symlink-root' else mode_path)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(external, target_is_directory=True)
        output = external / ('dev/destination' if route == 'symlink-root' and mode == 'development' else 'destination')
    elif route == 'internal-alias':
        prefix = project / mode_path
        (prefix / 'actual').mkdir(parents=True)
        (prefix / 'alias').symlink_to(prefix / 'actual', target_is_directory=True)
        config['uel']['output_path'] = 'alias/destination'
        output = prefix / 'actual/destination'
    elif route == 'contained-parent':
        config['uel']['output_path'] = 'unused/../destination'
    submitted = project / 'experiment.yaml'
    if route == 'nested-yaml':
        submitted = project / 'nested/experiment.yaml'
        submitted.parent.mkdir()
    _write(submitted, config)
    monkeypatch.chdir(project)
    if route in ('explicit-base', 'symlink-base'):
        assert run_experiment(submitted, results_base=base, progress_bar=False)
    else:
        result = CliRunner().invoke(cli, ['run', '--no-progress-bar', str(submitted)])
        assert result.exit_code == 0, result.output
        assert f'Results → {output.resolve()}' in result.output
    assert output.is_dir()
    _assert_artifacts(output, submitted, config)
    if route == 'nested-yaml':
        assert not (submitted.parent / mode_path).exists()


@pytest.mark.parametrize('mode', ('development', 'production'))
def test_committed_uri_ignores_absolute_template_but_stored_file_does_not(tmp_path, monkeypatch, mode):
    from limen.yaml.store import commit_manifest, resolve_manifest_uri

    project = tmp_path / 'project'
    config = _project(project)
    config['metadata']['mode'] = mode
    config['uel']['output_path'] = str(tmp_path / 'ignored-absolute-output')
    # The URI's project-root output routing is independent of source-path routing.
    config['sfd']['manifest']['data_source']['params']['file_path_or_url'] = str(
        project / 'inputs/spot_1h_20240101_20241231.parquet',
    )
    submitted = project / 'experiment.yaml'
    _write(submitted, config)
    manifest_id, _ = commit_manifest(submitted, project)
    uri = 'manifest://' + manifest_id
    committed, _ = resolve_manifest_uri(uri, project)
    subdirectory = project / 'nested'
    subdirectory.mkdir()
    monkeypatch.chdir(subdirectory)
    runner = CliRunner()
    dry = runner.invoke(cli, ['run', '--dry-run', uri])
    assert dry.exit_code == 0, dry.output
    result = runner.invoke(cli, ['run', '--no-progress-bar', uri])
    assert result.exit_code == 0, result.output
    prefix = project / ('results/dev' if mode == 'development' else 'results') / committed.stem[:8]
    outputs = list(prefix.glob('*/metadata.json'))
    assert len(outputs) == 1
    metadata = _assert_artifacts(outputs[0].parent, committed, config, copied_name='manifest.yaml')
    assert metadata['manifest_id'] == manifest_id
    assert not Path(config['uel']['output_path']).exists()
    before = _snapshot(tmp_path)
    refused = runner.invoke(cli, ['run', '--no-progress-bar', str(committed)])
    assert refused.exit_code == 1, refused.output
    assert 'uel.output_path' in refused.output
    assert _snapshot(tmp_path) == before


def test_relocated_legacy_absolute_output_run_resumes_without_template_admission(tmp_path, monkeypatch):
    from limen.cli.commands import run as run_command
    from limen.experiment.experiment_core import UniversalExperimentLoop

    project = tmp_path / 'project'
    config = _project(project)
    original_destination = project / 'legacy-output'
    config['uel'].update(output_path=str(original_destination), n_permutations=2, feedback_interval=1)
    submitted = project / 'experiment.yaml'
    _write(submitted, config)
    monkeypatch.chdir(project)
    original_run = UniversalExperimentLoop.run

    def stop_after_first(loop, *args, **kwargs):
        def stop(log, _msq):
            if log.height == 1:
                loop._shutdown_requested = True
        loop._intra_callback = stop
        return original_run(loop, *args, **kwargs)

    # Produce the real checkpoint through the old destination-selection seam.
    with monkeypatch.context() as context:
        context.setattr(run_command, 'admit_output', lambda prefix, expanded: Path(expanded))
        context.setattr(UniversalExperimentLoop, 'run', stop_after_first)
        interrupted = CliRunner().invoke(cli, ['run', '--no-progress-bar', str(submitted)])
    assert interrupted.exit_code == 1, interrupted.output
    assert 'Interrupted; checkpoint retained' in interrupted.output
    saved_yaml = (original_destination / submitted.name).read_bytes()
    relocated = tmp_path / 'relocated-explicit-run'
    shutil.move(original_destination, relocated)
    metadata_before = json.loads((relocated / 'metadata.json').read_text())
    assert metadata_before['yaml_reference']['uel']['output_path'] == str(original_destination)
    before = pl.read_csv(relocated / 'results.csv')
    assert before.height == 1
    result = CliRunner().invoke(cli, ['run', '--no-progress-bar', '--resume', str(relocated)])
    assert result.exit_code == 0, result.output
    after = pl.read_csv(relocated / 'results.csv')
    assert after.height == 2
    assert after.head(1).equals(before)
    assert (relocated / submitted.name).read_bytes() == saved_yaml
    assert json.loads((relocated / 'metadata.json').read_text())['yaml_reference'] == config
    assert not original_destination.exists()

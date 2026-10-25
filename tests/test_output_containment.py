import json
import shutil
import subprocess
from pathlib import Path

import polars as pl
import pytest
from click.testing import CliRunner
from ruamel.yaml import YAML

from limen.cli.main import cli
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


@pytest.mark.parametrize('dry_run', (False, True))
@pytest.mark.parametrize('kind', (
    'absolute-outside', 'absolute-inside', 'parent', 'production', 'sibling',
    'symlink', 'symlink-parent', 'empty', 'dot', 'root-alias',
))
def test_direct_output_refuses_before_project_or_external_writes(tmp_path, monkeypatch, kind, dry_run):
    project = tmp_path / 'project'
    config = _project(project)
    foreign = tmp_path / 'foreign'
    _project(foreign)
    (foreign / 'sentinel').write_text('existing external content')
    prefix = project / 'results/dev'
    prefix.mkdir(parents=True)
    (prefix / 'external').symlink_to(foreign, target_is_directory=True)
    (prefix / 'root').symlink_to(prefix, target_is_directory=True)
    config['uel']['output_path'] = {
        'absolute-outside': str(foreign / 'output'),
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
    assert (output / submitted.name).read_bytes() == submitted.read_bytes()
    metadata = json.loads(outputs[0].read_text())
    assert metadata['yaml_reference'] == config
    results = pl.read_csv(output / 'results.csv')
    assert results.height == 1
    assert results['accuracy'].is_finite().all()
    assert results['auc'].is_finite().all()
    assert (output / 'round_data.jsonl.gz').is_file()

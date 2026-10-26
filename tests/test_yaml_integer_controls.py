import json
from pathlib import Path

import polars as pl
import pytest
from click.testing import CliRunner

from limen.cli.main import cli
from limen.yaml import CompiledSFD, parse, validate

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'docs/examples/compiled-research/recovery.yaml'
FIXTURE = ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet'


def _reference(field, spelling, *, boolean_anchor=None, integer_anchor=False):
    text = REFERENCE.read_text()
    if boolean_anchor is not None:
        text = text.replace('    type: ml\n', f'    type: ml\n    strict_mode: &flag {boolean_anchor}\n', 1)
    if field == 'n_permutations':
        text = text.replace('  n_permutations: 4\n', f'  n_permutations: {spelling}\n', 1)
    else:
        text = text.replace('uel:\n', f'uel:\n  workers: {spelling}\n', 1)
    if integer_anchor:
        text = text.replace('uel:\n', 'uel:\n  feedback_interval: &count 1\n', 1)
    return text


def _boolean_source(field, value, form):
    spelling = value if form == 'literal' else f'&flag {value}' if form == 'anchor' else '*flag'
    return _reference(field, spelling, boolean_anchor=value if form == 'alias' else None)


@pytest.mark.parametrize('field,diagnostic', [
    ('n_permutations', "'uel.n_permutations' must be an int"),
    ('workers', "'uel.workers' must be a int"),
])
@pytest.mark.parametrize('value', ['true', 'false'])
@pytest.mark.parametrize('form', ['literal', 'anchor', 'alias'])
def test_yaml_integer_controls_refuse_boolean_logical_types(field, diagnostic, value, form):
    config, errors = parse(_boolean_source(field, value, form))
    assert not errors
    before = config['uel'][field]
    expected_type = 'bool' if form == 'literal' else 'ScalarBoolean'
    assert type(before).__name__ == expected_type
    result = validate(config)
    assert not result.valid and not result.warnings
    assert [(error.path, error.message) for error in result.errors] == [
        (f'uel.{field}', f'{diagnostic} (got {expected_type})'),
    ]
    assert config['uel'][field] is before


@pytest.mark.parametrize('field', ['n_permutations', 'workers'])
@pytest.mark.parametrize('spelling', ['1', '0x1', '0o1', '001', '&count 1', '*count'])
def test_yaml_integer_controls_preserve_positive_integer_representations(field, spelling):
    config, errors = parse(_reference(field, spelling, integer_anchor=spelling == '*count'))
    assert not errors
    before = config['uel'][field]
    assert isinstance(before, int) and not isinstance(before, bool) and before == 1
    result = validate(config)
    assert result.valid and not result.errors and not result.warnings
    assert config['uel'][field] is before


def _snapshot(root):
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob('*') if path.is_file()}


@pytest.mark.parametrize('field', ['n_permutations', 'workers'])
@pytest.mark.parametrize('value', ['true', 'false'])
@pytest.mark.parametrize('form', ['anchor', 'alias'])
def test_anchored_boolean_controls_refuse_before_cli_construction_and_writes(tmp_path, monkeypatch, field, value, form):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'limen.toml').write_text('')
    submitted = tmp_path / 'experiment.yaml'
    submitted.write_text(_boolean_source(field, value, form).replace('mode: development', 'mode: production'))
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError('Boolean integer control reached compilation or source access')

    monkeypatch.setattr(CompiledSFD, '__init__', forbidden)
    monkeypatch.setattr('limen.experiment.manifest_core.DataSourceResolver.resolve', staticmethod(forbidden))
    before = _snapshot(tmp_path)
    for command in (['validate'], ['run'], ['run', '--dry-run'], ['profile'], ['commit']):
        result = CliRunner().invoke(cli, [*command, str(submitted)])
        assert result.exit_code == 1, result.output
        assert f'[uel.{field}]' in result.output and 'got ScalarBoolean' in result.output
        assert calls == [] and _snapshot(tmp_path) == before
        assert not (tmp_path / 'results').exists() and not (tmp_path / 'manifests').exists()


def test_anchored_integer_budget_and_aliased_worker_run_recorded_trial(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'limen.toml').write_text('')
    text = _reference('n_permutations', '&budget 1')
    text = text.replace('  n_permutations: &budget 1\n', '  n_permutations: &budget 1\n  workers: *budget\n', 1)
    text = text.replace('inputs/' + FIXTURE.name, str(FIXTURE))
    submitted = tmp_path / 'recorded.yaml'
    submitted.write_text(text)
    result = CliRunner().invoke(cli, ['run', '--no-progress-bar', str(submitted)])
    assert result.exit_code == 0, result.output
    output = tmp_path / 'results/dev/compiled-recovery'
    rows = pl.read_csv(output / 'results.csv')
    assert rows.height == 1 and rows['accuracy'].is_finite().all() and rows['auc'].is_finite().all()
    metadata = json.loads((output / 'metadata.json').read_text())
    assert metadata.get('workers', 1) == 1
    assert metadata['yaml_reference']['uel']['n_permutations'] == 1
    assert metadata['yaml_reference']['uel']['workers'] == 1
    assert (output / submitted.name).read_bytes() == submitted.read_bytes()

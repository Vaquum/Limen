import json
from pathlib import Path
import shutil

from click.testing import CliRunner
import polars as pl
import pytest

from limen.cli.main import cli
from limen.yaml import parse, validate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def recorded_cli(tmp_path_factory):
    project = tmp_path_factory.mktemp('yaml-cli-handoff')
    (project / 'inputs').mkdir()
    shutil.copyfile(ROOT / 'tests/fixtures/spot_15m_20250101_20250531.parquet', project / 'inputs/spot_15m_20250101_20250531.parquet')
    shutil.copyfile(ROOT / 'docs/examples/compiled-research/selection.yaml', project / 'selection.yaml')
    runner = CliRunner()
    import os
    previous = Path.cwd()
    os.chdir(project)
    try:
        result = runner.invoke(cli, ['run', '--no-progress-bar', 'selection.yaml'])
        assert result.exit_code == 0, result.output
    finally:
        os.chdir(previous)
    return project


def test_cli_reconstructs_selected_sensor_and_cohort(recorded_cli, monkeypatch):
    monkeypatch.chdir(recorded_cli)
    result = CliRunner().invoke(cli, ['assess', 'selection.yaml', '--results', 'results/dev/compiled-cli'])
    assert result.exit_code == 0, result.output
    output = recorded_cli / 'results/dev/compiled-cli'
    log = pl.read_csv(output / 'results.csv')
    selection = json.loads((output / 'assessment.json').read_text())
    assert selection['selected_ids'] == log.sort(['val_backtest_total_return', 'id'], descending=[True, False])['id'].head(1).to_list()
    tape = pl.read_parquet(output / 'predictions.parquet')
    available = tape.filter(pl.col('reason').is_null())
    assert available.height > 0
    assert available['probability'].is_between(0, 1).all()
    assert available['prediction'].is_in([0., 1.]).all()
    assert available['available_at_ns'].null_count() == available.height
    assert available['datetime'].min().isoformat().startswith('2025-01-22')


def test_assessment_rejects_changed_declaration_before_reconstruction(recorded_cli, monkeypatch):
    monkeypatch.chdir(recorded_cli)
    changed = recorded_cli / 'changed.yaml'
    changed.write_text((recorded_cli / 'selection.yaml').read_text().replace('top_k: 1', 'top_k: 2'))
    def forbidden(*args, **kwargs):
        raise AssertionError('A changed declaration reached reconstruction')
    monkeypatch.setattr('limen.Trainer', forbidden)
    result = CliRunner().invoke(cli, ['assess', 'changed.yaml', '--results', 'results/dev/compiled-cli'])
    assert result.exit_code != 0 and 'differs from the recorded' in result.output


@pytest.mark.parametrize('change', [{'metric': 'accuracy'}, {'top_k': True}, {'folds': [0, 0]}, {'predict': 'yes'}, {'unused': True}])
def test_invalid_assessment_is_rejected(change):
    document, errors = parse(ROOT / 'docs/examples/compiled-research/selection.yaml')
    assert not errors
    document['assessment'].update(change)
    report = validate(document)
    assert not report.valid and any(item.path == 'assessment/holdout' for item in report.errors)


def test_guarded_declaration_requires_reservation_before_fitting(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = ROOT / 'docs/examples/compiled-research/final/holdout.yaml'
    def forbidden(*args, **kwargs):
        raise AssertionError('Unreserved research reached model fitting')
    monkeypatch.setattr('limen.cli.commands.run.UniversalExperimentLoop', forbidden)
    result = CliRunner().invoke(cli, ['run', '--no-progress-bar', str(source)])
    assert result.exit_code != 0 and 'limen reserve' in result.output


def test_cli_accepts_multiple_unchanged_yaml_targets(recorded_cli, monkeypatch):
    calls = []
    monkeypatch.chdir(recorded_cli)
    def observe(path, **kwargs):
        calls.append(path)
        return True
    monkeypatch.setattr('limen.cli.commands.run.run_experiment', observe)
    result = CliRunner().invoke(cli, ['run', '--dry-run', 'selection.yaml', 'selection.yaml'])
    assert result.exit_code == 0 and calls == [Path('selection.yaml'), Path('selection.yaml')]
    result = CliRunner().invoke(cli, ['run', 'selection.yaml', '--resume', 'results/dev/compiled-cli'])
    assert result.exit_code != 0 and 'Cannot specify both' in result.output

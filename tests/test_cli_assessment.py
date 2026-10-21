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


def test_cli_promotes_yaml_walk_forward_folds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'inputs').mkdir()
    for name in ('spot_1h_20240101_20240116.parquet', 'spot_1h_20240101_20241231.parquet'):
        shutil.copyfile(ROOT / 'tests/fixtures' / name, tmp_path / 'inputs' / name)
    shutil.copyfile(ROOT / 'docs/examples/compiled-research/walk_forward.yaml', tmp_path / 'folds.yaml')
    runner = CliRunner()
    executed = runner.invoke(cli, ['run', '--no-progress-bar', 'folds.yaml'])
    assert executed.exit_code == 0, executed.output
    assessed = runner.invoke(cli, ['assess', 'folds.yaml', '--results', 'results/dev/compiled-walk_forward'])
    assert assessed.exit_code == 0, assessed.output
    report = json.loads((tmp_path / 'results/dev/compiled-walk_forward/assessment.json').read_text())
    assert report['folds'] == [0, 1] and report['members'] == 2
    assert {member[2] for member in report['member_ids']} == {0, 1}
    tape = pl.read_parquet(tmp_path / 'results/dev/compiled-walk_forward/predictions.parquet')
    assert tape.filter(pl.col('reason').is_null()).height > 0


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


def test_cli_guarded_holdout_freezes_before_final_assessment(tmp_path, monkeypatch):
    import subprocess
    from limen.experiment import holdout
    from limen.sfd.reference_architecture import LogRegBinary
    project = tmp_path / 'final'
    project.mkdir()
    (project / 'limen.toml').write_text('[project]\nname="recorded-cli-final"\n')
    source = ROOT / 'docs/examples/compiled-research/final/holdout.yaml'
    shutil.copyfile(source, project / 'holdout.yaml')
    (tmp_path / 'inputs').mkdir()
    shutil.copyfile(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet', tmp_path / 'inputs/spot_1h_20240101_20241231.parquet')
    subprocess.run(['git', 'init', '--quiet', str(project)], check=True)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    reserved = runner.invoke(cli, ['reserve', 'final/holdout.yaml'])
    assert reserved.exit_code == 0, reserved.output
    identity = reserved.stdout.strip()
    later = project / 'later.yaml'
    later.write_text(source.read_text().replace('"2024-01-05"', '"2024-01-06"').replace('"2024-01-06"\n  interval_seconds', '"2024-01-07"\n  interval_seconds'))
    second = runner.invoke(cli, ['reserve', 'final/later.yaml'])
    assert second.exit_code == 0, second.output
    fitted = []
    original_train = LogRegBinary.train
    def train(model, prepared, **params):
        assert prepared['holdout']['source']['admitted']['rows'] == 96
        fitted.append(model)
        return original_train(model, prepared, **params)
    monkeypatch.setattr(LogRegBinary, 'train', train)
    executed = runner.invoke(cli, ['run', '--no-progress-bar', 'final/holdout.yaml'])
    assert executed.exit_code == 0, executed.output
    with monkeypatch.context() as guard:
        def no_reconstruction(*args, **kwargs):
            raise AssertionError('A mismatched reservation reached reconstruction')
        guard.setattr('limen.Trainer', no_reconstruction)
        rejected = runner.invoke(cli, ['assess', 'final/holdout.yaml', '--results', 'results/dev/compiled-holdout', '--reservation', second.stdout.strip()])
        assert rejected.exit_code != 0 and 'must be contained' in rejected.output
    original_freeze = holdout.freeze_holdout_selection
    def forbidden(*args, **kwargs):
        raise AssertionError('Final CLI assessment fitted after the selection was frozen')
    def freeze(*args, **kwargs):
        result = original_freeze(*args, **kwargs)
        monkeypatch.setattr(LogRegBinary, 'train', forbidden)
        return result
    monkeypatch.setattr(holdout, 'freeze_holdout_selection', freeze)
    assessed = runner.invoke(cli, ['assess', 'final/holdout.yaml', '--results', 'results/dev/compiled-holdout', '--reservation', identity])
    assert assessed.exit_code == 0, assessed.output
    assert len(fitted) == 2
    report = json.loads((tmp_path / 'results/dev/compiled-holdout/final-report.json').read_text())
    assert report['classification']['accuracy']['denominator'] == 23
    assert report['exposure']['class'] == 'unknown'
    retained = holdout.read_holdout_report(project, report['attempt_id'])
    assert retained == report


def test_cli_resume_rejects_changed_pinned_source(tmp_path, monkeypatch):
    from limen.sfd.reference_architecture import LogRegBinary
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'inputs').mkdir()
    source = tmp_path / 'inputs/spot_1h_20240101_20241231.parquet'
    original = (ROOT / 'tests/fixtures' / source.name).read_bytes()
    source.write_bytes(original)
    shutil.copyfile(ROOT / 'docs/examples/compiled-research/recovery.yaml', tmp_path / 'recovery.yaml')
    runner = CliRunner()
    executed = runner.invoke(cli, ['run', '--no-progress-bar', 'recovery.yaml'])
    assert executed.exit_code == 0, executed.output
    output = tmp_path / 'results/dev/compiled-recovery'
    before = {name: (output / name).read_bytes() for name in ('results.csv', 'metadata.json', 'checkpoint.json')}
    source.write_bytes(original + b'changed')
    def forbidden(*args, **kwargs):
        raise AssertionError('Changed source reached fitting')
    monkeypatch.setattr(LogRegBinary, 'train', forbidden)
    resumed = runner.invoke(cli, ['run', '--no-progress-bar', '--resume', str(output)])
    assert resumed.exit_code != 0 and 'Recorded source SHA-256 differs' in resumed.output
    assert {name: (output / name).read_bytes() for name in before} == before

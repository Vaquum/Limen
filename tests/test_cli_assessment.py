import json
import hashlib
import math
from datetime import datetime
from pathlib import Path
import shutil

from click.testing import CliRunner
import polars as pl
import pytest

from limen.cli.main import cli
from limen.inference import ReconstructionError, Trainer
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


@pytest.fixture(scope='module')
def recorded_cli_admission(tmp_path_factory):
    project = tmp_path_factory.mktemp('yaml-cli-admission')
    (project / 'inputs').mkdir()
    name = 'spot_1h_20240101_20241231.parquet'
    shutil.copyfile(ROOT / 'tests/fixtures' / name, project / 'inputs' / name)
    source = (ROOT / 'docs/examples/compiled-research/recovery.yaml').read_text().replace(
        'output_format: csv', 'output_format: parquet',
    )
    submitted = project / 'recovery.yaml'
    submitted.write_text(source)
    (project / 'limen.toml').write_text('[project]\nname = "recorded-cli-admission"\n')
    output = project / 'results/dev/compiled-recovery'
    output.mkdir(parents=True)
    saved = output / submitted.name
    saved.write_text(source.replace('    - 0.1\n', '    - 0.25\n'))
    assert saved.read_bytes() != submitted.read_bytes()
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(project)
        result = CliRunner().invoke(cli, ['run', '--no-progress-bar', str(submitted)])
    assert result.exit_code == 0, result.output
    assert saved.read_bytes() == submitted.read_bytes()
    assert all((output / name).is_file() for name in (
        'checkpoint.json', 'metadata.json', 'results.csv', 'results.parquet',
        'round_data.jsonl.gz', 'convergence_report.json',
    ))
    assert pl.read_csv(output / 'results.csv').height == 4
    return project


def _run_snapshot(directory):
    return {
        str(path.relative_to(directory)):
        (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) if path.is_file() else None
        for path in sorted(directory.rglob('*'))
    }


@pytest.mark.parametrize('kind,partial,workers', (
    ('same', False, 1),
    ('same', True, 1),
    ('different', False, 2),
    ('self', False, 1),
    ('committed', False, 1),
))
def test_refused_cli_run_preserves_saved_directory(
    recorded_cli_admission, tmp_path, monkeypatch, kind, partial, workers,
):
    project = tmp_path / 'project'
    shutil.copytree(recorded_cli_admission, project)
    monkeypatch.chdir(project)
    output = project / 'results/dev/compiled-recovery'
    submitted = project / 'recovery.yaml'
    if kind in ('same', 'different'):
        attempted = project / 'attempt' / ('recovery.yaml' if kind == 'same' else 'different.yaml')
        attempted.parent.mkdir()
        attempted.write_text(submitted.read_text().replace('    - 0.1\n', '    - 0.25\n'))
        target = str(attempted)
    elif kind == 'self':
        target = str(output / 'recovery.yaml')
    else:
        from limen.yaml.store import commit_manifest, resolve_manifest_uri
        manifest_id, _ = commit_manifest(submitted, project)
        target = 'manifest://' + manifest_id
        committed, _ = resolve_manifest_uri(target, project)

        class FixedDateTime:
            @staticmethod
            def now():
                return datetime(2026, 1, 1)

        monkeypatch.setattr('limen.cli.commands.run.datetime', FixedDateTime)
        destination = project / 'results/dev' / committed.stem[:8] / '20260101_000000'
        shutil.copytree(output, destination)
        (destination / 'manifest.yaml').write_bytes(committed.read_bytes())
        output = destination
    if partial:
        (output / 'checkpoint.json').unlink()
    before = _run_snapshot(output)
    copy_calls = []
    if kind in ('self', 'committed'):
        def forbidden_copy(*args, **kwargs):
            copy_calls.append(args)
            raise AssertionError('An occupied run reached YAML copying')
        monkeypatch.setattr('limen.cli.commands.run.shutil.copy2', forbidden_copy)
    result = CliRunner().invoke(cli, ['run', '--no-progress-bar', '--workers', str(workers), target])
    message = 'contains partial artifacts' if partial else 'already contains artifacts'
    assert result.exit_code == 1 and message in result.output, result.output
    assert _run_snapshot(output) == before
    assert copy_calls == []


def test_cli_resume_keeps_saved_yaml_and_recorded_definition(recorded_cli_admission, tmp_path, monkeypatch):
    project = tmp_path / 'project'
    shutil.copytree(recorded_cli_admission, project)
    monkeypatch.chdir(project)
    output = project / 'results/dev/compiled-recovery'
    saved = output / 'recovery.yaml'
    saved.write_text(saved.read_text().replace('    - 0.1\n', '    - 0.25\n'))
    copied_yaml = saved.read_bytes()
    definition = json.loads((output / 'metadata.json').read_text())['yaml_reference']
    results = pl.read_csv(output / 'results.csv')
    resumed = CliRunner().invoke(cli, ['run', '--no-progress-bar', '--resume', str(output)])
    assert resumed.exit_code == 0, resumed.output
    assert saved.read_bytes() == copied_yaml
    assert json.loads((output / 'metadata.json').read_text())['yaml_reference'] == definition
    assert pl.read_csv(output / 'results.csv').equals(results)


@pytest.mark.parametrize('deterministic', (False, True))
@pytest.mark.parametrize('original_index', range(4))
@pytest.mark.parametrize('replay_index', range(4))
def test_recorded_metric_nonfinite_comparison(recorded_cli, deterministic, original_index, replay_index):
    finite = pl.read_csv(recorded_cli / 'results/dev/compiled-cli/results.csv')['auc'][0]
    assert isinstance(finite, float) and math.isfinite(finite)
    values = (finite, math.nan, math.inf, -math.inf)
    original, replayed = values[original_index], values[replay_index]
    trainer = object.__new__(Trainer)
    trainer._param_keys = frozenset()
    trainer._original_log = {'recorded': {'auc': original}}
    mismatches = trainer._validate_metrics('recorded', {'auc': replayed}, deterministic)
    assert mismatches == ([] if original_index == replay_index else
                          [f'auc: original={original}, new={replayed}'])


@pytest.mark.parametrize('deterministic', (False, True))
def test_metric_comparison_preserves_scalar_policy(deterministic):
    trainer = object.__new__(Trainer)
    trainer._param_keys = frozenset()
    for original, replayed, exact_match, stochastic_match in (
        (0.0, 1e-6, True, True),
        (0.0, math.nextafter(1e-6, math.inf), False, True),
        (0.0, 0.01, False, True),
        (0.0, math.nextafter(0.01, math.inf), False, False),
        (99.0, 100.0, False, True),
        (99.0, math.nextafter(100.0, math.inf), False, False),
        (2**53, 2**53 + 1, False, True),
        (7, 7.0, True, True), (7.0, 7, True, True),
        (True, 1, True, True), (1, True, True, True), (False, 0, True, True),
        (None, math.nan, True, True), (math.nan, None, True, True),
        ('nan', math.nan, True, True), (math.nan, 'nan', True, True),
    ):
        trainer._original_log = {'recorded': {'auc': original}}
        mismatches = trainer._validate_metrics('recorded', {'auc': replayed}, deterministic)
        assert (not mismatches) == (exact_match if deterministic else stochastic_match)
    if deterministic:
        for replayed, matches in ((10**400, True), (10**400 + 1, False)):
            trainer._original_log = {'recorded': {'count': 10**400}}
            assert (not trainer._validate_metrics('recorded', {'count': replayed}, True)) == matches


def test_recorded_replay_rejects_nan_before_sensor(recorded_cli, monkeypatch):
    monkeypatch.chdir(recorded_cli)
    trainer = Trainer(recorded_cli / 'results/dev/compiled-cli')
    permutation_id = next(iter(trainer._round_data))
    original = trainer._original_log[permutation_id]['auc']
    assert isinstance(original, float) and math.isfinite(original)
    sensors = trainer.train([permutation_id])
    assert len(sensors) == 1 and sensors[0].permutation_id == permutation_id
    assert sensors[0]._model.deterministic is False
    run_model = trainer._manifest.run_model
    replayed, constructions = [], []

    def faulty_metric(data, params):
        results = run_model(data, params)
        assert results['_model'].deterministic is sensors[0]._model.deterministic
        results['auc'] = math.nan
        replayed.append(permutation_id)
        return results

    def forbidden_sensor(**kwargs):
        constructions.append(kwargs['permutation_id'])
        raise AssertionError('A nonfinite metric mismatch reached Sensor construction')

    monkeypatch.setattr(trainer._manifest, 'run_model', faulty_metric)
    monkeypatch.setattr('limen.inference.trainer.Sensor', forbidden_sensor)
    with pytest.raises(ReconstructionError, match='auc'):
        trainer.train([permutation_id])
    assert replayed == [permutation_id]
    assert constructions == []


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


@pytest.mark.parametrize('variant', ['final', 'final-event'])
def test_cli_guarded_holdout_freezes_before_final_assessment(tmp_path, monkeypatch, variant):
    import subprocess
    from limen.experiment import holdout
    from limen.sfd.reference_architecture import LogRegBinary
    project = tmp_path / 'final'
    project.mkdir()
    (project / 'limen.toml').write_text('[project]\nname="recorded-cli-final"\n')
    source = ROOT / 'docs/examples/compiled-research' / variant / 'holdout.yaml'
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
    import limen.cli.commands.run as run_command
    original_loop = run_command.UniversalExperimentLoop
    def construct(*args, **kwargs):
        with pytest.raises(ValueError, match='Cannot change reservations inside active research'):
            holdout.withdraw_holdout(project, identity)
        return original_loop(*args, **kwargs)
    monkeypatch.setattr(run_command, 'UniversalExperimentLoop', construct)
    executed = runner.invoke(cli, ['run', '--no-progress-bar', 'final/holdout.yaml'])
    assert executed.exit_code == 0, executed.output
    results = 'results/dev/compiled-holdout_event' if variant == 'final-event' else 'results/dev/compiled-holdout'
    with monkeypatch.context() as guard:
        def no_reconstruction(*args, **kwargs):
            raise AssertionError('A mismatched reservation reached reconstruction')
        guard.setattr('limen.Trainer', no_reconstruction)
        rejected = runner.invoke(cli, ['assess', 'final/holdout.yaml', '--results', results, '--reservation', second.stdout.strip()])
        assert rejected.exit_code != 0 and 'must be contained' in rejected.output
    original_freeze = holdout.freeze_holdout_selection
    def forbidden(*args, **kwargs):
        raise AssertionError('Final CLI assessment fitted after the selection was frozen')
    def freeze(*args, **kwargs):
        result = original_freeze(*args, **kwargs)
        monkeypatch.setattr(LogRegBinary, 'train', forbidden)
        return result
    monkeypatch.setattr(holdout, 'freeze_holdout_selection', freeze)
    assessed = runner.invoke(cli, ['assess', 'final/holdout.yaml', '--results', results, '--reservation', identity])
    assert assessed.exit_code == 0, assessed.output
    assert len(fitted) == 2
    report = json.loads((tmp_path / results / 'final-report.json').read_text())
    assert report['classification']['accuracy']['denominator'] == 23
    assert report['exposure']['class'] == 'unknown'
    retained = holdout.read_holdout_report(project, report['attempt_id'])
    assert retained == report
    if variant == 'final-event':
        assert report['schema_version'] == 2 and report['scenarios'] == []
        assert len(report['execution']['tape']) == len(report['execution']['signals']) == 24
        assert report['execution']['ledger']['funding'] == []


@pytest.mark.parametrize('change', ['snapshot', 'cohort', 'fold'])
def test_empty_scenarios_requires_one_event_sensor(change):
    document, errors = parse(ROOT / 'docs/examples/compiled-research/final-event/holdout.yaml')
    assert not errors
    if change == 'snapshot':
        del document['sfd']['manifest']['backtest']['product']
    elif change == 'cohort':
        document['assessment']['top_k'] = 2
    else:
        document['assessment']['folds'] = [0]
    report = validate(document)
    assert not report.valid and any('one ordinary selected Sensor' in item.message for item in report.errors)


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


@pytest.mark.parametrize('signum', ['SIGINT', 'SIGTERM'])
def test_cli_interrupt_stops_before_next_yaml(tmp_path, monkeypatch, signum):
    import signal
    import limen.cli.commands.run as command
    from limen.sfd.reference_architecture import LogRegBinary
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'inputs').mkdir()
    name = 'spot_1h_20240101_20241231.parquet'
    shutil.copyfile(ROOT / 'tests/fixtures' / name, tmp_path / 'inputs' / name)
    for name in ('recovery.yaml', 'selection.yaml'):
        shutil.copyfile(ROOT / 'docs/examples/compiled-research' / name, tmp_path / name)
    invoked = []
    original_run, original_train = command.run_experiment, LogRegBinary.train
    def run(path, **kwargs):
        invoked.append(path)
        return original_run(path, **kwargs)
    def interrupt(model, data, **params):
        result = original_train(model, data, **params)
        signal.raise_signal(getattr(signal, signum))
        return result
    monkeypatch.setattr(command, 'run_experiment', run)
    monkeypatch.setattr(LogRegBinary, 'train', interrupt)
    previous = {item: signal.getsignal(item) for item in (signal.SIGINT, signal.SIGTERM)}
    try:
        result = CliRunner().invoke(cli, ['run', '--no-progress-bar', 'recovery.yaml', 'selection.yaml'])
    finally:
        for item, handler in previous.items():
            signal.signal(item, handler)
    assert result.exit_code != 0 and 'Interrupted; checkpoint retained' in result.output
    assert invoked == [Path('recovery.yaml')]
    assert (tmp_path / 'results/dev/compiled-recovery/checkpoint.json').is_file()
    assert not (tmp_path / 'results/dev/compiled-cli').exists()

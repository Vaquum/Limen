import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner
from threadpoolctl import threadpool_limits

from limen.cli.main import cli
from limen.experiment.manifest_core import DataSourceResolver
from limen.yaml import CompiledSFD, parse, validate
from limen.yaml.config import round_trip_yaml
from limen.yaml.store import commit_manifest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet'


def _config():
    config, errors = parse(ROOT / 'docs/examples/compiled-research/recovery.yaml')
    assert not errors
    return config


@pytest.mark.parametrize('field', ['feedback_interval', 'checkpoint_interval'])
@pytest.mark.parametrize('value', [0, -1, -100])
def test_nonpositive_intervals_are_rejected_at_admission(field, value):
    config = _config()
    config['uel'][field] = value
    result = validate(config)
    assert not result.valid
    assert [(error.path, error.message) for error in result.errors] == [
        (f'uel.{field}', f"'uel.{field}' must be a positive integer")]


@pytest.mark.parametrize('field', ['feedback_interval', 'checkpoint_interval'])
@pytest.mark.parametrize('value', [True, False, 1.0, 0.0, -1.0, '100', None, [1], {}, float('nan'), float('inf')])
def test_interval_type_errors_remain_single_and_precede_range_checks(field, value):
    config = _config()
    config['uel'][field] = value
    result = validate(config)
    assert not result.valid
    assert [(error.path, error.message) for error in result.errors] == [
        (f'uel.{field}', f"'uel.{field}' must be a int (got {type(value).__name__})")]


@pytest.mark.parametrize('field', ['feedback_interval', 'checkpoint_interval'])
@pytest.mark.parametrize('value', [None, 1, 2, 1000, 10**9])
def test_omission_and_positive_intervals_keep_their_existing_admission(field, value):
    config = _config()
    config['uel'].pop(field, None)
    if value is not None:
        config['uel'][field] = value
    result = validate(config)
    assert result.valid and not result.errors and not result.warnings


@pytest.mark.parametrize('spelling', ['1', '0x01', '0o1', '01', '&value 1'])
def test_real_yaml_integer_subclasses_and_aliases_are_preserved(spelling):
    config = _config()
    text = 'feedback_interval: ' + spelling + '\ncheckpoint_interval: ' + ('*value' if spelling.startswith('&') else spelling)
    parsed, errors = parse(text)
    assert not errors
    config['uel'].update(parsed)
    assert validate(config).valid
    assert all(isinstance(value, int) and not isinstance(value, bool) and value == 1 for value in parsed.values())


@pytest.mark.parametrize('literal', ['true', 'false'])
def test_anchored_yaml_booleans_cannot_masquerade_as_integer_intervals(literal):
    parsed, errors = parse(f'feedback_interval: &flag {literal}\ncheckpoint_interval: *flag')
    assert not errors
    config = _config()
    config['uel'].update(parsed)
    result = validate(config)
    assert not result.valid
    assert [error.path for error in result.errors] == ['uel.feedback_interval', 'uel.checkpoint_interval']
    assert all('must be a int' in error.message for error in result.errors)


def test_both_invalid_intervals_and_workers_keep_independent_diagnostics():
    config = _config()
    config['uel'].update(feedback_interval=0, checkpoint_interval=-1, workers=0)
    result = validate(config)
    assert [(error.path, error.message) for error in result.errors] == [
        (f'uel.{field}', f"'uel.{field}' must be a positive integer")
        for field in ('workers', 'feedback_interval', 'checkpoint_interval')]


def _write(path, config):
    with path.open('w') as stream:
        round_trip_yaml().dump(config, stream)


def _snapshot(directory):
    return {str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob('*') if path.is_file()}


def _forbidden(*args, **kwargs):
    raise AssertionError('Invalid interval reached compilation or source access')


@pytest.mark.parametrize('field', ['feedback_interval', 'checkpoint_interval'])
@pytest.mark.parametrize('value', [0, -1])
@pytest.mark.parametrize('command', [('validate',), ('run', '--dry-run'), ('run', '--no-progress-bar'), ('commit',)])
def test_cli_rejects_before_creating_even_an_orphan_results_tree(tmp_path, monkeypatch, field, value, command):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'limen.toml').write_text('')
    config = _config()
    config['uel'].update({field: value, 'output_path': 'refused-interval'})
    path = tmp_path / 'interval.yaml'
    _write(path, config)
    before = _snapshot(tmp_path)
    monkeypatch.setattr(CompiledSFD, '__init__', _forbidden)
    monkeypatch.setattr(DataSourceResolver, 'resolve', staticmethod(_forbidden))
    result = CliRunner().invoke(cli, [*command, str(path)])
    assert result.exit_code == 1 and f'ERROR  [uel.{field}]' in result.output
    assert _snapshot(tmp_path) == before
    assert not (tmp_path / 'results').exists() and not (tmp_path / 'manifests').exists()


def test_previously_stored_invalid_interval_is_refused_without_store_mutation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'limen.toml').write_text('')
    config = _config()
    config['uel']['feedback_interval'] = 0
    path = tmp_path / 'legacy.yaml'
    _write(path, config)
    identity, _ = commit_manifest(path, tmp_path)
    before = _snapshot(tmp_path)
    monkeypatch.setattr(CompiledSFD, '__init__', _forbidden)
    result = CliRunner().invoke(cli, ['run', '--no-progress-bar', 'manifest://' + identity])
    assert result.exit_code == 1 and 'ERROR  [uel.feedback_interval]' in result.output
    assert _snapshot(tmp_path) == before


def test_recorded_periodic_feedback_and_checkpoint_controls_are_independent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    baseline = None
    with threadpool_limits(limits=1):
        for feedback, checkpoint in ((1, 1), (1, 1000), (100, 1), (100, 1000)):
            config = _config()
            config['sfd']['manifest']['data_source']['params']['file_path_or_url'] = str(FIXTURE)
            config['sfd']['params']['C'] = config['sfd']['params']['C'][:2]
            name = f'interval-{feedback}-{checkpoint}'
            config['uel'].update(n_permutations=2, output_path=name, feedback_interval=feedback, checkpoint_interval=checkpoint)
            config['uel'].pop('pruning_strategies', None)
            path = tmp_path / (name + '.yaml')
            _write(path, config)
            result = CliRunner().invoke(cli, ['run', '--no-progress-bar', str(path)])
            assert result.exit_code == 0, result.output
            directory = tmp_path / 'results/dev' / name
            results = pl.read_csv(directory / 'results.csv')
            assert results.height == 2 and np.isfinite(results.select('accuracy', 'auc').to_numpy()).all()
            scientific = results.drop('execution_time')
            if baseline is None:
                baseline = scientific
            else:
                assert scientific.equals(baseline)
            audit_path, checkpoint_path = directory / 'audit.jsonl', directory / 'checkpoint.json'
            assert audit_path.exists() is (feedback == 1)
            if audit_path.exists():
                audit = [json.loads(line) for line in audit_path.read_text().splitlines()]
                assert len(audit) == 1 and audit[0]['trigger_number'] == 0
                assert audit[0]['errors'] == audit[0]['interventions'] == []
            assert checkpoint_path.exists() is (checkpoint == 1)
            if checkpoint_path.exists():
                state = json.loads(checkpoint_path.read_text())
                assert state['metadata']['experiment_round'] == 1
                assert state['metadata']['target_permutations'] == 2

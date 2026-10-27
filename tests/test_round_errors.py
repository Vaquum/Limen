import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from limen.experiment import UniversalExperimentLoop
from limen.experiment import _round_storage as storage
from limen.experiment._holdout_history import HoldoutEvidenceError
from limen.yaml import CompiledSFD, build_search_strategy, validate
from tests.test_factorized_preparation import _bars, _config

FULL = dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True)


def config(values=(1., -1., .1)):
    result = _config()
    result['sfd']['manifest']['reference_architecture'] = 'limen.sfd.reference_architecture.logreg_binary'
    result['sfd']['params'] = {'lookback': [2], 'quantile': [.5], 'fee': [4.],
                             'C': list(values), 'random_state': [42], 'max_iter': [30]}
    result['uel'].update(n_permutations=len(values), recording=FULL,
                         on_round_error='record', max_round_errors=2)
    return result


def loop(reference, directory, *, mode='record', limit=2, callback=None, search=True, checkpoint=1000):
    return UniversalExperimentLoop(sfd=CompiledSFD(reference), data=_bars(),
        yaml_reference=reference, experiment_dir=directory,
        search_strategy=build_search_strategy(reference) if search else None,
        checkpoint_interval=checkpoint, feedback_interval=1, intra_callback=callback,
        on_round_error=mode, max_round_errors=limit)


def run(experiment, count=3, **kwargs):
    experiment.run('errors', n_permutations=count, prep_each_round=True,
                   progress_bar=False, recording=FULL, **kwargs)


def rows(directory):
    with (directory / 'results.csv').open(newline='') as stream:
        return list(csv.DictReader(stream))


def fingerprint(directory):
    return {str(path.relative_to(directory)): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for path in directory.rglob('*') if path.is_file()}


def test_recorded_later_failure_retains_candidate_and_continues(tmp_path):
    experiment = loop(config(), tmp_path)
    run(experiment)
    records = rows(tmp_path)
    assert len(records) == 3
    assert [row['C'] for row in records] == ['1.0', '-1.0', '0.1']
    assert [bool(row['strict_mode_error']) for row in records] == [False, True, False]
    assert 'InvalidParameterError' in records[1]['strict_mode_error']
    assert records[1]['auc'] == '' and records[1]['accuracy'] == ''
    assert records[1]['id'] == experiment._search_strategy.compute_param_hash({'lookback': 2, 'quantile': .5, 'fee': 4., 'C': -1., 'random_state': 42, 'max_iter': 30})
    with storage.open_text(storage.round_path(tmp_path)) as stream:
        successful = [json.loads(line) for line in stream]
    assert [entry['_round_index'] for entry in successful] == [0, 2]
    assert json.loads((tmp_path / 'metadata.json').read_text())['round_error_policy'] == {'on_round_error': 'record', 'max_round_errors': 2}


def test_default_and_first_round_failures_abort(tmp_path):
    with pytest.raises(ValueError):
        run(loop(config(), tmp_path / 'default', mode='abort', limit=None))
    assert len(rows(tmp_path / 'default')) == 1
    with pytest.raises(RuntimeError, match='successful first round'):
        run(loop(config((-1., 1., .1)), tmp_path / 'first'))
    assert not (tmp_path / 'first/results.csv').exists()


def test_budget_exhaustion_commits_failed_prefix_and_refuses_resume(tmp_path):
    reference = config((1., -1., -2., .1))
    experiment = loop(reference, tmp_path, limit=1)
    with pytest.raises(RuntimeError, match='budget exceeded'):
        run(experiment, count=4)
    with pytest.raises(ValueError, match='successful'):
        experiment.save(tmp_path / 'failed.uel')
    assert len(rows(tmp_path)) == 3
    checkpoint = json.loads((tmp_path / 'checkpoint.json').read_text())
    assert checkpoint['metadata']['experiment_round'] == 2
    assert checkpoint['msq_state']['yielded_count'] == 3
    before = fingerprint(tmp_path)
    with pytest.raises(ValueError, match='exhausted'):
        run(loop(reference, tmp_path, limit=1), count=4, resume=True)
    assert fingerprint(tmp_path) == before


def test_resume_preserves_failed_occurrences_and_policy(tmp_path):
    reference = config((1., -1., .1, .2))
    interrupted = loop(reference, tmp_path)

    def stop(log, _msq):
        if log.height == 2:
            interrupted._shutdown_requested = True

    interrupted._intra_callback = stop
    run(interrupted, count=4)
    assert len(rows(tmp_path)) == 2
    before = fingerprint(tmp_path)
    for mode, limit in (('record', 1), ('abort', None)):
        with pytest.raises(ValueError, match='changed round error policy'):
            run(loop(reference, tmp_path, mode=mode, limit=limit), count=4, resume=True)
        assert fingerprint(tmp_path) == before
    resumed = loop(reference, tmp_path)
    run(resumed, count=4, resume=True)
    assert [row['C'] for row in rows(tmp_path)] == ['1.0', '-1.0', '0.1', '0.2']
    assert resumed._round_error_policy.failures == 1


def test_resume_counts_only_the_committed_failure_prefix(tmp_path):
    reference = config((1., .2, .3, -1., .1))
    interrupted = loop(reference, tmp_path, limit=1, checkpoint=2)
    original = interrupted.model

    def crash(data, round_params):
        if round_params['C'] == .1:
            raise SystemExit('uncheckpointed failure control')
        return original(data, round_params)

    interrupted.model = crash
    with pytest.raises(SystemExit, match='uncheckpointed failure control'):
        run(interrupted, count=5)
    assert len(rows(tmp_path)) == 4 and rows(tmp_path)[3]['strict_mode_error']
    assert json.loads((tmp_path / 'checkpoint.json').read_text())['metadata']['experiment_round'] == 2
    resumed = loop(reference, tmp_path, limit=1)
    run(resumed, count=5, resume=True)
    assert [row['C'] for row in rows(tmp_path)] == ['1.0', '0.2', '0.3', '-1.0', '0.1']
    assert resumed._round_error_policy.failures == 1


@pytest.mark.parametrize('options', [{'workers': 2}, {'factorize': True}, {'search': False}])
def test_record_mode_refuses_unsupported_routes_before_artifacts(tmp_path, options):
    directory = tmp_path / 'unsupported'
    settings = dict(options)
    with pytest.raises(ValueError, match='ordinary unprotected serial MSQ'):
        experiment = loop(config(), directory, search=settings.pop('search', True))
        run(experiment, **settings)
    assert not directory.exists()


def test_integrity_errors_are_not_candidate_failures(tmp_path):
    experiment = loop(config(), tmp_path)
    original = experiment.model

    def model(data, round_params):
        if round_params['C'] < 0:
            raise HoldoutEvidenceError('integrity witness changed')
        return original(data, round_params)

    experiment.model = model
    with pytest.raises(HoldoutEvidenceError, match='integrity witness'):
        run(experiment)
    assert len(rows(tmp_path)) == 1


@pytest.mark.parametrize('values', [{'on_round_error': 'skip'}, {'on_round_error': True}, {'max_round_errors': True},
                                   {'max_round_errors': 0}, {'max_round_errors': None}, {'workers': 2}, {'factorize': True}])
def test_yaml_refuses_invalid_or_unsupported_policy(values):
    reference = config()
    reference['uel'].update(values)
    assert not validate(reference).valid


def test_real_cli_records_a_later_failure(tmp_path):
    from ruamel.yaml import YAML
    reference = config()
    reference['uel']['output_path'] = 'candidate-errors'
    path = tmp_path / 'errors.yaml'
    with path.open('w') as stream:
        YAML().dump(reference, stream)
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    result = subprocess.run([sys.executable, '-m', 'limen', 'run', '--no-progress-bar', str(path)],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, check=True)
    assert 'Experiment complete' in result.stdout
    records = rows(tmp_path / 'results/dev/candidate-errors')
    assert len(records) == 3 and records[1]['strict_mode_error']

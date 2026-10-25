import csv
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from unittest.mock import patch

import polars as pl
import pytest

from limen.experiment._walk_forward_run import WalkForwardRun
from limen.experiment.errors import StrictModeError
from limen.experiment.manifest_core import MLManifest
from tests.test_parallel_rounds import records
from tests.test_result_field_collisions import (
    MODES, _assert_message, _configuration, _loop, _rows, _run, _snapshot,
)


OBJECTIVE = 'val_backtest_total_return'


def _strict_then_artifacts(original, manifest, data, round_params):
    if round_params['C'] == 0.1:
        raise StrictModeError('Recorded strict-mode control')
    result = original(manifest, data, round_params)
    result['extras'] = data['_alignment']
    result['models'] = result['_model']
    return result


def _controlled_worker(function, parameters):
    original = MLManifest.run_model

    def run_model(manifest, data, round_params):
        return _strict_then_artifacts(original, manifest, data, round_params)

    with patch.object(MLManifest, 'run_model', run_model):
        return function(parameters)


class _ArtifactPool(ProcessPoolExecutor):
    def submit(self, function, parameters):
        return super().submit(_controlled_worker, function, parameters)


@contextmanager
def _artifact_control():
    original = MLManifest.run_model

    def run_model(manifest, data, round_params):
        return _strict_then_artifacts(original, manifest, data, round_params)

    with patch.object(MLManifest, 'run_model', run_model), \
            patch('limen.experiment._parallel_rounds.ProcessPoolExecutor', _ArtifactPool):
        yield


@pytest.mark.parametrize('mode', MODES)
def test_strict_failures_and_extracted_artifacts_keep_their_contract(mode, tmp_path):
    config = _configuration()
    config['sfd']['params'].update(extras=['domain extras'], models=['caller models'],
                                  val_backtest_total_return=[7])
    context = {'extras': 'context extras', '_model': 'caller model', '_preds': 'caller predictions'}
    if mode == 'legacy':
        context['_experiment_details'] = 'caller detail without the details flag'
    with _artifact_control():
        loop = _loop(mode, config, tmp_path)
        _run(loop, mode, context_params=context, post_processing=mode != 'parallel')
    rows = _rows(tmp_path)
    assert [row['strict_mode_error'] for row in rows] == ['Recorded strict-mode control', '']
    assert {row['extras'] for row in rows} == {'context extras'}
    assert {row['models'] for row in rows} == {'caller models'}
    assert {row['_model'] for row in rows} == {'caller model'}
    assert {row['_preds'] for row in rows} == {'caller predictions'}
    assert {row[OBJECTIVE] for row in rows} == {'7'}
    assert loop.experiment_log['strict_mode_error'].to_list() == ['Recorded strict-mode control', None]
    if mode == 'legacy':
        assert {row['_experiment_details'] for row in rows} == {context['_experiment_details']}
    if mode != 'parallel':
        assert len(loop.extras) == len(loop.preds) == len(loop.round_params) == 1
        assert len(loop.models) == 2
        assert loop.models[0] is loop.models[1]
        assert loop.extras[0]['first_test_datetime'] is not None
        assert len(loop.preds[0]) > 0
    if mode != 'legacy':
        saved = records(tmp_path)
        assert len(saved) == 1
        assert saved[0]['_round_index'] == 1
        assert saved[0]['round_id'] == rows[1]['id']
        assert len(saved[0]['preds']) > 0


def test_legacy_override_is_validated_before_output(tmp_path):
    config = _configuration()
    loop = _loop('legacy', config, tmp_path / 'run')
    overridden = config['sfd']['params'] | {'id': ['caller identity'], '_param_hash': ['caller dedupe']}
    with pytest.raises(ValueError) as error:
        _run(loop, 'legacy', params=lambda: overridden)
    _assert_message(error, 'domain', ('_param_hash', 'id'))
    assert not (tmp_path / 'run').exists()


def test_msq_validates_its_effective_domain(tmp_path):
    config = _configuration()
    loop = _loop('msq', config, tmp_path)
    loop.params = {'id': ['inactive SFD declaration']}

    _run(loop, 'msq')
    assert len(records(tmp_path)) == 2
    assert all(row['id'] != 'inactive SFD declaration' for row in _rows(tmp_path))


def test_active_objective_reserves_only_its_parameter_name(tmp_path):
    config = _configuration(objective=True)
    config['sfd']['params'][OBJECTIVE] = [7]
    loop = _loop('msq', config, tmp_path / 'rejected')
    with pytest.raises(ValueError) as error:
        _run(loop, 'msq', context_params={OBJECTIVE: 7})
    _assert_message(error, 'domain', (OBJECTIVE,))
    _assert_message(error, 'context', (OBJECTIVE,))
    assert not (tmp_path / 'rejected').exists()
    config['sfd']['params'].pop(OBJECTIVE)
    valid = _loop('msq', config, tmp_path / 'valid')
    _run(valid, 'msq')
    assert valid.experiment_log[OBJECTIVE].dtype == pl.Float64
    assert valid.experiment_log[OBJECTIVE].null_count() == 0


def test_legacy_details_reservation_is_conditional(tmp_path):
    config = _configuration()
    loop = _loop('legacy', config, tmp_path / 'rejected')
    loop.params['_experiment_details'] = ['caller detail']
    with pytest.raises(ValueError) as error:
        _run(loop, 'legacy', maintain_details_in_params=True,
             context_params={'_experiment_details': 'caller detail'})
    _assert_message(error, 'domain', ('_experiment_details',))
    _assert_message(error, 'context', ('_experiment_details',))
    assert not (tmp_path / 'rejected').exists()
    valid = _loop('legacy', config, tmp_path / 'generated')
    _run(valid, 'legacy', maintain_details_in_params=True)
    assert '_experiment_details' not in valid.experiment_log.columns
    assert valid.experiment_log.height == 2


@pytest.mark.parametrize(('mode', 'forged'), (
    ('msq', (0, 1)), ('msq', (1,)), ('parallel', (1,)),
))
def test_resume_refuses_saved_forged_status_before_any_rewrite(mode, forged, tmp_path, monkeypatch):
    config = _configuration(walk_forward=True)
    loop = _loop(mode, config, tmp_path)
    _run(loop, mode)
    saved = _rows(tmp_path)
    evidence = records(tmp_path)
    assert len(saved) == len(evidence) == 2
    for index in forged:
        saved[index]['strict_mode_error'] = 'historical caller status'
    with (tmp_path / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(saved[0]))
        writer.writeheader()
        writer.writerows(saved)
    before = _snapshot(tmp_path)
    assert 'trial_returns.parquet' in {str(path) for path in before}

    def unexpected(*args, **kwargs):
        raise AssertionError('Forged-status resume reached restoration or execution')

    monkeypatch.setattr(WalkForwardRun, 'restore', unexpected)
    monkeypatch.setattr('limen.experiment._round_storage.truncate_rounds', unexpected)
    monkeypatch.setattr('limen.experiment._parallel_rounds.ProcessPoolExecutor', unexpected)
    resumed = _loop(mode, config, tmp_path)
    with pytest.raises(ValueError) as error:
        _run(resumed, mode, resume=True)
    assert 'CSV' in str(error.value)
    assert _snapshot(tmp_path) == before
    assert records(tmp_path) == evidence


@pytest.mark.parametrize('mode', ('msq', 'parallel'))
def test_static_rejected_resume_keeps_all_existing_artifacts(mode, tmp_path):
    config = _configuration(walk_forward=True)
    _run(_loop(mode, config, tmp_path), mode)
    before = _snapshot(tmp_path)
    loop = _loop(mode, config, tmp_path)
    with pytest.raises(ValueError) as error:
        _run(loop, mode, resume=True, context_params={'_id': 'caller identity', '_round_index': 0})
    _assert_message(error, 'context', ('_id', '_round_index'))
    assert _snapshot(tmp_path) == before

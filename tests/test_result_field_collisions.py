import csv
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from unittest.mock import patch

import polars as pl
import pytest
from threadpoolctl import threadpool_limits

from limen.experiment import UniversalExperimentLoop
from limen.experiment._walk_forward_run import WalkForwardRun
from limen.yaml import CompiledSFD
from tests.test_parallel_rounds import make_loop, parallel_config, recorded_source, records


MODES = ('legacy', 'msq', 'parallel')
RECORDING = dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True)


def _configuration(*, walk_forward=False, objective=False):
    config = parallel_config(walk_forward=walk_forward, objective=objective)
    config['sfd']['params']['C'] = [0.1, 0.3]
    config['uel'].pop('pruning_strategies', None)
    config['uel'].update(n_permutations=2, checkpoint_interval=1, feedback_interval=1)
    return config


def _loop(mode, config, directory, *, callback=None):
    if mode != 'legacy':
        return make_loop(config, directory, callback=callback)
    return UniversalExperimentLoop(sfd=CompiledSFD(config), data=recorded_source(),
                                   experiment_dir=directory, yaml_reference=config)


def _run(loop, mode, *, name='results', **kwargs):
    with threadpool_limits(limits=1):
        loop.run(name, n_permutations=2, prep_each_round=True, random_search=False,
                 progress_bar=False, workers=2 if mode == 'parallel' else 1,
                 record_execution=mode != 'legacy', record_model_outputs=mode != 'legacy',
                 recording=RECORDING if mode != 'legacy' else None, **kwargs)


def _snapshot(directory):
    return {file.relative_to(directory): (file.read_bytes(), file.stat().st_mtime_ns)
            for file in directory.rglob('*') if file.is_file()}


def _rows(directory):
    with (directory / 'results.csv').open(newline='') as stream:
        return list(csv.DictReader(stream))


def _assert_message(error, source, names):
    message = str(error.value)
    expected = ', '.join(sorted(names))
    assert f'{source}: {expected}' in message


def _alter_result(original, owner, data, round_params, fault):
    result = original(owner, data, round_params)
    if round_params['C'] == 0.3:
        if fault == 'metric':
            result['C'] = round_params['C']
        else:
            result['strict_mode_error'] = 'caller supplied success status'
    return result


def _faulted_worker(function, parameters, fault):
    original = WalkForwardRun.evaluate

    def evaluate(owner, data, round_params):
        return _alter_result(original, owner, data, round_params, fault)

    with patch.object(WalkForwardRun, 'evaluate', evaluate):
        return function(parameters)


class _CollisionPool(ProcessPoolExecutor):
    fault = 'metric'

    def submit(self, function, parameters):
        return super().submit(_faulted_worker, function, parameters, self.fault)


@contextmanager
def _collision(fault):
    original = WalkForwardRun.evaluate

    def evaluate(owner, data, round_params):
        return _alter_result(original, owner, data, round_params, fault)

    with patch.object(WalkForwardRun, 'evaluate', evaluate), \
            patch.object(_CollisionPool, 'fault', fault), \
            patch('limen.experiment._parallel_rounds.ProcessPoolExecutor', _CollisionPool):
        yield


@pytest.mark.parametrize('mode', MODES)
def test_static_collisions_precede_preparation_and_output(mode, tmp_path, monkeypatch):
    config = _configuration()
    config['sfd']['params'].update(id=['caller identity'], execution_time=[0])
    context = dict.fromkeys(('_convergence_warning', '_dropped_features', '_generation_index',
                             '_id', '_injected', '_param_hash', '_round_index',
                             '_search_strategy', '_warnings', 'strict_mode_error'), 0)
    path = tmp_path / 'output'
    if mode == 'msq':
        path.mkdir()
    loop = _loop(mode, config, None if mode == 'legacy' else path)

    def unexpected(*args, **kwargs):
        raise AssertionError('Static collision reached round execution')

    monkeypatch.setattr('limen.utils.param_space.ParamSpace.generate', unexpected)
    monkeypatch.setattr('limen.experiment.msq.MSQ.__next__', unexpected)
    monkeypatch.setattr('limen.experiment._parallel_rounds.ProcessPoolExecutor', unexpected)
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError) as error:
        _run(loop, mode, name=str(tmp_path / 'standalone'), context_params=context)
    _assert_message(error, 'domain', ('execution_time', 'id'))
    _assert_message(error, 'context', context)
    assert _snapshot(tmp_path) == before
    assert path.exists() is (mode == 'msq')
    assert not (tmp_path / 'standalone.csv').exists()


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('fault', ('metric', 'status'))
def test_later_collision_never_commits_the_affected_trial(mode, fault, tmp_path, monkeypatch):
    path = tmp_path / 'run'
    feedback = []
    loop = _loop(mode, _configuration(walk_forward=True), path,
                 callback=lambda log, msq: feedback.append(log.clone()))
    accepted = []
    accept = WalkForwardRun.accept

    def observe_accept(owner, identity, data):
        accepted.append(str(identity))
        return accept(owner, identity, data)

    monkeypatch.setattr(WalkForwardRun, 'accept', observe_accept)
    with _collision(fault), pytest.raises(ValueError) as error:
        _run(loop, mode)
    if fault == 'metric':
        _assert_message(error, 'round parameters/result', ('C',))
    else:
        _assert_message(error, 'result bookkeeping', ('strict_mode_error',))
    saved = _rows(path)
    assert len(saved) == 1
    assert float(saved[0]['C']) == 0.1
    assert saved[0]['strict_mode_error'] == ''
    identities = {saved[0]['id']}
    assert accepted == [saved[0]['id']]
    assert set(loop.fold_results['id'].cast(pl.String)) == identities
    tracks = pl.read_parquet(path / 'trial_returns.parquet')
    assert set(tracks['trial']) == identities
    if loop.experiment_log is not None:
        assert set(loop.experiment_log['id'].cast(pl.String)) == identities
    if mode != 'legacy':
        assert [row['round_id'] for row in records(path)] == [saved[0]['id']]
        assert feedback
        assert all(log.height == 1 and log['C'].to_list() == [0.1] for log in feedback)
        assert [row['round'] for row in records(path, 'audit.jsonl')] == [0]


@pytest.mark.parametrize('mode', MODES)
def test_walk_forward_context_preserves_trial_identity(mode, tmp_path):
    config = _configuration(walk_forward=True)
    config['sfd']['params']['n_jobs'] = [2]
    context = {'n_jobs': 1, '_note': 'allowed caller context'}
    path = tmp_path / mode
    loop = _loop(mode, config, path)
    _run(loop, mode, context_params=context)
    saved = _rows(path)
    assert len(saved) == 2
    assert {row['n_jobs'] for row in saved} == {'1'}
    assert {row['_note'] for row in saved} == {context['_note']}
    identities = {row['id'] for row in saved}
    assert len(identities) == 2
    assert set(loop.fold_results['id'].cast(pl.String)) == identities
    assert loop.fold_results.group_by('id').agg(pl.col('fold').sort())['fold'].to_list() == [[0, 1], [0, 1]]
    tracks = pl.read_parquet(path / 'trial_returns.parquet')
    assert set(tracks['trial']) == identities
    if mode != 'legacy':
        before = records(path)
        assert {row['round_id'] for row in before} == identities
        resumed = _loop(mode, config, path)
        _run(resumed, mode, context_params=context, resume=True)
        assert _rows(path) == saved
        assert records(path) == before
        assert resumed.fold_results.equals(loop.fold_results)
        assert pl.read_parquet(path / 'trial_returns.parquet').equals(tracks)

import copy
import hashlib
import logging
import multiprocessing
import sqlite3
import time

import pytest

from limen.experiment import _holdout_history as history
from limen.experiment import _holdout_research as research
from limen.experiment.holdout import reserve_holdout, withdraw_holdout
from limen.yaml import CompiledSFD
from limen.yaml.profiler import profile
from tests.test_holdout_research import _configuration, _project, _FIXTURE


def _profile_reservation_worker(root, pipe):
    pipe.send(('ready', history._BUSY_TIMEOUT_SECONDS))
    while pipe.recv() != 'stop':
        started = time.monotonic()
        try:
            identifier = reserve_holdout(root, '2024-01-05', '2024-01-06', interval_seconds=3600)
        except sqlite3.OperationalError as exc:
            pipe.send(('blocked', str(exc), time.monotonic() - started))
        else:
            pipe.send(('reserved', identifier, time.monotonic() - started))
    pipe.close()


@pytest.mark.parametrize(('kind', 'failure'), [('ml', None), ('rule', None), ('ml', 'load'), ('ml', 'model'), ('ml', 'interrupt')])
def test_profile_admission_covers_evaluation(tmp_path, monkeypatch, kind, failure):
    assert hashlib.sha256(_FIXTURE.read_bytes()).hexdigest() == 'a634bbd4f3692fac7bc88512da588e7772f4bd1b749567606d5314e54d0adf72'
    root = _project(tmp_path / 'project')
    config = _configuration(root)
    if kind == 'rule':
        config = copy.deepcopy(config)
        body = config['sfd']['manifest']
        body.update(type='rule_based', indicators=[], reference_architecture='limen.sfd.reference_architecture.rule_based.rule_based',
                    strategy={'conditions': [{'id': 'up', 'type': 'relative', 'column': 'close', 'operator': '>', 'other_column': 'open'}], 'entry': 'up'})
        for key in ('target', 'scaler'):
            body.pop(key)
        config['sfd']['params'] = {'sharpe_std_threshold': [0.5]}
    compiled = CompiledSFD(config, project_root=root)
    manifest = compiled.manifest()
    loader, evaluator = manifest.fetch_data, manifest.run_model
    logger = logging.getLogger('limen.experiment.manifest_core')
    handlers, strict = tuple(logger.handlers), getattr(manifest, 'strict_mode', None)
    fitted = []
    context = multiprocessing.get_context('spawn')
    parent, child = context.Pipe()
    worker = context.Process(target=_profile_reservation_worker, args=(root, child))
    worker.start()
    child.close()

    def refuse_first_reservation():
        with pytest.raises(ValueError, match='Cannot change reservations inside active research'):
            reserve_holdout(root, '2024-01-05', '2024-01-06', interval_seconds=3600)

    def refuse_child_reservation():
        parent.send('reserve')
        assert parent.poll(30)
        status, message, elapsed = parent.recv()
        assert status == 'blocked' and 'locked' in message and elapsed >= 4.5
        assert not (root / 'manifests/holdout.sqlite3').exists()

    def fetch():
        refuse_first_reservation()
        data = loader()
        assert data.height > 0
        if failure == 'load':
            refuse_child_reservation()
            raise OSError('recorded loader failure')
        return data

    def evaluate(data, params):
        result = evaluator(data, params)
        if kind == 'ml':
            assert result['_model'].model.coef_.size > 0
            assert manifest.strict_mode is False
        else:
            assert len(result['_preds']) == data['test'].height > 0
        fitted.append(result)
        refuse_first_reservation()
        refuse_child_reservation()
        if failure == 'model':
            raise RuntimeError('failure after real model fit')
        if failure == 'interrupt':
            raise KeyboardInterrupt('interruption after real model fit')
        return result

    def no_network(*args, **kwargs):
        raise AssertionError('profile proof must use recorded local market data')

    monkeypatch.setattr('requests.sessions.Session.request', no_network)
    monkeypatch.setattr(manifest, 'fetch_data', fetch)
    monkeypatch.setattr(manifest, 'run_model', evaluate)
    try:
        assert parent.poll(30) and parent.recv() == ('ready', 5.0)
        if failure == 'interrupt':
            with pytest.raises(KeyboardInterrupt, match='after real model fit'):
                profile(compiled)
        else:
            result = profile(compiled)
            assert result.sample_permutations_completed == (1 if failure is None else 0)
            assert bool(result.errors) == (failure is not None)
        assert len(fitted) == (0 if failure == 'load' else 1)
        assert tuple(logger.handlers) == handlers and getattr(manifest, 'strict_mode', None) == strict
        parent.send('reserve')
        assert parent.poll(30)
        status, identifier, _ = parent.recv()
        assert status == 'reserved'
    finally:
        parent.send('stop')
        worker.join(15)
        parent.close()
        if worker.is_alive():
            worker.kill()
            worker.join()
            pytest.fail('reservation worker did not stop after profiler cleanup')
        assert worker.exitcode == 0

    protected = CompiledSFD(config, project_root=root)
    protected_manifest = protected.manifest()
    with history.connection(root / 'manifests/holdout.sqlite3') as db:
        before = [tuple(row) for row in db.execute('SELECT * FROM reservations')]
        assert db.execute('SELECT COUNT(*) FROM research').fetchone()[0] == 0
    with monkeypatch.context() as patch:
        for method in ('fetch_data', 'prepare_data', 'run_model'):
            patch.setattr(protected_manifest, method, no_network)
        with pytest.raises(ValueError, match='Holdout research refuses profile'):
            profile(protected)
    with history.connection(root / 'manifests/holdout.sqlite3') as db:
        assert [tuple(row) for row in db.execute('SELECT * FROM reservations')] == before
        assert db.execute('SELECT COUNT(*) FROM research').fetchone()[0] == 0
    withdraw_holdout(root, identifier)

    raced = CompiledSFD(config, project_root=root)
    _ = raced.manifest()
    reject = research.reject_modes
    raced_reservations = []

    def reserve_after_precheck(value, **kwargs):
        reject(value, **kwargs)
        raced_reservations.append(reserve_holdout(root, '2024-01-05', '2024-01-06', interval_seconds=3600))

    with monkeypatch.context() as patch:
        patch.setattr(research, 'reject_modes', reserve_after_precheck)
        patch.setattr(raced.manifest(), 'fetch_data', no_network)
        with pytest.raises(ValueError, match='revision changed'):
            profile(raced)
    assert len(raced_reservations) == 1
    withdraw_holdout(root, raced_reservations[0])

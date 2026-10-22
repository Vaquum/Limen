import copy
import hashlib
import json
import logging
import multiprocessing
import sqlite3
import time

import pytest

from limen.experiment import _holdout_history as history
from limen.experiment import _holdout_research as research
from limen.experiment import _holdout_checkpoint as checkpoint_binding
from limen.experiment import _round_storage
from limen.experiment import UniversalExperimentLoop
from limen.experiment.holdout import reserve_holdout, withdraw_holdout, release_holdout
from limen.yaml import CompiledSFD
from limen.yaml.compiler import build_search_strategy
from limen.inference import Trainer
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


def _profile_admission(tmp_path, monkeypatch, kind, failure, binding):
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
    outside = tmp_path / 'outside'
    outside.mkdir()

    def compile_for_profile():
        with monkeypatch.context() as patch:
            patch.chdir(outside)
            value = CompiledSFD(config, project_root=root if binding == 'explicit' else None)
            cached = value.manifest()
            assert (research.binding_of(cached) is None) == (binding == 'lazy')
        return value

    compiled = compile_for_profile()
    monkeypatch.chdir(root)
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
        assert research.binding_of(manifest).root == root
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

    protected = compile_for_profile()
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

    raced = compile_for_profile()
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


@pytest.mark.parametrize(('kind', 'failure'), [('ml', None), ('rule', None), ('ml', 'load'), ('ml', 'model'), ('ml', 'interrupt')])
def test_profile_admission_covers_evaluation(tmp_path, monkeypatch, kind, failure):
    _profile_admission(tmp_path, monkeypatch, kind, failure, 'explicit')

@pytest.mark.parametrize('post_processing', [True, False])
def test_protected_checkpoint_binding(tmp_path, monkeypatch, post_processing):
    assert hashlib.sha256(_FIXTURE.read_bytes()).hexdigest() == 'a634bbd4f3692fac7bc88512da588e7772f4bd1b749567606d5314e54d0adf72'
    root = _project(tmp_path / 'protected')
    config = _configuration(root)
    config['sfd']['params']['C'] = [0.5, 1.0, 2.0]
    config['uel']['n_permutations'] = 3
    reservation_id = reserve_holdout(root, '2024-01-05', '2024-01-06', interval_seconds=3600)
    directory = root / 'results'
    loop = UniversalExperimentLoop(
        sfd=CompiledSFD(config, project_root=root), project_root=root,
        search_strategy=build_search_strategy(config), yaml_reference=config,
        experiment_dir=directory, checkpoint_interval=1,
    )
    checkpoint_now = loop._checkpoint
    def interrupt(*args, **kwargs):
        checkpoint_now(*args, **kwargs)
        raise RuntimeError('interrupted after saved real rounds')
    monkeypatch.setattr(loop, '_checkpoint', interrupt)
    with pytest.raises(RuntimeError, match='interrupted after saved real rounds'):
        loop.run('protected-checkpoint', n_permutations=3, progress_bar=False, post_processing=post_processing)
    binding = history.bind(root)
    checkpoint = json.loads((directory / 'checkpoint.json').read_text())
    metadata = json.loads((directory / 'metadata.json').read_text())
    header = checkpoint_binding.binding_header(binding)
    assert header == checkpoint['holdout_binding'] == metadata['holdout_binding']
    assert len(checkpoint['holdout_population']['prefix']) == 2
    assert len(checkpoint['holdout_population']['completed']) == 2
    checkpoint_binding.validate_checkpoint(directory, checkpoint, metadata, binding,
                                           vars(loop.manifest)['_holdout_source'])
    with _round_storage.open_text(_round_storage.round_path(directory)) as stream:
        rounds = [json.loads(line) for line in stream]
    assert [item['research_record_id'] for item in checkpoint['holdout_population']['completed']] == [
        item['holdout']['research_record_id'] for item in rounds
    ]
    source = rounds[0]['holdout']['source']
    forged_source = dict(source, admitted_sha256='changed-recorded-frame')
    with pytest.raises(ValueError, match='admitted source changed'):
        checkpoint_binding.validate_checkpoint(directory, checkpoint, metadata, binding, forged_source)
    snapshot = {path.relative_to(directory): (path.read_bytes(), path.stat().st_mtime_ns)
                for path in directory.rglob('*') if path.is_file()}
    for field in ('reservation_revision', 'project_id', 'cutoff_ns', 'interval_seconds', 'schema_version'):
        altered = copy.deepcopy(checkpoint)
        altered['holdout_binding'][field] = -1
        with pytest.raises(ValueError, match='binding/revision'):
            checkpoint_binding.validate_checkpoint(directory, altered, metadata, binding)
    for part in ('holdout_binding', 'holdout_population'):
        altered = copy.deepcopy(checkpoint)
        altered.pop(part)
        with pytest.raises(ValueError, match=r'binding|population'):
            checkpoint_binding.validate_checkpoint(directory, altered, metadata, binding)
    no_envelope = dict(checkpoint)
    no_envelope.pop('holdout_binding')
    no_metadata = dict(metadata)
    no_metadata.pop('holdout_binding')
    with pytest.raises(ValueError, match='original checkpoint population'):
        checkpoint_binding.validate_checkpoint(directory, no_envelope, no_metadata, binding)
    other_root = _project(tmp_path / 'other_project')
    reserve_holdout(other_root, '2024-01-05', '2024-01-06', interval_seconds=3600)
    with pytest.raises(ValueError, match='binding/revision'):
        checkpoint_binding.validate_checkpoint(directory, checkpoint, metadata, history.bind(other_root))
    altered = copy.deepcopy(checkpoint)
    altered['holdout_population']['completed'][0]['research_record_id'] = rounds[1]['holdout']['research_record_id']
    with pytest.raises(ValueError, match='population'):
        checkpoint_binding.validate_checkpoint(directory, altered, metadata, binding)
    forged = dict(rounds[0]['holdout'], research_record_id=rounds[1]['holdout']['research_record_id'])
    with pytest.raises(ValueError, match=r'history|occurrence'):
        checkpoint_binding.validate_trainer_round(binding, rounds[0]['round_id'], forged)
    for key, value in (('selected', {}), ('splits', {}), ('input_schema', None), ('model_schema', None), ('source', {})):
        altered_proof = dict(rounds[0]['holdout'], **{key: value})
        with pytest.raises(ValueError, match=r'source|membership|model-input|history'):
            checkpoint_binding.validate_trainer_round(binding, rounds[0]['round_id'], altered_proof)
    with history.connection(root / 'manifests/holdout.sqlite3') as db:
        db.execute('UPDATE research SET state=? WHERE id=?', ('failed', rounds[0]['holdout']['research_record_id']))
    with pytest.raises(ValueError, match='history'):
        checkpoint_binding.validate_checkpoint(directory, checkpoint, metadata, binding)
    with history.connection(root / 'manifests/holdout.sqlite3') as db:
        db.execute('UPDATE research SET state=? WHERE id=?', ('completed', rounds[0]['holdout']['research_record_id']))
    sensors = Trainer(directory, project_root=root).train([rounds[0]['round_id']])
    assert sensors[0]._fitted_params['_holdout']['experiment_record_id'] == rounds[0]['holdout']['research_record_id']
    assert {path.relative_to(directory): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob('*') if path.is_file()} == snapshot
    release_holdout(root, reservation_id)
    with pytest.raises(ValueError, match='binding/revision'):
        Trainer(directory, project_root=root)

    zero_root = _project(tmp_path / 'zero_active')
    zero_config = _configuration(zero_root)
    old_reservation = reserve_holdout(zero_root, '2024-01-05', '2024-01-06', interval_seconds=3600)
    withdraw_holdout(zero_root, old_reservation)
    zero_dir = zero_root / 'results'
    zero_loop = UniversalExperimentLoop(
        sfd=CompiledSFD(zero_config, project_root=zero_root), project_root=zero_root,
        search_strategy=build_search_strategy(zero_config), yaml_reference=zero_config,
        experiment_dir=zero_dir, checkpoint_interval=1,
    )
    zero_loop.run('zero-active', n_permutations=1, progress_bar=False, post_processing=False)
    zero_metadata = json.loads((zero_dir / 'metadata.json').read_text())
    zero_header = zero_metadata['holdout_binding']
    assert zero_header['cutoff_ns'] is None
    zero_checkpoint = checkpoint_binding.checkpoint_state(zero_dir, 0, history.bind(zero_root))
    assert zero_checkpoint['binding'] == zero_header
    reserve_holdout(zero_root, '2024-01-05', '2024-01-06', interval_seconds=3600)
    with pytest.raises(ValueError, match='binding/revision'):
        checkpoint_binding.validate_checkpoint(
            zero_dir,
            {'metadata': {'experiment_round': 0}, 'holdout_binding': zero_header,
             'holdout_population': zero_checkpoint['population']},
            zero_metadata, history.bind(zero_root),
        )
    with pytest.raises(ValueError, match='original saved project/population evidence'):
        Trainer(zero_dir, project_root=zero_root)

    ordinary_root = _project(tmp_path / 'ordinary')
    ordinary_config = _configuration(ordinary_root)
    ordinary_dir = ordinary_root / 'results'
    ordinary = UniversalExperimentLoop(
        sfd=CompiledSFD(ordinary_config, project_root=ordinary_root), project_root=ordinary_root,
        search_strategy=build_search_strategy(ordinary_config), yaml_reference=ordinary_config,
        experiment_dir=ordinary_dir, checkpoint_interval=1,
    )
    ordinary.run('ordinary', n_permutations=1, progress_bar=False, post_processing=False)
    with _round_storage.open_text(_round_storage.round_path(ordinary_dir)) as stream:
        ordinary_round = json.loads(next(stream))
    assert Trainer(ordinary_dir, project_root=ordinary_root).train([ordinary_round['round_id']])
    reserve_holdout(ordinary_root, '2024-01-05', '2024-01-06', interval_seconds=3600)
    with pytest.raises(ValueError, match='original saved project/population evidence'):
        Trainer(ordinary_dir, project_root=ordinary_root)

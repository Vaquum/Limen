import copy
import hashlib
import gzip
import json
import logging
import multiprocessing
import sqlite3
import time

import polars as pl
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


def _profile_reservation_worker(root, pipe, busy_timeout):
    pipe.send(('ready', history._BUSY_TIMEOUT_SECONDS))
    history._BUSY_TIMEOUT_SECONDS = busy_timeout
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
    # Prove the production wait once; the other paths prove the same lock and cleanup.
    busy_timeout = history._BUSY_TIMEOUT_SECONDS if (kind, failure, binding) == ('ml', None, 'explicit') else 0.2
    worker = context.Process(target=_profile_reservation_worker, args=(root, child, busy_timeout))
    worker.start()
    child.close()

    def refuse_first_reservation():
        with pytest.raises(ValueError, match='Cannot change reservations inside active research'):
            reserve_holdout(root, '2024-01-05', '2024-01-06', interval_seconds=3600)

    def refuse_child_reservation():
        parent.send('reserve')
        assert parent.poll(30)
        status, message, elapsed = parent.recv()
        assert status == 'blocked' and 'locked' in message and elapsed >= busy_timeout * 0.9
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
    config['uel']['recording'] = dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True)
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
    _checkpoint_boundaries(loop, config, checkpoint, metadata, monkeypatch, post_processing)
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


def _artifact_state(directory):
    return {path.relative_to(directory): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob('*') if path.is_file()}


@pytest.mark.parametrize('protected', [True, False])
@pytest.mark.parametrize('all_failed', [True, False])
def test_checkpoint_leading_failed_population(tmp_path, monkeypatch, protected, all_failed):
    from limen.experiment.errors import StrictModeError
    root = _project(tmp_path / 'failures')
    config = _configuration(root)
    config['sfd']['params']['C'] = [0.5, 1.0, 2.0]
    if protected:
        reserve_holdout(root, '2024-01-05', '2024-01-06', interval_seconds=3600)
    compiled = CompiledSFD(config, project_root=root)
    seen = []
    def select(frame):
        seen.append(frame.height)
        if len(seen) <= 2:
            raise StrictModeError('recorded-data candidate rejected')
        return frame
    compiled.manifest().set_pre_split_data_selector(select)
    directory = root / 'results'
    loop = UniversalExperimentLoop(sfd=compiled, project_root=root, experiment_dir=directory,
        search_strategy=build_search_strategy(config), yaml_reference=config, checkpoint_interval=1)
    first = []
    save = loop._checkpoint
    def capture(*args, **kwargs):
        save(*args, **kwargs)
        checkpoint = json.loads((directory / 'checkpoint.json').read_text())
        first.append(checkpoint)
        if len(first) == 1:
            assert len(checkpoint['holdout_population']['prefix']) == 2
            assert checkpoint['holdout_population']['completed'] == []
            assert not _round_storage.round_path(directory).exists()
    monkeypatch.setattr(loop, '_checkpoint', capture)
    loop.run('leading-failures', n_permutations=2 if all_failed else 3, progress_bar=False, post_processing=False)
    assert first
    assert len(seen) == (2 if all_failed else 3)
    assert loop.experiment_log.height == (2 if all_failed else 3)
    assert loop.experiment_log['strict_mode_error'].null_count() == (0 if all_failed else 1)
    metadata = json.loads((directory / 'metadata.json').read_text())
    checkpoint_binding.validate_checkpoint(directory, first[-1], metadata, history.bind(root), vars(loop.manifest).get('_holdout_source'))
    if not all_failed:
        checkpoint_binding.validate_checkpoint(directory, first[0], metadata, history.bind(root), vars(loop.manifest).get('_holdout_source'))
        assert first[-1]['holdout_population']['completed'][0]['index'] == 2
        assert set(pl.read_csv(directory / 'results.csv').columns) == set(loop.experiment_log.columns)


def test_protected_trainer_split_preflight(tmp_path, monkeypatch):
    from limen.targets.quantile_binary import QuantileBinaryTarget
    from limen.scalers.logreg_scaler import LogRegScaler
    root = _project(tmp_path / 'splits')
    config = _configuration(root)
    reserve_holdout(root, '2024-01-05', '2024-01-06', interval_seconds=3600)
    directory = root / 'results'
    loop = UniversalExperimentLoop(sfd=CompiledSFD(config, project_root=root), project_root=root,
        search_strategy=build_search_strategy(config), yaml_reference=config, experiment_dir=directory)
    loop.run('recorded', n_permutations=1, progress_bar=False, post_processing=False)
    with _round_storage.open_text(_round_storage.round_path(directory)) as stream:
        entry = json.loads(next(stream))
    trainer = Trainer(directory, project_root=root)
    original = (directory / 'metadata.json').read_bytes()
    with monkeypatch.context() as guard:
        def no_fit(*args, **kwargs):
            pytest.fail('population refusal must precede fitting')
        guard.setattr(QuantileBinaryTarget, '__init__', no_fit)
        guard.setattr(LogRegScaler, '__init__', no_fit)
        dates = list(trainer._manifest.split_dates)
        from datetime import date
        dates[1] = date(2024, 1, 2)
        trainer._manifest.set_split_dates(*dates)
        before = _artifact_state(directory)
        with pytest.raises(ValueError, match='selection/split geometry'):
            trainer.train([entry['round_id']])
        assert _artifact_state(directory) == before
        metadata = json.loads(original)
        metadata['yaml_reference']['sfd']['manifest']['split_dates']['train_end'] = '2024-01-02'
        (directory / 'metadata.json').write_text(json.dumps(metadata))
        before = _artifact_state(directory)
        with pytest.raises(ValueError, match='selection/split geometry'):
            Trainer(directory, project_root=root)
        assert _artifact_state(directory) == before
    (directory / 'metadata.json').write_bytes(original)
    assert Trainer(directory, project_root=root).train([entry['round_id']])


def test_zero_active_checkpoint_partial_selector(tmp_path):
    from functools import partial
    root = _project(tmp_path / 'zero-partial')
    config = _configuration(root)
    config['sfd']['params']['C'] = [0.5, 1.0]
    compiled = CompiledSFD(config, project_root=root)
    compiled.manifest().set_pre_split_data_selector(partial(pl.DataFrame.slice, offset=1))
    directory = root / 'results'
    loop = UniversalExperimentLoop(sfd=compiled, project_root=root, experiment_dir=directory,
        search_strategy=build_search_strategy(config), yaml_reference=config, checkpoint_interval=1)
    loop.run('partial-selector', n_permutations=2, progress_bar=False, post_processing=False)
    checkpoint = json.loads((directory / 'checkpoint.json').read_text())
    metadata = json.loads((directory / 'metadata.json').read_text())
    assert checkpoint['holdout_binding']['cutoff_ns'] is None
    assert checkpoint['holdout_binding']['project_id'] is None
    before = _artifact_state(directory)
    checkpoint_binding.validate_checkpoint(directory, checkpoint, metadata, history.bind(root))
    assert _artifact_state(directory) == before


def _checkpoint_boundaries(loop, config, checkpoint, metadata, monkeypatch, retain):
    from datetime import date
    from sklearn.linear_model import LogisticRegression
    from limen.targets.quantile_binary import QuantileBinaryTarget
    from limen.scalers.logreg_scaler import LogRegScaler
    from limen.experiment._parallel_resume import restore_parallel
    directory = loop._experiment_dir
    root = research.binding_of(loop.manifest).root
    restored = UniversalExperimentLoop(sfd=CompiledSFD(config, project_root=root), project_root=root,
        search_strategy=build_search_strategy(config), yaml_reference=config, experiment_dir=directory,
        checkpoint_interval=1)
    for name in ('_record_execution', '_record_model_outputs', '_recording', '_factorize'):
        setattr(restored, name, getattr(loop, name))
    parts = restored._setup_msq_components(experiment_name='protected-checkpoint', n_permutations=3)
    common = {key: parts[key] for key in ('content_hash', 'strategy_type', 'csv_path', 'round_data_path')}
    def restore(kind):
        args = (parts['msq'], parts['domain'], parts['feedback_controller'], parts['checkpoint_manager'])
        if kind == 'serial':
            return restored._restore_checkpoint_state(*args, **common, retain_round_artifacts=retain, context_params=None)
        return restore_parallel(restored, *args, **common, failed_round_params=[], n_permutations=3)
    assert restore('serial') == 2
    assert restored.experiment_log.height == 2
    if retain:
        assert [row['holdout']['research_record_id'] for row in restored._alignment] == [
            row['research_record_id'] for row in checkpoint['holdout_population']['completed']]
    else:
        assert restored._alignment == []
    path = _round_storage.round_path(directory)
    round_bytes = path.read_bytes()
    checkpoint_bytes = (directory / 'checkpoint.json').read_bytes()
    metadata_bytes = (directory / 'metadata.json').read_bytes()
    def refused(pattern):
        before = _artifact_state(directory)
        for kind in ('serial', 'parallel'):
            with pytest.raises(ValueError, match=pattern):
                restore(kind)
        assert _artifact_state(directory) == before
    with monkeypatch.context() as guards:
        def no_state(*args, **kwargs):
            pytest.fail('holdout refusal must precede every state setter or artifact rewrite')
        for owner, name in ((parts['domain'], 'set_state'), (parts['msq'], 'set_state'),
                            (parts['feedback_controller'], 'set_state'), (restored._search_strategy, 'rebuild_seen_from_log'),
                            (restored, '_truncate_round_data'), (restored, '_load_round_data')):
            guards.setattr(owner, name, no_state)
        for factory, method in ((QuantileBinaryTarget, '__init__'), (LogRegScaler, '__init__'), (LogisticRegression, 'fit')):
            guards.setattr(factory, method, no_state)
        for key in ('holdout_binding', 'holdout_population'):
            corrupted = copy.deepcopy(checkpoint)
            corrupted.pop(key)
            (directory / 'checkpoint.json').write_text(json.dumps(corrupted))
            refused('binding|population')
        (directory / 'checkpoint.json').write_bytes(checkpoint_bytes)
        encoded = [json.loads(line) for line in gzip.decompress(round_bytes).splitlines()]
        for key in ('holdout', 'source', 'selected', 'splits', 'input_schema', 'model_schema'):
            corrupted = copy.deepcopy(encoded)
            if key == 'holdout':
                corrupted[0].pop(key)
            else:
                corrupted[0]['holdout'].pop(key)
            path.write_bytes(b''.join(gzip.compress((json.dumps(row) + '\n').encode()) for row in corrupted))
            refused('evidence|source|membership|history')
            before = _artifact_state(directory)
            with pytest.raises(ValueError, match=r'evidence|source|membership|history'):
                Trainer(directory, project_root=root).train([encoded[0]['round_id']])
            assert _artifact_state(directory) == before
        path.write_bytes(round_bytes)
        saved_data = restored.data
        restored.data = saved_data.drop('volume')
        refused('admitted source')
        before = _artifact_state(directory)
        with pytest.raises(ValueError, match='admitted source'):
            Trainer(directory, data=restored.data, project_root=root).train([encoded[0]['round_id']])
        assert _artifact_state(directory) == before
        restored.data = saved_data
        dates = restored.manifest.split_dates
        restored.manifest.set_split_dates(dates[0], date(2024, 1, 2), *dates[2:])
        refused('selection/split geometry')
        restored.manifest.set_split_dates(*dates)
        before = _artifact_state(directory)
        with pytest.raises(ValueError, match='Holdout'):
            restored.run('protected-checkpoint', n_permutations=3, resume=True, progress_bar=False)
        assert _artifact_state(directory) == before
    # Existing protected V1 selected artifacts predate the checkpoint envelope.
    legacy = copy.deepcopy(metadata)
    legacy.pop('holdout_binding')
    (directory / 'metadata.json').write_text(json.dumps(legacy))
    ids = [row['round_id'] for row in checkpoint['holdout_population']['completed']]
    before = _artifact_state(directory)
    sensors = Trainer(directory, project_root=root).train(ids[:1])
    assert sensors[0]._fitted_params['_holdout']['experiment_record_id'] == checkpoint['holdout_population']['completed'][0]['research_record_id']
    assert sensors[0]._fitted_params['_holdout']['research_record_id'] != sensors[0]._fitted_params['_holdout']['experiment_record_id']
    assert _artifact_state(directory) == before
    (directory / 'metadata.json').write_bytes(metadata_bytes)

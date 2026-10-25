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


@pytest.mark.parametrize('kind', ['ml', 'rule'])
def test_protected_factorized_provenance(kind, tmp_path, monkeypatch):
    from itertools import product
    from types import SimpleNamespace
    import numpy as np
    from limen.experiment import UniversalExperimentLoop
    from limen.experiment import _holdout_research as research
    from limen.experiment._factorize import FactorizedRounds
    from limen.experiment.param_domain import ParamDomain
    from limen.experiment.param_search import GridStrategy
    from limen.sfd.reference_architecture.rule_based import RuleBasedStrategy
    from tests.test_factorize_equivalence import _numeric_equal
    from tests.test_factorized_preparation import _config, _instrument
    from tests.test_holdout_research import _project

    config = _config()
    if kind == 'rule':
        body = config['sfd']['manifest']
        body.update(type='rule_based', indicators=[], features=[],
                    reference_architecture='limen.sfd.reference_architecture.rule_based.rule_based',
                    strategy={'conditions': [{'id': 'up', 'type': 'relative', 'column': 'close',
                                             'operator': '>', 'other_column': 'open'}], 'entry': 'up'})
        for name in ('target', 'scaler'):
            body.pop(name)
        config['sfd']['params'] = {'fee': [4., 9.]}
    root = _project(tmp_path / 'protected')
    reservation = reserve_holdout(root, '2024-01-07', '2024-01-09', interval_seconds=3600)
    manifest = CompiledSFD(config, project_root=root).manifest()
    raw = manifest.fetch_data()
    domain = config['sfd']['params']
    candidates = [dict(zip(domain, values, strict=True)) for values in product(*domain.values())]
    control = CompiledSFD(copy.deepcopy(config), project_root=_project(tmp_path / 'ordinary')).manifest()
    expected = []
    for params in candidates:
        prepared = control.prepare_data(raw.clone(), params.copy())
        prepared.update(_record_execution=True, _record_model_outputs=True)
        expected.append((prepared, control.run_model(prepared, params.copy())))
    counts = _instrument(monkeypatch)
    rule_fits = []
    original = RuleBasedStrategy.train
    def train(*args, **kwargs):
        rule_fits.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(RuleBasedStrategy, 'train', train)
    strategy = GridStrategy(ParamDomain(domain))
    rounds = FactorizedRounds(manifest=manifest, strategy=strategy, domain=domain, pruning=False,
        callback=False, context=None, prep=manifest.prepare_data, model=manifest.run_model,
        data=raw, record_execution=True, record_model_outputs=True)
    directory = root / 'results'
    directory.mkdir()
    path = directory / 'round_data.jsonl'
    writer = SimpleNamespace(manifest=manifest, _walk_forward=None, _recording=None,
        _yaml_reference=config, _record_execution=True, _record_model_outputs=True)
    preparations, values = [], []
    for index, params in enumerate(candidates):
        with research.operation(manifest, 'experiment'):
            data, result, _, _ = rounds.evaluate(params.copy())
            baseline_data, baseline_result = expected[index]
            _numeric_equal({name: value for name, value in baseline_result.items() if not name.startswith('_')},
                           {name: value for name, value in result.items() if not name.startswith('_')})
            np.testing.assert_array_equal(result['_preds'], baseline_result['_preds'])
            if kind == 'ml':
                np.testing.assert_array_equal(data['_alignment']['model_outputs']['probs'], baseline_data['_alignment']['model_outputs']['probs'])
                if 'x_train' in data:
                    for split in ('train', 'val', 'test'):
                        np.testing.assert_array_equal(data['x_' + split], baseline_data['x_' + split])
                        np.testing.assert_array_equal(data['y_' + split], baseline_data['y_' + split])
                    preparations.append(data['_alignment']['holdout'])
            UniversalExperimentLoop._append_round_data(writer, path, strategy.compute_param_hash(params),
                params, result['_preds'], data['_alignment'], round_index=index)
            values.append(result['backtest_pnl_per_bar_bps'] if kind == 'ml' else result['pnl_per_bar_bps_test'])
    assert len(set(values)) > 1
    if kind == 'ml':
        assert counts == {'target': 1, 'scaler': 1, 'model': 2, 'prepare': 2}
        assert preparations[1]['reuse'] == {'kind': 'fitted',
            'source_preparation_record_id': preparations[0]['research_record_id']}
    else:
        assert len(rule_fits) == 1
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len({row['round_id'] for row in rows}) == len(candidates)
    assert len({row['holdout']['research_record_id'] for row in rows}) == len(candidates)
    binding = history.bind(root)
    for row in rows:
        proof = row['holdout']
        record = history.research_record(binding, proof['research_record_id'])
        assert record['state'] == 'completed' and record['round_id'] == row['round_id']
        assert json.loads(record['provenance_json']) == proof
        if 'reuse' in proof:
            source = history.research_record(binding, proof['reuse']['source_preparation_record_id'])
            assert source['state'] == 'completed'
    retained = path.read_bytes()
    original_counts = counts.copy()
    with research.operation(manifest, 'experiment'):
        if rounds.preparation is not None:
            with pytest.raises(ValueError, match='source population changed'):
                rounds.preparation.prepare(raw.slice(1), candidates[0].copy(), rounds.prep)
        rounds.data = raw.slice(1)
        with pytest.raises(ValueError, match='source population changed'):
            rounds.evaluate(candidates[0].copy())
        rounds.data = raw
    saved_binding = vars(manifest).pop('_holdout_binding')
    try:
        if rounds.preparation is not None:
            with pytest.raises(ValueError, match='cannot become unprotected'):
                rounds.preparation.prepare(raw, candidates[0].copy(), rounds.prep)
        with pytest.raises(ValueError, match='binding changed'):
            rounds.evaluate(candidates[0].copy())
    finally:
        vars(manifest)['_holdout_binding'] = saved_binding
    release_holdout(root, reservation)
    if rounds.preparation is not None:
        with pytest.raises(ValueError, match='revision changed'):
            rounds.preparation.prepare(raw, candidates[0].copy(), rounds.prep)
    with pytest.raises(ValueError, match='revision changed'):
        rounds.evaluate(candidates[0].copy())
    assert counts == original_counts
    assert path.read_bytes() == retained


def _protected_worker_init(config, snapshot, authority, cwd):
    import os
    from limen.experiment._parallel_worker import worker_init
    os.chdir(cwd)
    worker_init(config, snapshot, True, True, authority=authority)


def _protected_rejected_worker(config, snapshot, authority, cwd, occurrence):
    from unittest.mock import patch
    from limen.experiment import MLManifest
    from limen.experiment._parallel_worker import evaluate_round
    _protected_worker_init(config, snapshot, authority, cwd)
    def prepare(*_args, **_kwargs):
        raise AssertionError('Invalid worker authority reached preparation')
    with patch.object(MLManifest, 'prepare_data', prepare):
        return evaluate_round(occurrence)


def _protected_strict_worker(config, snapshot, authority, cwd, occurrence):
    from unittest.mock import patch
    from limen.experiment._parallel_worker import evaluate_round
    from limen.experiment.errors import StrictModeError
    from limen.sfd.reference_architecture.lightgbm_binary import LightGBMBinary
    _protected_worker_init(config, snapshot, authority, cwd)
    original = LightGBMBinary.train
    def train(*args, **kwargs):
        original(*args, **kwargs)
        raise StrictModeError('Recorded worker failure control after genuine fitting')
    with patch.object(LightGBMBinary, 'train', train):
        return evaluate_round(occurrence)


@pytest.mark.parametrize('cwd_mode', ['outside', 'foreign'])
def test_protected_worker_handoff(cwd_mode, tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    from dataclasses import replace
    from functools import partial
    from types import SimpleNamespace
    from limen.experiment import _holdout_composition as protection
    from limen.experiment._parallel_commit import ParallelCommit
    from limen.experiment._parallel_sources import snapshot_sources
    from limen.experiment._parallel_worker import evaluate_round, startup_limits
    from limen.experiment.param_domain import ParamDomain
    from limen.experiment.param_search import GridStrategy
    from tests.test_factorized_preparation import _config, _params
    from tests.test_factorize_equivalence import _numeric_equal

    config = _config()
    root = _project(tmp_path / 'protected-worker')
    reserve_holdout(root, '2024-01-07', '2024-01-09', interval_seconds=3600)
    manifest = CompiledSFD(config, project_root=root).manifest()
    raw = manifest.fetch_data()
    cwd = _project(tmp_path / 'foreign') if cwd_mode == 'foreign' else tmp_path
    control = CompiledSFD(copy.deepcopy(config), project_root=_project(tmp_path / 'control')).manifest()
    params = _params(config)
    original = control.prepare_data(raw.clone(), params.copy())
    original.update(_record_execution=True, _record_model_outputs=True)
    expected = control.run_model(original, params.copy())
    identity = GridStrategy(ParamDomain(config['sfd']['params'])).compute_param_hash(params)
    occurrence = params | {'_round_index': 0, '_id': identity}
    directory = root / 'results'
    directory.mkdir()
    writer = SimpleNamespace(manifest=manifest, _walk_forward=None, _recording=None,
        _yaml_reference=config, _record_execution=True, _record_model_outputs=True)
    host = SimpleNamespace(manifest=manifest, data=raw, params=params, _walk_forward=None,
        _append_round_data=partial(UniversalExperimentLoop._append_round_data, writer), experiment_log=None)
    with research.operation(manifest, 'experiment') as operation:
        authority = protection.worker_authority(manifest)
        with startup_limits(), snapshot_sources(raw, None, None, tmp_path) as snapshot, ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context('spawn'), initializer=_protected_worker_init,
                initargs=(config, snapshot, authority, str(cwd))) as pool:
            payload = pool.submit(evaluate_round, occurrence).result(timeout=40)
            binding, source, parent_id = authority
            for refused in (
                (replace(binding, root=tmp_path), source, parent_id),
                (replace(binding, revision=binding.revision + 1), source, parent_id),
                (binding, source | {'admitted_sha256': '0' * 64}, parent_id),
                (binding, source, None),
            ):
                with pytest.raises(RuntimeError, match=r'revision|coordinator|source'):
                    pool.submit(_protected_rejected_worker, config, snapshot, refused, str(cwd), occurrence).result(timeout=40)
            failed_occurrence = occurrence | {'_id': identity + '-failed', '_round_index': 1}
            failed = pool.submit(_protected_strict_worker, config, snapshot, authority, str(cwd), failed_occurrence).result(timeout=40)
        _numeric_equal({key: value for key, value in expected.items() if not key.startswith('_')},
                       {key: value for key, value in payload.results.items() if not key.startswith('_')})
        assert payload.results['_preds'].tolist() == expected['_preds'].tolist()
        assert payload.alignment['model_outputs'] == original['_alignment']['model_outputs']
        proof = payload.alignment['holdout']
        assert proof['project_id'] == authority[0].project_id
        parent = history.research_record(authority[0], proof['experiment_record_id'])
        assert parent['state'] == 'completed'
        assert json.loads(parent['provenance_json'])['experiment_record_id'] == operation.record_id
        commit = ParallelCommit(host, directory / 'results.csv', directory / 'round_data.jsonl', [])
        invalid = [
            {key: value for key, value in payload.alignment.items() if key != 'holdout'},
            payload.alignment | {'holdout': proof | {'project_id': history.bind(_project(tmp_path / 'other')).project_id}},
            payload.alignment | {'holdout': proof | {'round_id': 'foreign-round'}},
        ]
        for alignment in invalid:
            before = _artifact_state(directory)
            with pytest.raises(ValueError, match=r'holdout|protection|Factorized|history'):
                commit.commit(occurrence, replace(payload, alignment=alignment,
                    results=payload.results | {'must_not_extend_header': 1.}))
            assert _artifact_state(directory) == before
        commit.commit(occurrence, payload)
        prefix = _artifact_state(directory)
        with pytest.raises(ValueError, match='occurrence'):
            commit.commit(occurrence | {'_id': 'foreign-round', '_round_index': 1}, payload)
        assert _artifact_state(directory) == prefix
        assert not failed.succeeded
        failure_proof = failed.alignment['holdout_failure']
        assert history.research_record(authority[0], failure_proof['research_record_id'])['state'] == 'failed'
        retained = (directory / 'round_data.jsonl').read_bytes()
        commit.commit(failed_occurrence, failed)
        assert host.experiment_log.height == 2 and len(commit.failed_round_params) == 1
        assert (directory / 'round_data.jsonl').read_bytes() == retained
    row = json.loads((directory / 'round_data.jsonl').read_text())
    assert row['round_id'] == identity
    assert row['holdout']['research_record_id'] != proof['research_record_id']


def _protected_orphan_worker(config, snapshot, authority, pipe):
    import os
    from unittest.mock import patch
    from limen.experiment._parallel_worker import worker_init, evaluate_round
    from limen.sfd.reference_architecture.lightgbm_binary import LightGBMBinary
    from tests.test_factorized_preparation import _params

    worker_init(config, snapshot, False, False, authority=authority)
    original = LightGBMBinary.train
    def train(*args, **kwargs):
        result = original(*args, **kwargs)
        pipe.send(('model', os.getpid()))
        assert pipe.recv() == 'finish'
        return result
    with patch.object(LightGBMBinary, 'train', train):
        evaluate_round(_params(config) | {'_id': 'orphan', '_round_index': 0})


def _protected_worker_coordinator(root, config, pipe):
    from pathlib import Path
    from limen.experiment import _holdout_composition as protection
    from limen.experiment._parallel_sources import snapshot_sources
    from limen.experiment._parallel_worker import startup_limits

    manifest = CompiledSFD(config, project_root=root).manifest()
    raw = manifest.fetch_data()
    with research.operation(manifest, 'experiment'), startup_limits(), snapshot_sources(raw, None, None, Path(root)) as snapshot:
        worker = multiprocessing.get_context('spawn').Process(target=_protected_orphan_worker,
            args=(config, snapshot, protection.worker_authority(manifest), pipe))
        worker.start()
        pipe.send(('worker', worker.pid))
        worker.join()


@pytest.mark.parametrize('active', [True, False])
def test_protected_worker_parent_death(active, tmp_path, monkeypatch):
    import os
    import signal
    from tests.test_factorized_preparation import _config

    root = _project(tmp_path / 'orphan-project')
    if active:
        reserve_holdout(root, '2024-01-07', '2024-01-09', interval_seconds=3600)
    context = multiprocessing.get_context('spawn')
    receiver, sender = context.Pipe()
    coordinator = context.Process(target=_protected_worker_coordinator, args=(str(root), _config(), sender))
    worker_pid = None
    coordinator.start()
    try:
        for _ in range(2):
            assert receiver.poll(40), 'The spawned worker did not reach real model evaluation'
            state, pid = receiver.recv()
            worker_pid = pid
            if state == 'model':
                break
        assert state == 'model'
        coordinator.kill()
        coordinator.join(10)
        assert not coordinator.is_alive()
        monkeypatch.setattr(history, '_BUSY_TIMEOUT_SECONDS', 0.15)
        start, end = ('2024-01-09', '2024-01-10') if active else ('2024-01-07', '2024-01-09')
        with pytest.raises(sqlite3.OperationalError, match='locked'):
            reserve_holdout(root, start, end, interval_seconds=3600)
        os.kill(worker_pid, signal.SIGKILL)
        worker_pid = None
        assert reserve_holdout(root, start, end, interval_seconds=3600)
        assert not (root / 'results').exists()
        with history.connection(root / 'manifests/holdout.sqlite3') as db:
            rows = db.execute('SELECT state FROM research').fetchall()
        if active:
            assert sum(row['state'] == 'started' for row in rows) >= 2
        else:
            assert not rows
    finally:
        if coordinator.is_alive():
            coordinator.kill()
            coordinator.join(10)
        if worker_pid is not None:
            os.kill(worker_pid, signal.SIGKILL)
        receiver.close()
        sender.close()


@pytest.mark.parametrize('population', ['full', 'gapped_selected_prefix'])
def test_protected_fold_lineage(population, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from time import perf_counter
    from limen.experiment.param_domain import ParamDomain
    from limen.experiment.param_search import GridStrategy
    from limen.experiment._walk_forward_run import WalkForwardRun
    from limen.log.log import Log
    from tests.test_factorize_equivalence import _numeric_equal
    from tests.test_factorized_preparation import _config, _params

    root = _project(tmp_path / 'fold-project')
    reservation = reserve_holdout(root, '2024-01-07', '2024-01-09', interval_seconds=3600)
    config = _config()
    body = config['sfd']['manifest']
    body.pop('split_dates')
    body['split_walk_forward'] = {'n_folds': 2, 'test_bars': 12, 'purge_bars': 1, 'embargo_bars': 1, 'anchored': True}
    if population == 'gapped_selected_prefix':
        body['pre_split_data_selector'] = {'func': 'limen.data.utils.random_slice', 'params': {'rows': 96, 'safe_range_low': 0., 'safe_range_high': 0.9, 'seed': 42}}
    compiled = CompiledSFD(config, project_root=root)
    manifest = compiled.manifest()
    market = pl.read_parquet(_FIXTURE).head(192)
    if population == 'gapped_selected_prefix':
        manifest.set_split_config(4, 1, 2)
    full = market
    if population == 'gapped_selected_prefix':
        full = full.head(132).filter(pl.col('datetime') != full['datetime'][50])
    runner = WalkForwardRun(manifest, full, root / 'results', promotion=True)
    if population == 'full':
        assert runner.raw.height == 144
        assert runner.promotion['selection_cutoff_ns'] == history.utc_day('2024-01-07')
    else:
        assert runner.raw.height == 131
        assert runner.promotion['selection_cutoff_ns'] < history.utc_day('2024-01-07')
    with pytest.raises(ValueError, match='Holdout research refuses'):
        UniversalExperimentLoop(sfd=CompiledSFD(config, project_root=root), project_root=root)
    params = _params(config)
    control = CompiledSFD(copy.deepcopy(config), project_root=_project(tmp_path / 'ordinary-fold')).manifest()
    if population == 'gapped_selected_prefix':
        control.set_split_config(4, 1, 2)
    ordinary = WalkForwardRun(control, runner.raw.clone(), tmp_path / 'control-output', promotion=True)
    baseline_data = ordinary.prepare(runner.raw.clone(), params)
    baseline_data.update(_record_execution=True, _record_model_outputs=True)
    baseline = ordinary.evaluate(baseline_data, params.copy())
    directory = root / 'results'
    directory.mkdir()
    path = directory / 'round_data.jsonl.gz'
    strategy = GridStrategy(ParamDomain(config['sfd']['params']))
    identity = strategy.compute_param_hash(params)
    writer = SimpleNamespace(manifest=manifest, _walk_forward=runner, _recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True),
        _yaml_reference=config, _record_execution=True, _record_model_outputs=True, _workers=1, _factorize=False,
        _search_strategy=strategy, _sfd_module_name=compiled.__name__, _get_limen_version=UniversalExperimentLoop._get_limen_version)
    with pytest.raises(RuntimeError, match='interruption control'), research.operation(manifest, 'experiment') as campaign:
        UniversalExperimentLoop._write_metadata(writer, directory)
        started = perf_counter()
        data = runner.prepare(full, params.copy())
        data.update(_record_execution=True, _record_model_outputs=True)
        result = runner.evaluate(data, params.copy())
        elapsed = perf_counter() - started
        _numeric_equal(baseline, result)
        runner.accept(identity, data)
        UniversalExperimentLoop._append_round_data(writer, path, identity, params, [], data['_alignment'], round_index=0)
        pl.DataFrame([params | result | {'id': identity, '_round_index': 0, 'strict_mode_error': None, 'execution_time': elapsed}]).write_csv(directory / 'results.csv')
        runner.finish()
        raise RuntimeError('interruption control after a genuinely committed trial')
    assert history.research_record(history.bind(root), campaign.record_id)['state'] == 'failed'
    with _round_storage.open_text(path) as stream:
        records = [json.loads(line) for line in stream]
    assert len(records) == 1
    recorded = records[0]
    assert len(recorded['holdout']['workflow']['folds']) == 2
    if population == 'gapped_selected_prefix':
        proof = recorded['holdout']
        assert proof['source']['admitted']['last_close_ns'] < proof['cutoff_ns']
        assert proof['selected']['last_close_ns'] < proof['source']['admitted']['last_close_ns']
        assert proof['selected']['gaps']
    selected_fold = recorded['folds'][0]['holdout']
    sensor = Trainer(directory, data=runner.raw, project_root=root).train([identity], fold=0)[0]
    fitted = sensor._fitted_params['_holdout']
    assert fitted['experiment_record_id'] == selected_fold['research_record_id']
    assert fitted['workflow']['fold'] == 0 and fitted['round_id'] == identity
    assert history.research_record(history.bind(root), fitted['research_record_id'])['state'] == 'completed'
    restored_manifest = CompiledSFD(config, project_root=root).manifest()
    if population == 'gapped_selected_prefix':
        restored_manifest.set_split_config(4, 1, 2)
    restored = WalkForwardRun(restored_manifest, runner.raw, directory, promotion=True)
    with research.operation(restored_manifest, 'experiment'):
        assert restored.restore(path, None, [(0, identity)]) == 1
    captured = Log(uel_object=SimpleNamespace(experiment_log=pl.read_csv(directory / 'results.csv'), fold_results=restored.results, manifest=manifest, data=runner.raw,
        _alignment=[data['_alignment']], round_params=[params]))
    assert captured.experiment_backtest_results()[['id', 'fold']].to_dict('records') == [{'id': identity, 'fold': 0}, {'id': identity, 'fold': 1}]
    with pytest.raises(ValueError, match='fold-specific test set'):
        captured.permutation_prediction_performance(0)
    invalid = Trainer(directory, data=runner.raw, project_root=root)
    original_records = copy.deepcopy(invalid._round_data)
    with monkeypatch.context() as guarded:
        from tests.test_factorized_preparation import _instrument
        counts = _instrument(guarded)
        for fold, foreign in ((None, None), (0, None), (0, recorded['folds'][1]['holdout'])):
            invalid._round_data = copy.deepcopy(original_records)
            trial = invalid._round_data[identity]
            destination = trial if fold is None else trial['folds'][fold]
            destination['holdout'] = foreign
            before = _artifact_state(directory)
            with pytest.raises(ValueError, match=r'fold|workflow|history|population'):
                invalid.train([identity], fold=0)
            assert _artifact_state(directory) == before
        assert counts == {'target': 0, 'scaler': 0, 'model': 0, 'prepare': 0}
    invalid._round_data = original_records
    invalid._data = runner.raw.slice(1)
    before = _artifact_state(directory)
    with pytest.raises(ValueError, match='source'):
        invalid.train([identity], fold=0)
    assert _artifact_state(directory) == before
    bad_directory = root / 'bad-restore'
    bad_directory.mkdir()
    (bad_directory / 'metadata.json').write_bytes((directory / 'metadata.json').read_bytes())
    broken = copy.deepcopy(recorded)
    broken['folds'][0].pop('holdout')
    bad_path = bad_directory / 'round_data.jsonl.gz'
    _round_storage.append_record(bad_path, broken)
    rejected = WalkForwardRun(restored_manifest, runner.raw, bad_directory, promotion=True)
    before = _artifact_state(bad_directory)
    with research.operation(restored_manifest, 'experiment'), pytest.raises(ValueError, match='fold'):
        rejected.restore(bad_path, None, [(0, identity)])
    assert _artifact_state(bad_directory) == before and rejected.results.is_empty()
    manifest.set_pre_split_data_selector(lambda _frame: market)
    def preflight(*_args, **_kwargs):
        raise AssertionError('A suffix-returning selector reached fold geometry')
    monkeypatch.setattr(runner, '_preflight', preflight)
    with research.operation(manifest, 'experiment'), pytest.raises(ValueError, match='outside the research prefix'):
        runner.prepare(full, _params(config))
    release_holdout(root, reservation)
    with pytest.raises(ValueError, match='revision changed'):
        invalid.train([identity], fold=0)


@pytest.mark.parametrize(('bar_type', 'secondary'), [('base', None), ('trade', None), ('base', 'direct'), ('base', 'fixed')])
def test_protected_native_and_derived_sources(bar_type, secondary, tmp_path, monkeypatch):
    import numpy as np
    from contextlib import nullcontext
    from time import perf_counter
    from types import SimpleNamespace
    from limen.experiment._fixed_trade_sources import fixed_trade_sources
    from limen.experiment._holdout_sources import source_operation
    from limen.experiment.param_domain import ParamDomain
    from limen.experiment.param_search import GridStrategy
    from tests.test_factorize_equivalence import _numeric_equal
    from tests.test_factorized_preparation import _config, _params

    root = _project(tmp_path / 'source-support')
    config = _config()
    body = config['sfd']['manifest']
    body['bar_formation'] = {'func': 'limen.data.bars.trade_bars', 'params': {'trade_threshold': 100000}}
    config['sfd']['params']['bar_type'] = [bar_type]
    raw = pl.read_parquet(_FIXTURE).head(192)
    raw = raw.filter(pl.col('datetime') != raw['datetime'][50])
    prefix = raw.filter(pl.col('datetime').dt.epoch('ns') + 3600_000_000_000 <= history.utc_day('2024-01-07'))
    control_config = copy.deepcopy(config)
    if secondary is not None:
        from tests.test_fixed_trade_sources import _funding_frame, FUNDING_FIXTURE
        execution = pl.read_parquet(_FIXTURE).head(192)
        starts = execution['datetime'].dt.epoch('ns')
        execution = execution.with_columns(starts.alias('start_ns'), (starts + 3600_000_000_000).alias('end_ns'),
            starts.alias('open_available_at_ns'), (starts + 3600_000_000_000).alias('available_at_ns'))
        learning_starts = raw['datetime'].dt.epoch('ns')
        raw = raw.with_columns(learning_starts.alias('start_ns'), (learning_starts + 3600_000_000_000).alias('end_ns'),
            learning_starts.alias('open_available_at_ns'), (learning_starts + 3600_000_000_000).alias('available_at_ns'))
        prefix = raw.filter(pl.col('available_at_ns') <= history.utc_day('2024-01-07'))
        funding = _funding_frame().with_columns(pl.lit(str(FUNDING_FIXTURE)).alias('recorded_origin'),
            pl.lit(hashlib.sha256(FUNDING_FIXTURE.read_bytes()).hexdigest()).alias('recorded_origin_sha256'))
        for name, frame, predicate in (('execution', execution, pl.col('available_at_ns') <= history.utc_day('2024-01-07')),
                                        ('funding', funding, pl.col('settlement_at') <= history.utc_day('2024-01-07'))):
            frame.write_parquet(root / (name + '.parquet'))
            frame.filter(predicate).write_parquet(root / (name + '-prefix.parquet'))
        for reference, suffix in ((config, ''), (control_config, '-prefix')):
            source_body = reference['sfd']['manifest']
            source_body['split_dates']['test_end'] = '2024-01-06'
            source_body['backtest'] = {'product': {'kind': 'linear_perpetual', 'instrument': 'BTCUSDT', 'base_currency': 'BTC', 'quote_currency': 'USDT',
                'quantity_step': 1e-9, 'min_notional': 0.0}, 'fee_bps': 4., 'slip_bps': 3., 'notional_rate': 0.5, 'source_interval_seconds': 3600.,
                'execution_data_source': {'method': 'limen.data.HistoricalData.get_any_file', 'params': {'file_path_or_url': str(root / ('execution' + suffix + '.parquet'))}},
                'funding': {'preset': 'binance_btcusdt', 'data_source': {'method': 'limen.data.HistoricalData.get_any_file',
                    'params': {'file_path_or_url': str(root / ('funding' + suffix + '.parquet'))}}}}
    ordinary = CompiledSFD(control_config, project_root=_project(tmp_path / 'ordinary-source')).manifest()
    params = _params(config)
    baseline = ordinary.prepare_data(prefix, params.copy())
    baseline.update(_record_execution=True, _record_model_outputs=True)
    baseline_result = ordinary.run_model(baseline, params.copy())
    reserve_holdout(root, '2024-01-07', '2024-01-09', interval_seconds=3600)
    compiled = CompiledSFD(config, project_root=root)
    manifest = compiled.manifest()
    directory = root / 'results'
    directory.mkdir()
    strategy = GridStrategy(ParamDomain(config['sfd']['params']))
    identity = strategy.compute_param_hash(params)
    writer = SimpleNamespace(manifest=manifest, _walk_forward=None, _recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True),
        _yaml_reference=config, _record_execution=True, _record_model_outputs=True, _workers=1, _factorize=False,
        _search_strategy=strategy, _sfd_module_name=compiled.__name__, _get_limen_version=UniversalExperimentLoop._get_limen_version)
    with fixed_trade_sources(manifest, config['sfd']['params']) if secondary == 'fixed' else nullcontext(), source_operation(manifest):
        with pytest.raises(ValueError, match='Holdout research refuses'):
            UniversalExperimentLoop(sfd=compiled, project_root=root)
        UniversalExperimentLoop._write_metadata(writer, directory)
        started = perf_counter()
        prepared = manifest.prepare_data(raw, params.copy())
        prepared.update(_record_execution=True, _record_model_outputs=True)
        result = manifest.run_model(prepared, params.copy())
        elapsed = perf_counter() - started
        UniversalExperimentLoop._append_round_data(writer, directory / 'round_data.jsonl.gz', identity, params,
            result['_preds'], prepared['_alignment'], round_index=0)
        pl.DataFrame([params | {key: value for key, value in result.items() if not key.startswith('_')} |
            {'id': identity, '_round_index': 0, 'strict_mode_error': None, 'execution_time': elapsed}]).write_csv(directory / 'results.csv')
        if secondary is not None:
            assert prepared['_trade_context'].partitions[-1].funding_events.height > 0
            if secondary == 'fixed':
                assert manifest.prepare_data(raw, params.copy())['holdout']['workflow'] == prepared['holdout']['workflow']
                execution.slice(1).write_parquet(root / 'execution.parquet')
                from tests.test_factorized_preparation import _instrument
                with monkeypatch.context() as guarded:
                    counts = _instrument(guarded)
                    with pytest.raises(ValueError, match='frozen source differs'):
                        manifest.prepare_data(raw, params.copy())
                    assert counts['target'] == counts['scaler'] == counts['model'] == 0
                execution.write_parquet(root / 'execution.parquet')
    _numeric_equal({key: value for key, value in baseline_result.items() if not key.startswith('_')},
                   {key: value for key, value in result.items() if not key.startswith('_')})
    np.testing.assert_array_equal(result['_preds'], baseline_result['_preds'])
    for partition in ('train', 'val', 'test'):
        assert prepared['x_' + partition].equals(baseline['x_' + partition])
    proof = prepared['holdout']
    assert history.research_record(history.bind(root), proof['research_record_id'])['state'] == 'completed'
    formation = proof['workflow']['formation']
    assert formation['active'] == (bar_type != 'base')
    assert proof['source']['admitted']['gaps']
    assert all(value <= proof['cutoff_ns'] for item in formation['partitions'] for value in item['last_available_ns'])
    for item in formation['partitions']:
        assert item['derived']['rows'] == len(item['members'])
        if bar_type == 'base':
            assert item['native'] == item['derived']
        else:
            assert any(len(group) > 1 for group in item['members'])
    if bar_type == 'trade':
        from limen.data.bars import trade_bars
        future_test = raw.filter(pl.col('datetime').dt.epoch('ns') >= history.utc_day('2024-01-05'))
        times = future_test['datetime'].dt.epoch('ns').to_list()
        offsets = {instant: index for index, instant in enumerate(times)}
        crossed = [instant for instant, count in zip(trade_bars(future_test, 100000)['datetime'].dt.epoch('ns'),
            trade_bars(future_test, 100000)['bar_count'], strict=True) if instant < proof['cutoff_ns'] <= times[offsets[instant] + count - 1]]
        assert crossed and not set(crossed) & {group[0] for group in formation['partitions'][-1]['members']}
        form, options = manifest.bar_formation
        def unretained(frame, **kwargs):
            return form(frame, **kwargs).drop('bar_count')
        manifest.set_bar_formation(unretained, **options)
        from tests.test_factorized_preparation import _instrument
        with source_operation(manifest), monkeypatch.context() as guarded:
            counts = _instrument(guarded)
            before = _artifact_state(directory)
            with pytest.raises(ValueError, match='constituent counts'):
                manifest.prepare_data(raw, params.copy())
            assert _artifact_state(directory) == before
            assert counts['target'] == counts['scaler'] == counts['model'] == 0
        manifest.set_bar_formation(form, **options)
    with source_operation(manifest), research.operation(manifest, 'replay') as replay:
        replay.replay_witness = proof
        repeated = manifest.prepare_data(prefix, params.copy())
    assert repeated['holdout']['workflow'] == proof['workflow']
    with source_operation(manifest):
        sensors = Trainer(directory, data=prefix, project_root=root).train([identity])
    fitted = sensors[0]._fitted_params['_holdout']
    with _round_storage.open_text(directory / 'round_data.jsonl.gz') as stream:
        published = json.loads(next(stream))['holdout']
    assert fitted['experiment_record_id'] == published['research_record_id']
    assert fitted['workflow'] == proof['workflow']
    if secondary is not None:
        from dataclasses import replace
        from limen.experiment._holdout_sources import _roles
        bound = replace(history.bind(root), cutoff_ns=history.utc_day('2024-01-05'))
        frames, witnesses = _roles(manifest.backtest_config, params, bound, None)
        assert 1704412800001_000000 not in witnesses['funding']['clocks']['actual_support_ns']
        assert 1704412800001_000000 in funding['settlement_at'].to_list()
        assert frames[1]['settlement_at'].max() <= bound.cutoff_ns
        assert witnesses['execution']['clocks']['end_ns'][-1] == bound.cutoff_ns
        if secondary == 'direct':
            _independent_role_clocks(root, manifest.backtest_config, params, bound)


def _independent_role_clocks(root, config, params, binding):
    from dataclasses import replace
    from zipfile import ZipFile
    from limen.experiment._holdout_sources import _roles
    from limen.experiment.manifest_core import DataSourceConfig

    quarter = pl.read_parquet(_FIXTURE.parent / 'spot_15m_20250101_20250531.parquet').head(480)
    starts = quarter['datetime'].dt.epoch('ns')
    quarter = quarter.with_columns(starts.alias('start_ns'), (starts + 900_000_000_000).alias('end_ns'),
        starts.alias('open_available_at_ns'), (starts + 900_000_000_000).alias('available_at_ns'))
    quarter_path = root / 'quarter-execution.parquet'
    quarter.write_parquet(quarter_path)
    configured = replace(config, funding=None, execution_data_source=DataSourceConfig(config.execution_data_source.method, {'file_path_or_url': str(quarter_path)}))
    current = replace(binding, cutoff_ns=history.utc_day('2025-01-05'))
    frames, witness = _roles(configured, params, current, None)
    assert (frames[0]['end_ns'] - frames[0]['start_ns']).unique().to_list() == [900_000_000_000]
    assert witness['execution']['clocks']['end_ns'][-1] == current.cutoff_ns
    archive = _FIXTURE.parent / 'binance_trades_btcusdt_2025-05-23_90s.zip'
    with ZipFile(archive) as source, source.open(source.namelist()[0]) as stream:
        trades = pl.read_csv(stream, has_header=False)
    stamp, price = trades['column_5'] * 1000, trades['column_2']
    points = pl.DataFrame({'row_id': trades['column_1'].cast(pl.String), 'start_ns': stamp, 'end_ns': stamp,
        'open_available_at_ns': stamp, 'available_at_ns': stamp, **dict.fromkeys(('open', 'high', 'low', 'close'), price)})
    point_path = root / 'point-execution.parquet'
    points.write_parquet(point_path)
    configured = replace(configured, execution_data_source=DataSourceConfig(config.execution_data_source.method, {'file_path_or_url': str(point_path)}))
    frames, witness = _roles(configured, params, replace(current, cutoff_ns=history.utc_day('2025-05-24')), None)
    assert frames[0].equals(points)
    assert witness['execution']['clocks']['start_ns'] == witness['execution']['clocks']['end_ns'] == stamp.to_list()

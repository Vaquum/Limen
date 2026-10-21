import copy
import hashlib
import gzip
import json
import logging
import multiprocessing
import os
import shutil
import signal
import sqlite3
import time
from types import SimpleNamespace

import polars as pl
import pytest

from limen.experiment import UniversalExperimentLoop
from limen.experiment import _holdout_checkpoint as checkpoint
from limen.experiment import _holdout_history as history
from limen.experiment import _parallel_resume
from limen.experiment import _holdout_research as research
from limen.experiment import _round_storage as storage
from limen.experiment._recording import prepare_recording
from limen.experiment.holdout import release_holdout, reserve_holdout, withdraw_holdout
from limen.inference import Trainer
from limen.log.log import Log
from limen.yaml import CompiledSFD
from limen.yaml.compiler import build_search_strategy
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


def _checkpoint_loop(config, root, directory, *, data=None):
    compiled = CompiledSFD(config, project_root=root)
    loop = UniversalExperimentLoop(sfd=compiled, data=data, project_root=root, experiment_dir=directory,
                                   search_strategy=build_search_strategy(config), yaml_reference=config,
                                   checkpoint_interval=1)
    loop._recording = prepare_recording(None, reference=config, compiled_reference=loop._compiled_yaml,
                                       directory=directory, has_search=True, resume=directory.exists())
    return loop


def _checkpoint_rows(directory):
    with storage.open_text(storage.round_path(directory)) as stream:
        return [json.loads(line) for line in stream]


def _artifact_snapshot(directory):
    return {str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob('*') if path.is_file()}


def _restore_binding(loop, *, retain, parts=None):
    parts = loop._setup_msq_components(experiment_name='guarded', n_permutations=3) if parts is None else parts
    return loop._restore_checkpoint_state(parts['msq'], parts['domain'], parts['feedback_controller'], parts['checkpoint_manager'],
        content_hash=parts['content_hash'], strategy_type=parts['strategy_type'], csv_path=parts['csv_path'],
        round_data_path=parts['round_data_path'], retain_round_artifacts=retain)


def _refuse_checkpoint(loop, monkeypatch, *, retain):
    directory = loop._experiment_dir
    before = _artifact_snapshot(directory)
    parts = loop._setup_msq_components(experiment_name='guarded', n_permutations=3)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError('invalid population reached a state setter, fit or fold-return write')

    with monkeypatch.context() as patch:
        for item in (parts['domain'], parts['msq'], parts['feedback_controller']):
            patch.setattr(item, 'set_state', forbidden)
        patch.setattr(loop.manifest, 'prepare_data', forbidden)
        patch.setattr(loop.manifest, 'run_model', forbidden)
        patch.setattr(loop, '_walk_forward', SimpleNamespace(restore=forbidden, writer=SimpleNamespace(clear=forbidden, append=forbidden)))
        patch.setattr(loop, '_pruning_strategies', [SimpleNamespace(set_state=forbidden)])
        for parallel in (False, True):
            with pytest.raises(ValueError, match='Holdout|Protected'):
                if parallel:
                    _parallel_resume.restore_parallel(loop, parts['msq'], parts['domain'], parts['feedback_controller'], parts['checkpoint_manager'],
                        content_hash=parts['content_hash'], strategy_type=parts['strategy_type'], csv_path=parts['csv_path'],
                        round_data_path=parts['round_data_path'], failed_round_params=[], n_permutations=3)
                else:
                    _restore_binding(loop, retain=retain, parts=parts)
    assert calls == []
    assert _artifact_snapshot(directory) == before


def _refuse_trainer(directory, root, monkeypatch, *, data=None):
    from limen.targets.quantile_binary import QuantileBinaryTarget
    from limen.scalers.logreg_scaler import LogRegScaler
    from sklearn.linear_model import LogisticRegression

    before = _artifact_snapshot(directory)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError('invalid saved lineage reached target/scaler/model fitting')

    with monkeypatch.context() as patch:
        patch.setattr(QuantileBinaryTarget, '__init__', forbidden)
        patch.setattr(LogRegScaler, '__init__', forbidden)
        patch.setattr(LogisticRegression, 'fit', forbidden)
        with pytest.raises(ValueError, match='Holdout|Protected'):
            trainer = Trainer(directory, data=data, project_root=root)
            trainer.train([_checkpoint_rows(directory)[0]['round_id']])
    assert calls == []
    assert _artifact_snapshot(directory) == before


@pytest.mark.parametrize('retain', [False, True])
def test_protected_checkpoint_binding(tmp_path, monkeypatch, retain):
    assert hashlib.sha256(_FIXTURE.read_bytes()).hexdigest() == 'a634bbd4f3692fac7bc88512da588e7772f4bd1b749567606d5314e54d0adf72'
    root = _project(tmp_path / 'project')
    config = _configuration(root)
    config['sfd']['params']['C'] = [1.0, 0.5, 0.25]
    config['uel']['n_permutations'] = 3
    reservation = reserve_holdout(root, '2024-01-05', '2024-01-06', interval_seconds=3600)
    directory = root / 'results'
    loop = _checkpoint_loop(config, root, directory)
    save = UniversalExperimentLoop._checkpoint

    def interrupted(host, *args, **kwargs):
        save(host, *args, **kwargs)
        if host is loop and not host._shutdown_requested:
            signal.raise_signal(signal.SIGTERM)

    handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    try:
        with monkeypatch.context() as patch:
            patch.setattr(UniversalExperimentLoop, '_checkpoint', interrupted)
            loop.run('guarded', n_permutations=3, progress_bar=False, post_processing=False)
        rows = _checkpoint_rows(directory)
        assert len(rows) == 2 and loop.preds == [] and loop._shutdown_requested
        cp = json.loads((directory / 'checkpoint.json').read_text())
        metadata = json.loads((directory / 'metadata.json').read_text())
        assert cp['metadata']['experiment_round'] == 1
        expected = checkpoint.header(loop.manifest)
        assert metadata['holdout_binding'] == expected
        assert {key: cp['holdout_binding'][key] for key in expected} == expected
        assert cp['holdout_binding']['completed_rounds'] == [[row['round_id'], row['holdout']['research_record_id']] for row in rows]
        assert storage.round_path(directory).suffix == '.gz' and (directory / 'round_signals').is_dir()
        saved = _artifact_snapshot(directory)

        def restore_files():
            for relative, (contents, mtime) in saved.items():
                path = directory / relative
                path.write_bytes(contents)
                os.utime(path, ns=(mtime, mtime))

        def altered(cp_change=None, metadata_change=None, row_change=None):
            value, meta, records = copy.deepcopy(cp), copy.deepcopy(metadata), copy.deepcopy(rows)
            if cp_change is not None:
                cp_change(value)
                (directory / 'checkpoint.json').write_text(json.dumps(value))
            if metadata_change is not None:
                metadata_change(meta)
                (directory / 'metadata.json').write_text(json.dumps(meta))
            if row_change is not None:
                row_change(records)
                storage.round_path(directory).write_bytes(b''.join(gzip.compress((json.dumps(row) + '\n').encode(), mtime=0) for row in records))
            _refuse_checkpoint(_checkpoint_loop(config, root, directory), monkeypatch, retain=retain)
            restore_files()

        altered(cp_change=lambda value: value.pop('holdout_binding'))
        altered(metadata_change=lambda value: value.pop('holdout_binding'))
        altered(cp_change=lambda value: value['holdout_binding'].update(schema_version=2))
        altered(cp_change=lambda value: value['holdout_binding'].pop('cutoff_ns'))
        altered(metadata_change=lambda value: value['holdout_binding'].update(schema_version=True))
        altered(cp_change=lambda value: value['holdout_binding']['population'].update(input_sha256=''))
        altered(row_change=lambda values: values[0].pop('holdout'))
        altered(row_change=lambda values: values[0]['holdout']['selected'].update(rows=0))
        altered(row_change=lambda values: values[0]['holdout'].update(model_schema=[]))
        altered(row_change=lambda values: values[0]['holdout']['source'].update(admitted_sha256=''))
        # A different genuine completed experiment row at this same binding is insufficient.
        assert history.research_record(history.bind(root), rows[1]['holdout']['research_record_id'])['state'] == 'completed'
        altered(row_change=lambda values: values[0]['holdout'].update(research_record_id=values[1]['holdout']['research_record_id']))
        altered(row_change=lambda values: values[0]['holdout'].update(experiment_record_id=values[1]['holdout']['research_record_id']))
        fresh = _checkpoint_loop(config, root, directory)
        original = fresh.data
        for changed in (original.slice(1), original.with_columns(pl.col('close').cast(pl.Float32))):
            fresh.data = changed
            _refuse_checkpoint(fresh, monkeypatch, retain=retain)
        changed = copy.deepcopy(config)
        changed['sfd']['manifest']['split_dates']['train_start'] = '2024-01-02'
        _refuse_checkpoint(_checkpoint_loop(changed, root, directory), monkeypatch, retain=retain)
        alternate_source = tmp_path / 'same-recorded-source.parquet'
        alternate_source.write_bytes(_FIXTURE.read_bytes())
        changed = copy.deepcopy(config)
        changed['sfd']['manifest']['data_source']['params']['file_path_or_url'] = str(alternate_source)
        _refuse_checkpoint(_checkpoint_loop(changed, root, directory), monkeypatch, retain=retain)
        relocated = tmp_path / 'relocated-project'
        shutil.copytree(root, relocated)
        _refuse_checkpoint(_checkpoint_loop(config, relocated, relocated / 'results'), monkeypatch, retain=retain)
        fresh = _checkpoint_loop(config, root, directory)
        fresh.manifest.set_pre_split_data_selector(lambda frame: frame.slice(1))
        _refuse_checkpoint(fresh, monkeypatch, retain=retain)

        # Metadata predating this envelope remains valid only with its genuine ordinary V1 source proof.
        legacy = copy.deepcopy(metadata)
        legacy.pop('holdout_binding')
        legacy['limen_version'] = '5.45.26'
        (directory / 'metadata.json').write_text(json.dumps(legacy))
        sensor = Trainer(directory, project_root=root).train([rows[0]['round_id']])[0]
        assert sensor._fitted_params['_holdout']['experiment_record_id'] == rows[0]['holdout']['research_record_id']
        restore_files()
        damaged = copy.deepcopy(rows)
        damaged[0].pop('holdout')
        storage.round_path(directory).write_bytes(b''.join(gzip.compress((json.dumps(row) + '\n').encode(), mtime=0) for row in damaged))
        _refuse_trainer(directory, root, monkeypatch)
        restore_files()

        resumed = _checkpoint_loop(config, root, directory)
        with pytest.raises(ValueError, match='refuses.*resume'):
            resumed.run('guarded', n_permutations=3, progress_bar=False, resume=True)
        # Exercise the reviewed private restore boundary; public admission remains guarded until workflow composition.
        with research.operation(resumed.manifest, 'experiment'):
            resumed.data = research.admit(resumed.manifest, resumed.data)
            assert _restore_binding(resumed, retain=retain) == 2
            assert len(resumed._alignment) == (2 if retain else 0)
            resumed._run('guarded', n_permutations=3, resume=True, post_processing=retain, progress_bar=False)
        complete = _checkpoint_rows(directory)
        assert len(complete) == 3
        assert complete[0]['holdout'] == rows[0]['holdout']
        assert complete[2]['holdout']['experiment_record_id'] != rows[0]['holdout']['experiment_record_id']
        restored = _checkpoint_loop(config, root, directory)
        assert _restore_binding(restored, retain=True) == 3
        assert restored._alignment[0]['holdout'] == rows[0]['holdout']
        assert Log(uel_object=restored)._get_test_data_with_all_cols(0).height > 0
        sensors = Trainer(directory, project_root=root).train([complete[0]['round_id'], complete[2]['round_id']])
        assert [item._fitted_params['_holdout']['experiment_record_id'] for item in sensors] == [complete[index]['holdout']['research_record_id'] for index in (0, 2)]

        release_holdout(root, reservation)
        _refuse_checkpoint(_checkpoint_loop(config, root, directory), monkeypatch, retain=retain)
        _refuse_trainer(directory, root, monkeypatch)
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)

    # A real, already-truncated unprotected population must not acquire protection by UUID substitution.
    control_root = _project(tmp_path / 'control')
    control_config = _configuration(control_root)
    control_dir = control_root / 'results'
    raw = pl.read_parquet(_FIXTURE).head(96)
    control = _checkpoint_loop(control_config, control_root, control_dir, data=raw)
    control.run('control', n_permutations=1, progress_bar=False, post_processing=False)
    parts = control._setup_msq_components(experiment_name='control', n_permutations=1)
    control._checkpoint(parts['msq'], parts['domain'], control_dir, parts['checkpoint_manager'], 0, 1,
                        strategy_type=parts['strategy_type'], content_hash=parts['content_hash'])
    zero = json.loads((control_dir / 'metadata.json').read_text())['holdout_binding']
    assert zero['cutoff_ns'] is None and set(zero) == {'schema_version', 'project_id', 'reservation_revision', 'cutoff_ns', 'interval_seconds'}
    assert _restore_binding(_checkpoint_loop(control_config, control_root, control_dir, data=raw), retain=retain) == 1
    row = _checkpoint_rows(control_dir)[0]
    assert Trainer(control_dir, data=raw, project_root=control_root).train([row['round_id']])
    reserve_holdout(control_root, '2024-01-05', '2024-01-06', interval_seconds=3600)
    _refuse_checkpoint(_checkpoint_loop(control_config, control_root, control_dir, data=raw), monkeypatch, retain=retain)
    _refuse_trainer(control_dir, control_root, monkeypatch, data=raw)
    before = json.loads((control_dir / 'metadata.json').read_text())
    before.pop('holdout_binding')
    before['limen_version'] = '5.45.26'
    (control_dir / 'metadata.json').write_text(json.dumps(before))
    _refuse_trainer(control_dir, control_root, monkeypatch, data=raw)

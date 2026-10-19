import copy
from contextlib import ExitStack
import json
import logging
import multiprocessing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from limen.cohort import Cohort
from limen.experiment import UniversalExperimentLoop
from limen.experiment import _holdout_attempts as attempts
from limen.experiment import _holdout_assessment as assessment
from limen.experiment import _holdout_history as history
from limen.experiment import _holdout_report as reports
from limen.experiment import manifest_core
from limen.experiment._holdout_freeze import owned_copy
from limen.experiment.holdout import FrozenHoldoutSelection, final_holdout_report, freeze_holdout_selection, read_holdout_report, release_holdout, reserve_holdout, withdraw_holdout
from limen.inference import Trainer
from limen.targets.next_bar_up import NextBarUpTarget
from limen.yaml import CompiledSFD
from limen.yaml.compiler import build_search_strategy
from tests.test_holdout_research import _configuration, _project

_SCENARIO = {'fee_bps': 5.0, 'slip_bps': 2.0, 'notional_rate': 0.25, 'execution_lag_bars': 1, 'take_profit_bps': 20.0, 'stop_loss_bps': 20.0}
_REPEAT_KEYS = ('spec', 'spec_sha256', 'state_sha256', 'scenarios_sha256', 'source', 'source_schema', 'source_column_map', 'planned', 'classification', 'scenarios', 'rows_sha256')


@pytest.fixture(scope='module')
def assessment_case(tmp_path_factory):
    root = _project(tmp_path_factory.mktemp('holdout') / 'project')
    config = _configuration(root)
    reservation = reserve_holdout(root, '2024-01-05', '2024-01-07', interval_seconds=3600)
    trained, sources = [], []
    for architecture in ('logreg', 'lightgbm'):
        current = copy.deepcopy(config)
        if architecture == 'logreg':
            current['sfd']['manifest']['calibration'] = {'probability_calibration': {'func': 'limen.calibration.sklearn_probability_calibrator', 'params': {'method': 'sigmoid'}}}
            current['sfd']['params']['C'] = [1.0, 0.5]
            current['uel']['n_permutations'] = 2
        else:
            current['sfd']['manifest']['reference_architecture'] = 'limen.sfd.reference_architecture.lightgbm_binary'
            current['sfd']['params'] = {'scaler_type': ['logreg'], 'n_estimators': [3], 'num_leaves': [3], 'n_jobs': [1], 'early_stopping_rounds': [0], 'verbosity': [-1]}
        directory = root / architecture
        loop = UniversalExperimentLoop(sfd=CompiledSFD(current, project_root=root), project_root=root, experiment_dir=directory,
            search_strategy=build_search_strategy(current), yaml_reference=current)
        loop.run(architecture, n_permutations=current['uel']['n_permutations'], progress_bar=False, post_processing=True)
        from limen.experiment import _round_storage
        with _round_storage.open_text(_round_storage.round_path(directory)) as stream:
            ids = [json.loads(line)['round_id'] for line in stream if line.strip()]
        trained.append(Trainer(directory, project_root=root).train(ids))
        sources.append(directory)
    cohort = Cohort(experiment_log_path=str(sources[0]), permutation_ids=[sensor.permutation_id for sensor in trained[0]])
    cohort.set_members(trained[0])
    return root, reservation, trained, cohort


def _freeze(selection, *, context='2024-01-04', target=True, scenarios=None):
    return freeze_holdout_selection(selection, context_start=context, scenarios=[_SCENARIO] if scenarios is None else scenarios,
        evaluation_target=NextBarUpTarget(pl.DataFrame(), 'observed_up') if target else None)


def _report(frozen, case):
    return final_holdout_report(frozen, project_root=case[0], reservation_id=case[1], end='2024-01-06')


def test_holdout_frozen_causal_assessment(assessment_case):
    root, _, trained, cohort = assessment_case
    frozen = _freeze(cohort)
    with pytest.raises(TypeError):
        FrozenHoldoutSelection(object())
    probe = pl.read_parquet(Path(trained[0][0]._yaml_reference['sfd']['manifest']['data_source']['params']['file_path_or_url'])).head(120)
    for sensor in (trained[0][0], trained[1][0]):
        assert sensor.predict(probe).reason is None
        clone = owned_copy(sensor)
        second = owned_copy(clone)
        assert sensor.predict(probe) == clone.predict(probe) == second.predict(probe)
    trained[0][0]._round_params['C'] = 999
    calls = []
    paths = ('limen.sfd.reference_architecture.logreg_binary.LogRegBinary.train', 'limen.sfd.reference_architecture.lightgbm_binary.LightGBMBinary.train',
        'sklearn.linear_model.LogisticRegression.fit', 'lightgbm.LGBMClassifier.fit', 'sklearn.calibration.CalibratedClassifierCV.fit',
        'sklearn.decomposition.PCA.fit', 'sklearn.decomposition.PCA.fit_transform', 'limen.calibration.grid_threshold_optimizer',
        'limen.sfd.reference_architecture.logreg_binary.fit_calibrator', 'limen.sfd.reference_architecture.lightgbm_binary.fit_calibrator')
    with ExitStack() as stack:
        for path in paths:
            stack.enter_context(patch(path, side_effect=lambda *args, **kwargs: calls.append('fit')))
        for sensor in trained[0] + trained[1]:
            for value in sensor._fitted_params.values():
                if type(value).__module__.startswith(('limen.scalers.', 'limen.targets.')):
                    stack.enter_context(patch.object(type(value), '__init__', side_effect=lambda *args, **kwargs: calls.append('fitted-transform')))
        report = _report(frozen, assessment_case)
    assert calls == []
    rows = pl.read_parquet(root / report['rows_path'])
    assert rows.height == 48 and rows.filter(pl.col('role') == 'final').height == 24
    assert all(len(row) == 2 for row in rows.filter(pl.col('role') == 'final')['members'].to_list())
    assert read_holdout_report(root, report['attempt_id']) == report
    repeat = _report(frozen, assessment_case)
    assert {key: repeat[key] for key in _REPEAT_KEYS} == {key: report[key] for key in _REPEAT_KEYS}
    assert repeat['exposure']['class'] == 'subsequent'
    assert _report(_freeze(trained[1][0]), assessment_case)['state'] == 'completed'
    changed = copy.deepcopy(trained[1][0])
    changed._model.model.booster_.best_iteration = 1
    with pytest.raises(ValueError, match='best_iteration'):
        _freeze(changed)
    changed = copy.deepcopy(trained[0][1])
    changed._model._fitted_calibrator = None
    with pytest.raises(ValueError, match='unfitted calibration'):
        _freeze(changed)
    changed = copy.deepcopy(trained[0][1])
    changed._get_manifest().decoder_lookback = 2
    with pytest.raises(ValueError, match='cached manifest'):
        _freeze(changed)
    for key, invalid in (('fee_bps', True), ('slip_bps', -1), ('notional_rate', 0), ('execution_lag_bars', 1.0), ('stop_loss_bps', float('nan'))):
        with pytest.raises(ValueError):
            _freeze(trained[0][1], scenarios=[{**_SCENARIO, key: invalid}])


def _attempt_worker(root, reservation, pipe):
    history._BUSY_TIMEOUT_SECONDS = 0.2
    attempts.migrate(root)
    with history.admission(root):
        declaration = attempts.reservation(root, reservation, None, '2024-01-06', '2024-01-04')
        pipe.send('leased')
        assert pipe.recv() == 'allocate'
        attempt = attempts.allocate(root, declaration, {'context_start_ns': history.utc_day('2024-01-04'), 'start_ns': history.utc_day('2024-01-05'), 'end_ns': history.utc_day('2024-01-06')}, spec='structural', state='structural', scenarios='structural')
        pipe.send(attempt['id'])
        if pipe.recv() == 'fail':
            attempts.finalize(root, attempt['id'], source=None, error={'stage': 'load', 'exception': 'ValueError', 'message': 'structural interruption', 'member_id': None, 'bar_ns': None})
            pipe.send('failed')


def test_holdout_attempt_history(assessment_case, monkeypatch, tmp_path):
    root, reservation, trained, _ = assessment_case
    binding = history.read_binding(root)
    assert binding.project_id
    attempts.migrate(root)
    frozen = _freeze(trained[0][1])
    with frozen._lock, pytest.raises(ValueError, match='idle'):
        _report(frozen, assessment_case)
    with patch('limen.experiment._holdout_assessment.load_admitted_source', side_effect=AssertionError('must not read')) as loader:
        with pytest.raises(ValueError, match='contained'):
            final_holdout_report(frozen, project_root=root, reservation_id=reservation, start='2024-01-03')
        assert loader.call_count == 0
    with patch('limen.experiment._holdout_assessment.load_admitted_source', side_effect=ValueError('recorded source unavailable')), pytest.raises(ValueError, match='Holdout attempt') as error:
        _report(frozen, assessment_case)
    with history.connection(root / history._HISTORY) as db:
        row = db.execute('SELECT * FROM attempts ORDER BY assessment_index DESC LIMIT 1').fetchone()
        assert row['state'] == 'failed' and row['id'] in str(error.value)
        assert json.loads(row['error_json'])['stage'] == 'load'
        assert db.execute('PRAGMA user_version').fetchone()[0] == 2
    with pytest.raises(ValueError, match='used'):
        withdraw_holdout(root, reservation)
    report = _report(frozen, assessment_case)
    assert row['id'] in [entry['id'] for entry in report['prior']['attempts']]
    with patch.object(reports, '_sync_replace', side_effect=OSError('disk write failed')), pytest.raises(ValueError, match='disk write failed'):
        _report(frozen, assessment_case)
    with history.connection(root / history._HISTORY) as db:
        failed = db.execute('SELECT id,state FROM attempts ORDER BY assessment_index DESC LIMIT 1').fetchone()
        assert failed['state'] == 'failed'
    with pytest.raises(ValueError, match='not completed'):
        read_holdout_report(root, failed['id'])
    scale = manifest_core._apply_scaler
    def changed_schema(*args):
        frame, fitted = scale(*args)
        return frame.with_columns(pl.col(pl.Float64).cast(pl.Float32)), fitted
    with patch.object(manifest_core, '_apply_scaler', side_effect=changed_schema), pytest.raises(ValueError, match='schema mismatch'):
        _report(frozen, assessment_case)
    with history.connection(root / history._HISTORY) as db:
        error = json.loads(db.execute('SELECT error_json FROM attempts ORDER BY assessment_index DESC LIMIT 1').fetchone()[0])
        assert error['stage'] == 'predict' and error['member_id'] and error['bar_ns'] == history.utc_day('2024-01-05')
    assert not any(isinstance(handler, assessment._Failures) for handler in logging.getLogger('limen.inference.sensor').handlers)
    monkeypatch.setattr(history, '_BUSY_TIMEOUT_SECONDS', 0.2)
    context = multiprocessing.get_context('fork')
    pipes, workers = [], []
    try:
        for _ in range(2):
            parent, child = context.Pipe()
            worker = context.Process(target=_attempt_worker, args=(root, reservation, child))
            worker.start()
            assert parent.poll(5) and parent.recv() == 'leased'
            pipes.append(parent)
            workers.append(worker)
        ids = []
        for pipe in pipes:
            pipe.send('allocate')
            assert pipe.poll(5)
            ids.append(pipe.recv())
        pipes[1].send('fail')
        assert pipes[1].poll(5) and pipes[1].recv() == 'failed'
        workers[0].terminate()
        for worker in workers:
            worker.join(5)
            assert not worker.is_alive()
        with history.connection(root / history._HISTORY) as db:
            entries = [dict(db.execute('SELECT * FROM attempts WHERE id=?', (value,)).fetchone()) for value in ids]
        assert [entry['state'] for entry in entries] == ['started', 'failed']
        assert entries[1]['assessment_index'] == entries[0]['assessment_index'] + 1
        assert ids[0] in [entry['id'] for entry in json.loads(entries[1]['prior_json'])['attempts']]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(5)
        for pipe in pipes:
            pipe.close()
    migration = _project(tmp_path / 'migration')
    unused = reserve_holdout(migration, '2024-01-05', '2024-01-06', interval_seconds=3600)
    with history.connection(migration / history._HISTORY) as db:
        before = dict(db.execute('SELECT * FROM reservations').fetchone())
        assert db.execute('PRAGMA user_version').fetchone()[0] == 1
    attempts.migrate(migration)
    with history.connection(migration / history._HISTORY) as db:
        assert dict(db.execute('SELECT * FROM reservations').fetchone()) == before
        assert db.execute('PRAGMA user_version').fetchone()[0] == 2
    withdraw_holdout(migration, unused)


def test_holdout_report_accounting(assessment_case):
    root, reservation, trained, _ = assessment_case
    frozen = _freeze(trained[0][1], scenarios=[_SCENARIO, _SCENARIO])
    report = _report(frozen, assessment_case)
    rows = pl.read_parquet(root / report['rows_path']).filter(pl.col('role') == 'final')
    assert rows['label'][-1] is None and rows['label_reason'][-1] == 'terminal-label-unavailable'
    assert report['classification']['accuracy']['denominator'] == 23
    ledgers = rows['scenarios'].to_list()
    for values in ledgers:
        assert len(values) == 2 and values[0]['raw_net'] == values[1]['raw_net']
        for ledger in values:
            for key in ('position', 'gross', 'net', 'cost'):
                assert ledger[f'deployed_{key}'] == ledger[f'raw_{key}'] * 0.25
            assert ledger['raw_cost'] == ledger['raw_gross'] - ledger['raw_net']
    net = np.array([values[0]['deployed_net'] for values in ledgers])
    assert report['scenarios'][0]['metrics']['pnl_per_bar_bps']['value'] == pytest.approx(net.mean() * 10000)
    absent = _report(_freeze(trained[0][1], target=False), assessment_case)
    assert all(value['value'] is None and value['reason'] == 'no-labels' for value in absent['classification'].values())
    assert reports.classification([0, 0, None], [0, 0, 1], [None, None, None])['precision']['reason'] == 'zero-denominator'
    assert reports.classification([1, 1], [1, 1], [0.8, 0.9])['auc']['reason'] == 'one-class'
    path = root / report['rows_path']
    original = path.read_bytes()
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash mismatch'):
        read_holdout_report(root, report['attempt_id'])
    path.write_bytes(original)
    release_holdout(root, reservation)
    assert _report(frozen, assessment_case)['exposure']['class'] == 'post-release'

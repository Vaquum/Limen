import copy
from contextlib import ExitStack, contextmanager
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
import json
import logging
import multiprocessing
import os
from importlib.metadata import version
from pathlib import Path
from threading import Thread
from typing import ClassVar
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
from limen.experiment._holdout_freeze import members, owned_copy
from limen.experiment.holdout import FrozenHoldoutSelection, final_holdout_report, freeze_holdout_selection, read_holdout_report, release_holdout, reserve_holdout, withdraw_holdout
from limen.inference import Trainer
from limen.inference.sensor import Sensor
from limen.sfd.reference_architecture.logreg_binary import LogRegBinary
from limen.targets.next_bar_up import NextBarUpTarget
from limen.yaml import CompiledSFD
from limen.yaml.compiler import build_search_strategy
from tests.test_holdout_research import _configuration, _project

_SCENARIO = {'fee_bps': 5.0, 'slip_bps': 2.0, 'notional_rate': 0.25, 'execution_lag_bars': 1, 'take_profit_bps': 20.0, 'stop_loss_bps': 20.0}
_REPEAT_KEYS = ('spec', 'spec_sha256', 'state_sha256', 'scenarios_sha256', 'source', 'source_schema', 'source_column_map', 'planned', 'classification', 'scenarios', 'rows_sha256')


_LATE_RULES = {'offset': 0.0}
_COPY_AFTER_FREEZE = False
_GLOBAL_FITTED_PROVIDER = None
_ALIAS_SENTINEL = object()
_ALIAS_REGISTRY = {'default': _ALIAS_SENTINEL}


def later_probabilities(values):
    return np.clip(values + _LATE_RULES['offset'], 0, 1)


fit_calibrator = later_probabilities


def identity_probabilities(values):
    return values


class InferenceOffset:
    def __init__(self):
        self.offset = _LATE_RULES['offset']
    def adjust(self, values):
        return np.clip(values + self.offset, 0, 1)


def alternate_offset(self):
    self.offset = _LATE_RULES['offset'] + 0.2


_OFFSET_REGISTRY = {'alternate': alternate_offset, 'offset': InferenceOffset}


class _SourceCache:
    def __deepcopy__(self, memo):
        raise AssertionError('Source/log caches must not be traversed before allocation')


class ExternalRobust:
    def __init__(self, frame):
        from limen.scalers import RobustScaler
        self.fitted = RobustScaler(frame)
    def transform(self, frame):
        return self.fitted.transform(frame)


class FittedProbabilityAdjustment:
    __slots__ = ('__offset', 'estimator')
    def __init__(self, estimator, x, y):
        self.estimator = estimator
        self.__offset = float(np.mean(np.asarray(y)) - np.mean(estimator.predict_proba(np.asarray(x))[:, 1]))
    def predict_proba(self, x):
        positive = np.clip(self.estimator.predict_proba(np.asarray(x))[:, 1] + self.__offset, 0, 1)
        return np.column_stack((1 - positive, positive))


def external_calibrator(estimator, x, y, kind='native', **params):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from limen.calibration import sklearn_probability_calibrator
    if kind == 'passthrough':
        return estimator
    if kind == 'detached':
        return CalibratedClassifierCV(FrozenEstimator(copy.deepcopy(estimator)), method='sigmoid').fit(x, y)
    if kind == 'custom':
        return FittedProbabilityAdjustment(estimator, x, y)
    return sklearn_probability_calibrator(estimator, x, y, **params)


def readonly_wrapper(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        return function(*args, **kwargs) if args or kwargs else None
    return wrapped


class WrappedLogReg(LogRegBinary):
    predict = readonly_wrapper(LogRegBinary.predict)


class CalibratorFactory(partial):
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs) if args else None


def mixed_architecture(data, C=1.0, prediction_calibration_config=None):
    from limen.sfd.reference_architecture.logreg_binary import logreg_binary
    from limen.sfd.reference_architecture.lightgbm_binary import lightgbm_binary
    if prediction_calibration_config is not None and prediction_calibration_config.threshold_func is not None:
        result = logreg_binary(data, C=C, prediction_calibration_config=prediction_calibration_config)
        result['_model'].__class__ = WrappedLogReg
        return result
    return lightgbm_binary(data, n_estimators=5, num_leaves=3, n_jobs=1, early_stopping_rounds=2,
                           prediction_calibration_config=prediction_calibration_config)


def tree_architecture(data, prediction_calibration_config=None):
    from sklearn.tree import DecisionTreeClassifier
    from limen.sfd.reference_architecture.base import ReferenceModel
    from limen.sfd.reference_architecture.logreg_binary import LogRegBinary
    class FittedTree(ReferenceModel):
        predict = LogRegBinary.predict
        evaluate = LogRegBinary.evaluate
        def train(self, data, **params):
            return self
    model = FittedTree()
    model.model = DecisionTreeClassifier(max_depth=2, random_state=42).fit(np.asarray(data['x_train']), np.where(np.asarray(data['y_train']) == 1, 1, -1))
    model.prediction_calibration_config = prediction_calibration_config
    model._fitted_calibrator, model._calibration_threshold, model._val_score = None, 0.5, None
    model.research_label = 'selected fitted tree'
    result = model.evaluate(data)
    result['_model'] = model
    return result


@pytest.fixture(scope='module')
def assessment_case(tmp_path_factory):
    root = _project(tmp_path_factory.mktemp('holdout') / 'project')
    config = _configuration(root)
    reservation = reserve_holdout(root, '2024-01-05', '2024-01-07', interval_seconds=3600)
    from limen import calibration
    from limen import scalers
    import limen.sfd.reference_architecture as architectures
    aliases = ExitStack()
    aliases.enter_context(patch.object(calibration, 'f17_calibrator', CalibratorFactory(external_calibrator), create=True))
    aliases.enter_context(patch.object(calibration, 'f17_threshold', readonly_wrapper(calibration.grid_threshold_optimizer), create=True))
    aliases.enter_context(patch.object(scalers, 'f17_scaler', ExternalRobust, create=True))
    aliases.enter_context(patch.object(architectures, 'f17_mixed', mixed_architecture, create=True))
    aliases.enter_context(patch.object(architectures, 'f17_tree', tree_architecture, create=True))
    trained, sources = [], []
    for architecture in ('logreg', 'lightgbm', 'warmup', 'earlier_test'):
        current = copy.deepcopy(config)
        if architecture == 'logreg':
            current['sfd']['manifest']['reference_architecture'] = 'limen.sfd.reference_architecture.f17_mixed'
            current['sfd']['manifest']['calibration'] = {'probability_calibration': {'func': 'limen.calibration.f17_calibrator', 'params': {'method': 'sigmoid'}}, 'threshold_function': {'func': 'limen.calibration.f17_threshold'}}
            current['sfd']['params']['C'] = [1.0]
            current['sfd']['params']['use_calibration'] = [True, False]
            current['sfd']['params']['use_threshold'] = [True, False]
            current['uel']['n_permutations'] = 4
        elif architecture == 'warmup':
            current['sfd']['manifest']['reference_architecture'] = 'limen.sfd.reference_architecture.f17_tree'
            current['sfd']['manifest']['scaler'] = {'class': 'limen.scalers.f17_scaler'}
            current['sfd']['params'] = {}
            current['sfd']['manifest']['calibration'] = {'probability_calibration': {'func': 'limen.calibration.f17_calibrator', 'params': {'kind': 'custom'}}}
            current['sfd']['manifest']['indicators'][0]['params']['period'] = 30
            current['sfd']['manifest']['target']['fit_params']['source_column'] = 'roc_30'
        elif architecture == 'earlier_test':
            current['sfd']['manifest']['calibration'] = {'probability_calibration': {'func': 'limen.calibration.f17_calibrator', 'params': {'kind': 'passthrough'}}}
            current['sfd']['manifest']['split_dates'] = dict(zip(('train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end'), ('2024-01-01', '2024-01-02', '2024-01-02', '2024-01-03', '2024-01-03', '2024-01-04'), strict=True))
        else:
            current['sfd']['manifest']['reference_architecture'] = 'limen.sfd.reference_architecture.lightgbm_binary'
            current['sfd']['manifest']['calibration'] = {'probability_calibration': {'func': 'limen.calibration.f17_calibrator', 'params': {'kind': 'detached'}}}
            current['uel']['n_permutations'] = 2
            current['sfd']['params'] = {'use_calibration': [True, False], 'use_threshold': [False], 'scaler_type': ['logreg'], 'n_estimators': [3], 'num_leaves': [3], 'n_jobs': [1], 'early_stopping_rounds': [0], 'verbosity': [-1]}
        directory = root / architecture
        loop = UniversalExperimentLoop(sfd=CompiledSFD(current, project_root=root), project_root=root, experiment_dir=directory,
            search_strategy=build_search_strategy(current), yaml_reference=current)
        loop.run(architecture, n_permutations=current['uel']['n_permutations'], progress_bar=False, post_processing=True)
        from limen.experiment import _round_storage
        with _round_storage.open_text(_round_storage.round_path(directory)) as stream:
            ids = [json.loads(line)['round_id'] for line in stream if line.strip()]
        trained.append(Trainer(directory, project_root=root).train(ids))
        sources.append(directory)
    selected = [trained[0][0], trained[0][-1]]
    cohort = Cohort(experiment_log_path=str(sources[0]), permutation_ids=[sensor.permutation_id for sensor in selected])
    cohort.set_members(selected)
    cohort.aggregation_mode = 'probability_weighted'
    try:
        yield root, reservation, trained, cohort
    finally:
        aliases.close()


def _freeze(selection, *, context='2024-01-04', target=True, scenarios=None):
    return freeze_holdout_selection(selection, context_start=context, scenarios=[_SCENARIO] if scenarios is None else scenarios,
        evaluation_target=NextBarUpTarget(pl.DataFrame(), 'observed_up') if target else None)


def _report(frozen, case):
    return final_holdout_report(frozen, project_root=case[0], reservation_id=case[1], end='2024-01-06')


@contextmanager
def _mutated_selection(sensor):
    parameters = dict(sensor._round_params)
    weights = sensor._model.model.coef_
    original = weights.copy()
    sensor._round_params['C'] = 999
    weights[...] += 1
    try:
        yield
    finally:
        sensor._round_params.clear()
        sensor._round_params.update(parameters)
        weights[...] = original


def test_holdout_frozen_causal_assessment(assessment_case):
    root, _, trained, cohort = assessment_case
    target = NextBarUpTarget(pl.DataFrame(), 'observed_up')
    with patch.object(cohort, '_round_sources', _SourceCache()), patch.object(trained[0][0]._get_manifest(), '_unused_source_cache', _SourceCache(), create=True):
        frozen = freeze_holdout_selection(cohort, context_start='2024-01-04', scenarios=[_SCENARIO], evaluation_target=target)
    target.target_name = 'caller-mutated'
    with pytest.raises(TypeError):
        FrozenHoldoutSelection(object())
    probe = pl.read_parquet(Path(trained[0][0]._yaml_reference['sfd']['manifest']['data_source']['params']['file_path_or_url'])).head(120)
    def unavailable_python_callback(*args):
        raise AssertionError('Python callbacks must be refused before invocation')
    python_sensor = Sensor.from_python(fitted_state=trained[0][0]._model, params={}, infer=unavailable_python_callback,
        map_output=unavailable_python_callback, state_id='caller-owned', fitting_cutoff=probe['datetime'][24], selection_cutoff=probe['datetime'][48])
    with pytest.raises(ValueError, match='exact Sensor'):
        _freeze(python_sensor)
    for sensor in (*trained[0], trained[1][0]):
        assert sensor.predict(probe).reason is None
        _freeze(sensor)
        clone = owned_copy(sensor)
        second = owned_copy(clone)
        assert sensor.predict(probe) == clone.predict(probe) == second.predict(probe)
    calls = []
    predictions = []
    real_tape = assessment._tape
    def traced_tape(selection, raw, start, failures):
        with ExitStack() as stack:
            for member in members(selection):
                def predict(frame, original=member.predict, identity=member.permutation_id):
                    predictions.append((identity, frame.height, frame['datetime'][0], frame['datetime'][-1]))
                    return original(frame)
                stack.enter_context(patch.object(member, 'predict', side_effect=predict))
            return real_tape(selection, raw, start, failures)
    paths = ('limen.sfd.reference_architecture.logreg_binary.LogRegBinary.train', 'limen.sfd.reference_architecture.lightgbm_binary.LightGBMBinary.train',
        'sklearn.linear_model.LogisticRegression.fit', 'lightgbm.LGBMClassifier.fit', 'sklearn.calibration.CalibratedClassifierCV.fit',
        'sklearn.decomposition.PCA.fit', 'sklearn.decomposition.PCA.fit_transform', 'limen.calibration.threshold._validate_threshold_grid',
        'limen.sfd.reference_architecture.logreg_binary.fit_calibrator', 'limen.sfd.reference_architecture.lightgbm_binary.fit_calibrator')
    with ExitStack() as stack:
        stack.enter_context(_mutated_selection(trained[0][0]))
        stack.enter_context(patch.object(assessment, '_tape', side_effect=traced_tape))
        for path in paths:
            stack.enter_context(patch(path, new=lambda *args, **kwargs: calls.append('fit')))
        for sensor in trained[0] + trained[1]:
            for value in sensor._fitted_params.values():
                if type(value).__module__.startswith(('limen.scalers.', 'limen.targets.')):
                    stack.enter_context(patch.object(type(value), '__init__', new=lambda *args, **kwargs: calls.append('fitted-transform')))
        report = _report(frozen, assessment_case)
    assert calls == []
    assert report['spec']['target']['params']['target_name'] == 'observed_up'
    assert report['spec']['versions']['vaquum-limen'] == version('vaquum-limen')
    assert len(predictions) == 48
    assert [row[1] for row in predictions] == [size for size in range(25, 49) for _ in range(2)]
    assert len({row[2] for row in predictions}) == 1
    assert all(predictions[index][3] == predictions[index + 1][3] for index in range(0, 48, 2))
    rows = pl.read_parquet(root / report['rows_path'])
    assert rows.height == 48 and rows.filter(pl.col('role') == 'final').height == 24
    assert all(len(row) == 2 for row in rows.filter(pl.col('role') == 'final')['members'].to_list())
    assert read_holdout_report(root, report['attempt_id']) == report
    repeat = _report(frozen, assessment_case)
    assert {key: repeat[key] for key in _REPEAT_KEYS} == {key: report[key] for key in _REPEAT_KEYS}
    assert repeat['exposure']['class'] == 'subsequent'
    majority = copy.deepcopy(cohort)
    majority.aggregation_mode = 'majority_vote'
    majority_report = _report(_freeze(majority), assessment_case)
    majority_rows = pl.read_parquet(root / majority_report['rows_path']).filter(pl.col('role') == 'final')
    assert majority_rows['aggregate_probability'].null_count() == 24
    assert majority_report['classification']['auc']['reason'] == 'no-probability'
    assert majority_report['classification']['accuracy']['value'] is not None
    assert _report(_freeze(trained[1][0]), assessment_case)['state'] == 'completed'
    changed = copy.deepcopy(trained[1][0])
    changed._model.model.booster_.best_iteration = 1
    _freeze(changed)
    changed = copy.deepcopy(trained[0][1])
    changed._model._fitted_calibrator = None
    with pytest.raises(ValueError, match='unfitted calibration'):
        _freeze(changed)
    changed = copy.deepcopy(trained[0][1])
    changed._get_manifest().decoder_lookback = 2
    with pytest.raises(ValueError, match='cached manifest'):
        _freeze(changed)
    changed = copy.deepcopy(trained[0][1])
    changed._fitted_params['_undeclared'] = object()
    with pytest.raises(ValueError, match='undeclared fitted state'):
        _freeze(changed)
    changed = copy.deepcopy(trained[0][1])
    changed._fitted_params['_scaler'].transform = lambda frame: frame
    with pytest.raises(ValueError, match='replaced'):
        _freeze(changed)
    for key, invalid in (('fee_bps', True), ('slip_bps', -1), ('notional_rate', 0), ('execution_lag_bars', 1.0), ('stop_loss_bps', float('nan'))):
        with pytest.raises(ValueError):
            _freeze(trained[0][1], scenarios=[{**_SCENARIO, key: invalid}])
    failures = assessment._Failures()
    def unrelated_error():
        failures.emit(logging.LogRecord('limen.inference.sensor', logging.WARNING, __file__, 0, 'unrelated', (), None))
    thread = Thread(target=unrelated_error)
    thread.start()
    thread.join(1)
    assert not thread.is_alive() and failures.message is None
    failures.emit(logging.LogRecord('limen.inference.sensor', logging.WARNING, __file__, 0, 'owned', (), None))
    assert failures.message == 'owned'


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
    with pytest.raises(ValueError, match='warm-up'):
        _report(_freeze(trained[2][0]), assessment_case)
    with history.connection(root / history._HISTORY) as db:
        warmup_id = db.execute('SELECT id FROM attempts ORDER BY assessment_index DESC LIMIT 1').fetchone()[0]
    recovered = _report(_freeze(trained[2][0], context='2024-01-03'), assessment_case)
    assert warmup_id in [entry['id'] for entry in recovered['prior']['attempts']]
    narrowed = final_holdout_report(_freeze(trained[2][0]), project_root=root, reservation_id=reservation, start='2024-01-06', end='2024-01-07')
    assert narrowed['state'] == 'completed' and narrowed['planned']['start_ns'] == history.utc_day('2024-01-06')
    scale = manifest_core._apply_scaler
    def changed_schema(*args):
        frame, fitted = scale(*args)
        return frame.with_columns(pl.col(pl.Float64).cast(pl.Float32)), fitted
    with patch.object(manifest_core, '_apply_scaler', side_effect=changed_schema), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='prediction dependency changed'):
            _report(frozen, assessment_case)
        assert loader.call_count == 0
    original_tape = assessment._tape
    def changed_after_allocation(*args):
        with patch.object(manifest_core, '_apply_scaler', side_effect=changed_schema):
            return original_tape(*args)
    with patch.object(assessment, '_tape', side_effect=changed_after_allocation), pytest.raises(ValueError, match='schema mismatch'):
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
        for pipe in pipes:
            assert pipe.poll(5)
            ids.append(pipe.recv())
        pipes[1].send('fail')
        assert pipes[1].poll(5) and pipes[1].recv() == 'failed'
        workers[0].kill()
        for worker in workers:
            worker.join(5)
            assert not worker.is_alive()
        with history.connection(root / history._HISTORY) as db:
            entries = [dict(db.execute('SELECT * FROM attempts WHERE id=?', (value,)).fetchone()) for value in ids]
        assert [entry['state'] for entry in entries] == ['started', 'failed']
        ordered = sorted(entries, key=lambda entry: entry['assessment_index'])
        assert ordered[1]['assessment_index'] == ordered[0]['assessment_index'] + 1
        assert ordered[0]['id'] in [entry['id'] for entry in json.loads(ordered[1]['prior_json'])['attempts']]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
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
    replacement = reserve_holdout(migration, '2024-01-05', '2024-01-06', interval_seconds=3600)
    with history.admission(migration):
        declaration = attempts.reservation(migration, replacement, None, None, '2024-01-04')
        standalone = attempts.allocate(migration, declaration, {'context_start_ns': history.utc_day('2024-01-04'), 'start_ns': history.utc_day('2024-01-05'), 'end_ns': history.utc_day('2024-01-06')}, spec='structural', state='structural', scenarios='structural')
    with history.connection(migration / history._HISTORY) as db:
        assert db.execute('SELECT COUNT(*) FROM research').fetchone()[0] == 0
    with pytest.raises(ValueError, match='used holdout'):
        withdraw_holdout(migration, replacement)
    release_holdout(migration, replacement)
    with history.connection(migration / history._HISTORY) as db:
        assert db.execute('SELECT state FROM attempts WHERE id=?', (standalone['id'],)).fetchone()[0] == 'started'
    earlier = reserve_holdout(root, '2024-01-04', '2024-01-05', interval_seconds=3600)
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('pre-access rejection')) as loader:
        with pytest.raises(ValueError, match='research source overlaps final'):
            final_holdout_report(_freeze(trained[3][0], context='2024-01-03'), project_root=root, reservation_id=earlier)
        assert loader.call_count == 0
    coverage = attempts.research_coverage(root, attempts.reservation(root, earlier, None, None, '2024-01-03'))
    assert coverage['overlapping_ids']
    withdraw_holdout(root, earlier)


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
    positive = rows.filter(pl.col('label') == 1)
    assert positive.height > 0
    degenerate = reports.classification(positive['label'].to_list(), positive['aggregate_prediction'].to_list(), positive['aggregate_probability'].to_list())
    assert degenerate['fpr']['reason'] == 'zero-denominator' and degenerate['auc']['reason'] == 'one-class'
    retained = pl.read_parquet(root / report['rows_path'])
    raw = retained.select([pl.col(column).alias(original) for original, column in report['source_column_map'].items()])
    raw = raw.with_columns(pl.col('no_of_trades').alias('row_sha256'), pl.col('maker_ratio').alias('source_row_sha256'))
    tape = [{'members': row['members'], 'prediction': row['aggregate_prediction'], 'probability': row['aggregate_probability'], 'reason': row['aggregate_reason']} for row in rows.iter_rows(named=True)]
    escaped, mapping = reports.retained_rows(raw, 3600 * 1_000_000_000, report['planned']['start_ns'], tape, rows['label'].to_list(), True, rows['scenarios'].to_list())
    assert len(set(mapping.values())) == len(raw.columns)
    assert mapping['row_sha256'] != 'source_row_sha256'
    assert escaped['source_row_sha256'].dtype == pl.String and all(len(value) == 64 for value in escaped['source_row_sha256'])
    assert escaped[mapping['row_sha256']].equals(raw['row_sha256'])
    assert escaped[mapping['source_row_sha256']].equals(raw['source_row_sha256'])
    path = root / report['rows_path']
    original = path.read_bytes()
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash mismatch'):
        read_holdout_report(root, report['attempt_id'])
    path.write_bytes(original)
    release_holdout(root, reservation)
    assert _report(frozen, assessment_case)['exposure']['class'] == 'post-release'


def test_compiled_fitted_components_complete_final_assessment(assessment_case):
    root, _, trained, cohort = assessment_case
    assert len({type(sensor._model) for group in trained for sensor in group}) >= 3
    assert len({type(sensor._model.model) for sensor in members(cohort)}) == 2
    assert set(trained[2][0]._model.model.classes_) == {-1, 1}
    for selected in [sensor for group in trained for sensor in group] + [cohort]:
        frozen = _freeze(selected, context='2024-01-01')
        first = _report(frozen, assessment_case)
        assert first['state'] == 'completed'
        assert read_holdout_report(root, first['attempt_id']) == first
        if selected is cohort or selected is trained[2][0]:
            repeat = _report(frozen, assessment_case)
            assert all(first[key] == repeat[key] for key in _REPEAT_KEYS)
            assert first['attempt_id'] != repeat['attempt_id']
            assert read_holdout_report(root, repeat['attempt_id']) == repeat


@pytest.mark.parametrize('limit', ['equal', 'truncated'])
@pytest.mark.parametrize('calibrated', [True, False])
def test_owned_lightgbm_preserves_effective_iteration(assessment_case, limit, calibrated):
    sensor = copy.deepcopy(assessment_case[2][1][0 if calibrated else 1])
    booster = sensor._model.model.booster_
    assert booster.current_iteration() > 1
    selected = booster.current_iteration() if limit == 'equal' else 1
    booster.best_iteration = selected
    if calibrated:
        sensor._model._fitted_calibrator.estimator.estimator.booster_.best_iteration = selected
    from limen.experiment._holdout_prediction import _probe_output
    from limen.experiment._holdout_dependencies import fitted_witness
    expected = fitted_witness(_probe_output(sensor))
    clone = owned_copy(sensor)
    second = owned_copy(clone)
    assert expected == fitted_witness(_probe_output(clone)) == fitted_witness(_probe_output(second))
    assert clone._model.model.booster_.best_iteration == second._model.model.booster_.best_iteration == selected
    if calibrated:
        assert clone._model._fitted_calibrator.estimator.estimator.booster_.best_iteration == selected
        assert clone._model._fitted_calibrator.estimator.estimator is not clone._model.model
    frozen = _freeze(sensor)
    first, repeat = _report(frozen, assessment_case), _report(frozen, assessment_case)
    assert first['rows_sha256'] == repeat['rows_sha256']
    assert first['state_sha256'] == repeat['state_sha256']


def test_frozen_dependencies_and_invalid_state(assessment_case):
    import limen.scalers.logreg_scaler as scaler_module
    from sklearn.linear_model import LogisticRegression
    from limen.sfd.reference_architecture.logreg_binary import LogRegBinary

    class Lazy(LogRegBinary):
        def predict(self, data):
            self.model.fit(data['x_test'], [0])
            return super().predict(data)
    class Random(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_preds'] = np.random.choice([0, 1], len(data['x_test']))
            return result
    class Mutating(LogRegBinary):
        def predict(self, data):
            self.counter = getattr(self, 'counter', 0) + 1
            return super().predict(data)
    external_state = {'calls': 0}
    class ExternalMutation(LogRegBinary):
        def predict(self, data):
            external_state['calls'] += 1
            return super().predict(data)
    class Nonpreserving(LogRegBinary):
        def __deepcopy__(self, memo):
            result = type(self).__new__(type(self))
            vars(result).update(copy.deepcopy(vars(self), memo))
            result.extra = 'changed selected state'
            return result
    from lightgbm import Booster
    original_booster_predict = Booster.predict
    for selected in assessment_case[2][1]:
        native_handle = _freeze(selected)
        with patch.object(Booster, 'predict', lambda *args, **kwargs: original_booster_predict(*args, **kwargs)), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
            with pytest.raises(ValueError, match='prediction dependency changed'):
                _report(native_handle, assessment_case)
            assert loader.call_count == 0
    altered_native = copy.deepcopy(assessment_case[2][1][1])
    native_booster = altered_native._model.model.booster_
    native_booster.predict = lambda *args, **kwargs: original_booster_predict(native_booster, *args, **kwargs)
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='replaced Booster prediction methods'):
            _freeze(altered_native)
        assert loader.call_count == 0
    frozen = _freeze(assessment_case[2][0][0])
    with patch.dict(scaler_module.SCALING_RULES, dict.fromkeys(('open', 'high', 'low', 'close'), 'none')), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='prediction dependency changed'):
            _report(frozen, assessment_case)
        assert loader.call_count == 0
    import limen.inference.sensor as sensor_module
    with patch.object(sensor_module, '_extract_scalar', lambda value: 0), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='prediction dependency changed'):
            _report(frozen, assessment_case)
        assert loader.call_count == 0
    for kind, reason in ((Lazy, 'attempted fitting'), (Random, 'random dependency'), (Mutating, 'mutated fitted state'), (Nonpreserving, 'copy changed')):
        changed = copy.deepcopy(assessment_case[2][0][0])
        changed._model.__class__ = kind
        with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
            with pytest.raises(ValueError, match=reason):
                _freeze(changed)
            assert loader.call_count == 0
    changed = copy.deepcopy(assessment_case[2][0][0])
    changed._model.__class__ = ExternalMutation
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='mutated prediction dependencies'):
            _freeze(changed)
        assert loader.call_count == 0 and external_state['calls'] > 0
    original_scale = manifest_core._apply_scaler
    def mutating_scale(*args):
        external_state['calls'] += 1
        return original_scale(*args)
    with patch.object(manifest_core, '_apply_scaler', mutating_scale), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='mutated prediction dependencies'):
            _freeze(assessment_case[2][0][0])
        assert loader.call_count == 0
    class CopyDependency(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = later_probabilities(result['_probs'])
            return result
        def __deepcopy__(self, memo):
            result = type(self).__new__(type(self))
            vars(result).update(copy.deepcopy(vars(self), memo))
            _LATE_RULES['offset'] += 0.1
            return result
    copy_dependency = copy.deepcopy(assessment_case[2][0][0])
    copy_dependency._model.__class__ = CopyDependency
    with patch.dict(_LATE_RULES), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='copy changed prediction dependencies'):
            _freeze(copy_dependency)
        assert loader.call_count == 0
    defaults, closure = {'offset': 0.0}, {'offset': 0.0}
    def correction(values=defaults):
        return values['offset'] + closure['offset']
    class Configured(LogRegBinary):
        rules: ClassVar[dict[str, float]] = {'offset': 0.0}
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = np.clip(result['_probs'] + correction() + self.rules['offset'], 0, 1)
            return result
    configured = copy.deepcopy(assessment_case[2][0][0])
    configured._model.__class__ = Configured
    sealed = _freeze(configured)
    for dependency in (defaults, closure, Configured.rules):
        with patch.dict(dependency, {'offset': 0.1}), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
            with pytest.raises(ValueError, match='prediction dependency changed'):
                _report(sealed, assessment_case)
            assert loader.call_count == 0
    with patch.object(Configured, 'predict', LogRegBinary.predict), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='prediction dependency changed'):
            _report(sealed, assessment_case)
        assert loader.call_count == 0
    slot_name = 'offset'
    class SlotOffset:
        __slots__ = slot_name
        def __init__(self, offset):
            self.offset = offset
        def adjust(self, values):
            return np.clip(values + self.offset, 0, 1)
    class Adjusted(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = self.policy.adjust(result['_probs'])
            return result
    slotted = copy.deepcopy(assessment_case[2][0][0])
    slotted._model.__class__ = Adjusted
    slotted._model.policy = SlotOffset(slotted._model._calibration_threshold)
    stable = _freeze(slotted)
    first = _report(stable, assessment_case)
    slotted._model.policy.offset += 0.1
    assert _report(stable, assessment_case)['rows_sha256'] == first['rows_sha256']
    class NonpreservingSlot(SlotOffset):
        def __deepcopy__(self, memo):
            return type(self)(self.offset + 0.1)
    slotted._model.policy = NonpreservingSlot(slotted._model._calibration_threshold)
    with pytest.raises(ValueError, match='copy changed'):
        _freeze(slotted)
    class Late(LogRegBinary):
        def predict(self, data):
            if not np.array_equal(np.asarray(data['x_test']), self.research_x):
                if self.refit == 'callback':
                    self.prediction_calibration_config.calibration_func()
                elif self.refit == 'wrapped':
                    self.prediction_calibration_config.threshold_func()
                elif self.refit:
                    self.model.fit(data['x_test'], [0])
                else:
                    self.counter += 1
            return super().predict(data)
    for refit in (False, True, 'callback', 'wrapped'):
        late = copy.deepcopy(assessment_case[2][0][0])
        prepared, _ = late._get_manifest().sensor_input_prep(late._fitted_params['_holdout_probe'], late._fitted_params, late._round_params)
        columns = [name for name in prepared.columns if name not in ('datetime', '__trade_available_at_ns__')]
        late._model.__class__ = Late
        late._model.research_x = np.asarray(prepared.select(columns).tail(1), dtype=float)
        late._model.counter, late._model.refit = 0, refit
        handle = _freeze(late)
        with pytest.raises(ValueError, match='attempted fitting' if refit else 'changed fitted state') as failed:
            _report(handle, assessment_case)
        with history.connection(assessment_case[0] / history._HISTORY) as db:
            record = db.execute('SELECT * FROM attempts ORDER BY assessment_index DESC LIMIT 1').fetchone()
            assert record['state'] == 'failed' and record['id'] in str(failed.value)
            assert json.loads(record['error_json'])['stage'] == 'predict'
        with pytest.raises(ValueError, match='not completed'):
            read_holdout_report(assessment_case[0], record['id'])
    class NativeRandom(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            if not np.array_equal(np.asarray(data['x_test']), self.research_x):
                result['_probs'] = np.clip(result['_probs'] + self.rng.normal(), 0, 1)
            return result
    native_random = copy.deepcopy(assessment_case[2][0][0])
    native_random._model.__class__ = NativeRandom
    native_random._model.research_x = late._model.research_x.copy()
    native_random._model.rng = np.random.default_rng(42)
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='random dependency'):
            _freeze(native_random)
        assert loader.call_count == 0
    class Branch(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            if not np.array_equal(np.asarray(data['x_test']), self.research_x):
                if self.helper_class == 'local':
                    class LocalOffset:
                        def __init__(self):
                            self.offset = _LATE_RULES['offset']
                        def adjust(self, values):
                            return np.clip(values + self.offset, 0, 1)
                    result['_probs'] = LocalOffset().adjust(result['_probs'])
                elif self.helper_class == 'registry':
                    result['_probs'] = _OFFSET_REGISTRY['offset']().adjust(result['_probs'])
                else:
                    result['_probs'] = InferenceOffset().adjust(result['_probs']) if self.helper_class else later_probabilities(result['_probs'])
            return result
    for helper_class in (False, True, 'local', 'registry'):
        branched = copy.deepcopy(assessment_case[2][0][0])
        branched._model.__class__ = Branch
        branched._model.research_x = late._model.research_x.copy()
        branched._model.helper_class = helper_class
        branch_handle = _freeze(branched)
        with patch.dict(_LATE_RULES, {'offset': 0.1}), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
            with pytest.raises(ValueError, match='prediction dependency changed'):
                _report(branch_handle, assessment_case)
            assert loader.call_count == 0
        if helper_class is True or helper_class == 'registry':
            with patch.object(InferenceOffset, '__init__', _OFFSET_REGISTRY['alternate']), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
                with pytest.raises(ValueError, match='prediction dependency changed'):
                    _report(branch_handle, assessment_case)
                assert loader.call_count == 0
            with patch.object(InferenceOffset, 'offset', 0.2, create=True):
                descriptor_handle = _freeze(branched)
                with patch.object(InferenceOffset, 'adjust', classmethod(InferenceOffset.adjust)), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
                    with pytest.raises(ValueError, match='prediction dependency changed'):
                        _report(descriptor_handle, assessment_case)
                    assert loader.call_count == 0
    class Imported(LogRegBinary):
        def predict(self, data):
            import limen.calibration
            result = super().predict(data)
            result['_probs'] = limen.calibration.f17_later_probabilities(result['_probs'])
            return result
    class Aliased(LogRegBinary):
        def predict(self, data):
            import limen.calibration as imported
            helpers = imported
            result = super().predict(data)
            result['_probs'] = helpers.f17_later_probabilities(result['_probs'])
            return result
    for implementation in (Imported, Aliased):
        imported = copy.deepcopy(assessment_case[2][0][0])
        imported._model.__class__ = implementation
        with patch('limen.calibration.f17_later_probabilities', later_probabilities, create=True):
            imported_handle = _freeze(imported)
            with patch.dict(_LATE_RULES, {'offset': 0.1}), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
                with pytest.raises(ValueError, match='prediction dependency changed'):
                    _report(imported_handle, assessment_case)
                assert loader.call_count == 0
    class Keyed(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = next(iter(self.adjustments))(result['_probs'])
            return result
    keyed = copy.deepcopy(assessment_case[2][0][0])
    keyed._model.__class__ = Keyed
    keyed._model.adjustments = {later_probabilities: True}
    keyed_handle = _freeze(keyed)
    with patch.dict(_LATE_RULES, {'offset': 0.1}), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='prediction dependency changed'):
            _report(keyed_handle, assessment_case)
        assert loader.call_count == 0
    for container in (set, frozenset, lambda values: np.asarray(values, dtype=object)):
        keyed._model.adjustments = container([later_probabilities])
        container_handle = _freeze(keyed)
        with patch.dict(_LATE_RULES, {'offset': 0.1}), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
            with pytest.raises(ValueError, match='prediction dependency changed'):
                _report(container_handle, assessment_case)
            assert loader.call_count == 0
    class CallbackState(Keyed):
        def predict(self, data):
            result = super().predict(data)
            callback = next(iter(self.adjustments))
            result['_probs'] = np.clip(result['_probs'] + callback.offset, 0, 1)
            return result
    keyed._model.__class__ = CallbackState
    keyed._model.adjustments = {identity_probabilities: True}
    with patch.object(identity_probabilities, 'offset', 0.0, create=True):
        callback_handle = _freeze(keyed)
        with patch.object(identity_probabilities, 'offset', 0.1), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
            with pytest.raises(ValueError, match='prediction dependency changed'):
                _report(callback_handle, assessment_case)
            assert loader.call_count == 0
    class MappingPolicy(dict):
        def __init__(self, offset):
            super().__init__()
            self.offset = offset
        def adjust(self, values):
            return np.clip(values + self.offset, 0, 1)
    mapped = copy.deepcopy(assessment_case[2][0][0])
    mapped._model.__class__ = Adjusted
    mapped._model.policy = MappingPolicy(mapped._model._calibration_threshold)
    mapping_copy = owned_copy(mapped)
    _freeze(mapping_copy)
    expected_offset = mapping_copy._model.policy.offset
    mapped._model.policy.offset += 0.1
    assert mapping_copy._model.policy.offset == expected_offset
    class NonpreservingMapping(MappingPolicy):
        def __deepcopy__(self, memo):
            return type(self)(self.offset + 0.1)
    mapped._model.policy = NonpreservingMapping(expected_offset)
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='copy changed'):
            _freeze(mapped)
        assert loader.call_count == 0
    class HiddenMapping(dict):
        def items(self):
            return ()
        def adjust(self, values):
            return np.clip(values + self['offset'], 0, 1)
        def __deepcopy__(self, memo):
            return type(self)((key, copy.deepcopy(value, memo)) for key, value in dict.items(self))
    class NonpreservingHidden(HiddenMapping):
        def __deepcopy__(self, memo):
            clone = super().__deepcopy__(memo)
            clone['offset'] += 0.1
            return clone
    mapped._model.policy = HiddenMapping(offset=expected_offset)
    _freeze(mapped)
    mapped._model.policy = NonpreservingHidden(offset=expected_offset)
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='copy changed'):
            _freeze(mapped)
        assert loader.call_count == 0
    class HiddenSequence(list):
        def __iter__(self):
            return iter(())
        def adjust(self, values):
            return np.clip(values + list.__getitem__(self, 0), 0, 1)
        def __deepcopy__(self, memo):
            return type(self)(copy.deepcopy(value, memo) for value in list.__iter__(self))
    class NonpreservingSequence(HiddenSequence):
        def __deepcopy__(self, memo):
            clone = super().__deepcopy__(memo)
            clone[0] += 0.1
            return clone
    mapped._model.policy = HiddenSequence([expected_offset])
    _freeze(mapped)
    mapped._model.policy = NonpreservingSequence([expected_offset])
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='copy changed'):
            _freeze(mapped)
        assert loader.call_count == 0
    class HiddenArray(np.ndarray):
        def tolist(self):
            return []
        def adjust(self, values):
            return np.clip(values + np.ndarray.__getitem__(self, 0), 0, 1)
        def __deepcopy__(self, memo):
            clone = np.ndarray.copy(self)
            memo[id(self)] = clone
            return clone
    class NonpreservingArray(HiddenArray):
        def __deepcopy__(self, memo):
            clone = super().__deepcopy__(memo)
            clone[0] += 0.1
            return clone
    mapped._model.policy = np.asarray([expected_offset], dtype=object).view(HiddenArray)
    _freeze(mapped)
    mapped._model.policy = np.asarray([expected_offset], dtype=object).view(NonpreservingArray)
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='copy changed'):
            _freeze(mapped)
        assert loader.call_count == 0
    class HiddenNumeric(HiddenArray):
        def __reduce_ex__(self, protocol):
            return int, (0,)
    class NonpreservingNumeric(HiddenNumeric):
        def __deepcopy__(self, memo):
            clone = super().__deepcopy__(memo)
            clone[0] += 0.1
            return clone
    mapped._model.policy = np.asarray([expected_offset], dtype=float).view(HiddenNumeric)
    _freeze(mapped)
    mapped._model.policy = np.asarray([expected_offset], dtype=float).view(NonpreservingNumeric)
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='copy changed'):
            _freeze(mapped)
        assert loader.call_count == 0
    class HiddenAttributes:
        def __init__(self, offset):
            self.offset = offset
        def __getattribute__(self, name):
            return {} if name == '__dict__' else object.__getattribute__(self, name)
        def adjust(self, values):
            return np.clip(values + self.offset, 0, 1)
        def __deepcopy__(self, memo):
            return type(self)(self.offset)
    class NonpreservingAttributes(HiddenAttributes):
        def __deepcopy__(self, memo):
            return type(self)(self.offset + 0.1)
    mapped._model.policy = HiddenAttributes(expected_offset)
    _freeze(mapped)
    mapped._model.policy = NonpreservingAttributes(expected_offset)
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='copy changed'):
            _freeze(mapped)
        assert loader.call_count == 0
    class HiddenSlots:
        __slots__ = ('offset', 'optional')
        def __init__(self, offset):
            self.offset = offset
        def __getattribute__(self, name):
            if name == 'offset':
                raise AttributeError('hidden learned state')
            return object.__getattribute__(self, name)
        def adjust(self, values):
            return np.clip(values + object.__getattribute__(self, 'offset'), 0, 1)
        def __deepcopy__(self, memo):
            return type(self)(object.__getattribute__(self, 'offset'))
    class NonpreservingSlots(HiddenSlots):
        def __deepcopy__(self, memo):
            return type(self)(object.__getattribute__(self, 'offset') + 0.1)
    mapped._model.policy = HiddenSlots(expected_offset)
    _freeze(mapped)
    mapped._model.policy = NonpreservingSlots(expected_offset)
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='copy changed'):
            _freeze(mapped)
        assert loader.call_count == 0
    class IteratorSlots:
        __slots__ = iter(('offset',))
        def __init__(self, offset):
            self.offset = offset
        def adjust(self, values):
            return np.clip(values + self.offset, 0, 1)
        def __deepcopy__(self, memo):
            return type(self)(self.offset)
    class NonpreservingIteratorSlots(IteratorSlots):
        def __deepcopy__(self, memo):
            return type(self)(self.offset + 0.1)
    mapped._model.policy = IteratorSlots(expected_offset)
    _freeze(mapped)
    mapped._model.policy = NonpreservingIteratorSlots(expected_offset)
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='copy changed'):
            _freeze(mapped)
        assert loader.call_count == 0
    class AdjustProperty(property):
        def __get__(self, instance, owner=None):
            return lambda values: np.clip(values + self.offset + _LATE_RULES['offset'], 0, 1)
    class PropertyModel(LogRegBinary):
        @AdjustProperty
        def adjust(self):
            return identity_probabilities
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = self.adjust(result['_probs'])
            return result
    descriptor = vars(PropertyModel)['adjust']
    descriptor.offset = expected_offset
    described = copy.deepcopy(assessment_case[2][0][0])
    described._model.__class__ = PropertyModel
    property_handle = _freeze(described)
    with patch.dict(_LATE_RULES, {'offset': 0.1}), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='prediction dependency changed'):
            _report(property_handle, assessment_case)
        assert loader.call_count == 0
    with patch.object(descriptor, 'offset', expected_offset + 0.1), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='prediction dependency changed'):
            _report(property_handle, assessment_case)
        assert loader.call_count == 0
    class AliasPrediction(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = fit_calibrator(result['_probs'])
            return result
    aliased_helper = copy.deepcopy(assessment_case[2][0][0])
    aliased_helper._model.__class__ = AliasPrediction
    alias_handle = _freeze(aliased_helper)
    with patch.dict(_LATE_RULES, {'offset': 0.1}), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='prediction dependency changed'):
            _report(alias_handle, assessment_case)
        assert loader.call_count == 0
    class AssessmentCopy(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = later_probabilities(result['_probs'])
            return result
        def __deepcopy__(self, memo):
            clone = type(self).__new__(type(self))
            vars(clone).update(copy.deepcopy(vars(self), memo))
            if _COPY_AFTER_FREEZE:
                _LATE_RULES['offset'] += 0.1
            return clone
    copying = copy.deepcopy(assessment_case[2][0][0])
    copying._model.__class__ = AssessmentCopy
    copying_handle = _freeze(copying)
    with patch.dict(globals(), {'_COPY_AFTER_FREEZE': True}), patch.dict(_LATE_RULES), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='copy changed prediction dependencies before observation access'):
            _report(copying_handle, assessment_case)
        assert loader.call_count == 0
    class Relative(LogRegBinary):
        def predict(self, data):
            from . import test_holdout_research as helpers
            result = super().predict(data)
            result['_probs'] = helpers.f17_later_probabilities(result['_probs'])
            return result
    relative = copy.deepcopy(assessment_case[2][0][0])
    relative._model.__class__ = Relative
    with patch('tests.test_holdout_research.f17_later_probabilities', later_probabilities, create=True):
        relative_handle = _freeze(relative)
        with patch.dict(_LATE_RULES, {'offset': 0.1}), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
            with pytest.raises(ValueError, match='prediction dependency changed'):
                _report(relative_handle, assessment_case)
            assert loader.call_count == 0
    class CollidingOffset(float):
        def __hash__(self):
            return 0
    class SetChoice(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = np.clip(result['_probs'] + next(iter(self.policy)), 0, 1)
            return result
        def __deepcopy__(self, memo):
            clone = type(self).__new__(type(self))
            memo[id(self)] = clone
            vars(clone).update(copy.deepcopy(vars(self), memo))
            clone.policy = type(self.policy)(reversed(tuple(self.policy)))
            return clone
    reordered = copy.deepcopy(assessment_case[2][0][0])
    reordered._model.__class__ = SetChoice
    for container in (set, frozenset):
        reordered._model.policy = container((CollidingOffset(expected_offset), CollidingOffset(expected_offset + 0.1)))
        assert tuple(copy.deepcopy(reordered._model).policy) == tuple(reversed(tuple(reordered._model.policy)))
        with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
            with pytest.raises(ValueError, match='copy changed'):
                _freeze(reordered)
            assert loader.call_count == 0
    class FloatPolicy(float):
        def adjust(self, values):
            return np.clip(values + float(self), 0, 1)
    class NonpreservingFloat(FloatPolicy):
        def __deepcopy__(self, memo):
            return type(self)(float(self) + 0.1)
    mapped._model.policy = FloatPolicy(expected_offset)
    _freeze(mapped)
    mapped._model.policy = NonpreservingFloat(expected_offset)
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='copy changed'):
            _freeze(mapped)
        assert loader.call_count == 0
    class BytePolicy(bytearray):
        def adjust(self, values):
            return np.clip(values + np.frombuffer(self, dtype=np.float64)[0], 0, 1)
    class NonpreservingBytes(BytePolicy):
        def __deepcopy__(self, memo):
            return type(self)((np.frombuffer(self, dtype=np.float64) + 0.1).tobytes())
    learned_bytes = np.float64(expected_offset).tobytes()
    mapped._model.policy = BytePolicy(learned_bytes)
    _freeze(mapped)
    mapped._model.policy = NonpreservingBytes(learned_bytes)
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='copy changed'):
            _freeze(mapped)
        assert loader.call_count == 0
    from collections import deque
    class DequePolicy(deque):
        def adjust(self, values):
            return np.clip(values + self[0], 0, 1)
    class NonpreservingDeque(DequePolicy):
        def __deepcopy__(self, memo):
            return type(self)((self[0] + 0.1,), maxlen=self.maxlen)
    mapped._model.policy = DequePolicy((expected_offset,), maxlen=1)
    _freeze(mapped)
    mapped._model.policy = NonpreservingDeque((expected_offset,), maxlen=1)
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='copy changed'):
            _freeze(mapped)
        assert loader.call_count == 0
    from array import array
    class OpaqueNativePolicy(array):
        def adjust(self, values):
            return np.clip(values + self[0], 0, 1)
    mapped._model.policy = OpaqueNativePolicy('d', [expected_offset])
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='opaque native prediction state'):
            _freeze(mapped)
        assert loader.call_count == 0
    class ModuleLookup(LogRegBinary):
        def predict(self, data):
            import limen.calibration as helpers
            result = super().predict(data)
            name = 'f17_later_probabilities'
            result['_probs'] = getattr(helpers, name)(result['_probs'])
            return result
    class BranchImports(LogRegBinary):
        def predict(self, data):
            if not np.array_equal(np.asarray(data['x_test']), self.research_x):
                import limen.calibration as helpers
            else:
                from . import test_holdout_research as helpers
            result = super().predict(data)
            result['_probs'] = helpers.f17_later_probabilities(result['_probs'])
            return result
    class Reflective(LogRegBinary):
        def adjust(self, values):
            return later_probabilities(values)
        def predict(self, data):
            result = super().predict(data)
            method_name = 'adjust'
            result['_probs'] = getattr(self, method_name)(result['_probs'])
            return result
    class Derived(Reflective):
        def predict(self, data):
            return super().predict(data)
    class ClassBody(LogRegBinary):
        def predict(self, data):
            class Local:
                offset = _LATE_RULES['offset']
            result = super().predict(data)
            result['_probs'] = np.clip(result['_probs'] + Local.offset, 0, 1)
            return result
    class Descriptor:
        def __get__(self, instance, owner=None):
            return later_probabilities
        def __set__(self, instance, value):
            raise AssertionError('Read-only prediction descriptor')
    class Described(LogRegBinary):
        adjust = Descriptor()
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = self.adjust(result['_probs'])
            return result
    from collections import defaultdict
    class Factory(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = self.policy.default_factory(result['_probs'])
            return result
    class PartialState(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = self.policy.adjust(result['_probs'])
            return result
    class PartialPolicy(partial):
        def __call__(self, values):
            return later_probabilities(super().__call__(values))
    class SlottedPartial(PartialPolicy):
        __slots__ = ('offset',)
        def __call__(self, values):
            return np.clip(super().__call__(values) + self.offset, 0, 1)
        def __deepcopy__(self, memo):
            clone = type(self)(self.func, *self.args, **self.keywords)
            memo[id(self)] = clone
            vars(clone).update(copy.deepcopy(vars(self), memo))
            clone.offset = self.offset
            return clone
    class NonpreservingSlotted(SlottedPartial):
        def __deepcopy__(self, memo):
            clone = super().__deepcopy__(memo)
            clone.offset += 0.1
            return clone
    slotted_partial = SlottedPartial(identity_probabilities)
    slotted_partial.offset = expected_offset
    slotted_selection = copy.deepcopy(assessment_case[2][0][0])
    slotted_selection._model.__class__ = PartialState
    slotted_selection._model.policy = slotted_partial
    slotted_selection._model.policy.adjust = slotted_partial
    _freeze(slotted_selection)
    slotted_selection._model.policy = NonpreservingSlotted(identity_probabilities)
    slotted_selection._model.policy.offset = expected_offset
    slotted_selection._model.policy.adjust = slotted_selection._model.policy
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='copy changed'):
            _freeze(slotted_selection)
        assert loader.call_count == 0
    class NonpreservingPartial(PartialPolicy):
        def __deepcopy__(self, memo):
            return partial(self.func, *self.args, **self.keywords)
    class ProbabilityMeta(type):
        def __call__(cls, values):
            return later_probabilities(values)
    class ClassPolicy(metaclass=ProbabilityMeta):
        selected = True
    class CallablePolicy(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = self.policy(result['_probs'])
            return result
    class ArithmeticPolicy:
        def __add__(self, values):
            return later_probabilities(values)
    class Arithmetic(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = self.policy + result['_probs']
            return result
    class NativeFloatPolicy(np.float64):
        def __deepcopy__(self, memo):
            return type(self)(float(self))
        def adjust(self, values):
            return later_probabilities(np.clip(values + float(self), 0, 1))
    class IteratorPolicy:
        def __next__(self):
            return _LATE_RULES['offset']
    class Iterated(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = np.clip(result['_probs'] + next(self.policy), 0, 1)
            return result
    class UnaryPolicy:
        def __neg__(self):
            return _LATE_RULES['offset']
    class Negated(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = np.clip(result['_probs'] + (-self.policy), 0, 1)
            return result
    class RoundPolicy:
        def __round__(self, ndigits=None):
            return round(_LATE_RULES['offset'], ndigits)
    class Rounded(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = np.clip(result['_probs'] + round(self.policy, 2), 0, 1)
            return result
    class ContextPolicy:
        def __enter__(self):
            return _LATE_RULES['offset']
        def __exit__(self, *args):
            return False
    class Contextual(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            with self.policy as offset:
                result['_probs'] = np.clip(result['_probs'] + offset, 0, 1)
            return result
    class MissingPolicy(dict):
        def __missing__(self, key):
            return _LATE_RULES['offset']
    class MissingLookup(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = np.clip(result['_probs'] + self.policy['offset'], 0, 1)
            return result
    class ReversePolicy:
        def __reversed__(self):
            return iter((_LATE_RULES['offset'],))
    class ReversedLookup(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = np.clip(result['_probs'] + next(reversed(self.policy)), 0, 1)
            return result
    class ComparisonPolicy:
        def __gt__(self, other):
            return _LATE_RULES['offset'] > other
    class Compared(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = np.clip(result['_probs'] + (0.1 if self.policy > 0 else 0), 0, 1)
            return result
    class ConversionPolicy:
        def __float__(self):
            return _LATE_RULES['offset']
    class Converted(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = np.clip(result['_probs'] + float(self.policy), 0, 1)
            return result
    for provider in (PartialPolicy(identity_probabilities), ClassPolicy, CalibratorFactory(later_probabilities), ArithmeticPolicy(), ComparisonPolicy(), NativeFloatPolicy(expected_offset), IteratorPolicy(), UnaryPolicy(), RoundPolicy(), ContextPolicy(), MissingPolicy(), ReversePolicy(), ConversionPolicy()):
        callable_selection = copy.deepcopy(assessment_case[2][0][0])
        callable_selection._model.__class__ = Arithmetic if isinstance(provider, ArithmeticPolicy) else Compared if isinstance(provider, ComparisonPolicy) else PartialState if isinstance(provider, NativeFloatPolicy) else Iterated if isinstance(provider, IteratorPolicy) else Negated if isinstance(provider, UnaryPolicy) else Rounded if isinstance(provider, RoundPolicy) else Contextual if isinstance(provider, ContextPolicy) else MissingLookup if isinstance(provider, MissingPolicy) else ReversedLookup if isinstance(provider, ReversePolicy) else Converted if isinstance(provider, ConversionPolicy) else CallablePolicy
        callable_selection._model.policy = provider
        callable_handle = _freeze(callable_selection)
        with patch.dict(_LATE_RULES, {'offset': 0.1}), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
            with pytest.raises(ValueError, match='prediction dependency changed'):
                _report(callable_handle, assessment_case)
            assert loader.call_count == 0
    callable_selection._model.policy = NonpreservingPartial(identity_probabilities)
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='copy changed'):
            _freeze(callable_selection)
        assert loader.call_count == 0
    class ModuleAttribute(LogRegBinary):
        helpers = manifest_core
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = self.helpers.f17_later_probabilities(result['_probs'])
            return result
    def module_probabilities(module, values):
        return module.f17_later_probabilities(values)
    class ModuleArgument(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = module_probabilities(manifest_core, result['_probs'])
            return result
    class DynamicLookup(LogRegBinary):
        def adjust_late(self, values):
            return later_probabilities(values)
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = getattr(self, self.method_name)(result['_probs'])
            return result
    class DynamicModuleLookup(LogRegBinary):
        helpers = manifest_core
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = getattr(self.helpers, self.helper_name)(result['_probs'])
            return result
    class BoundLookup(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = self.policy('adjust')(result['_probs'])
            return result
    for implementation in (ModuleLookup, BranchImports, Reflective, Derived, ClassBody, Described, Factory, PartialState, ModuleAttribute, ModuleArgument, DynamicLookup, DynamicModuleLookup, BoundLookup):
        selection = copy.deepcopy(assessment_case[2][0][0])
        selection._model.__class__ = implementation
        selection._model.research_x = late._model.research_x.copy()
        if implementation is DynamicLookup:
            selection._model.method_name = 'adjust_late'
        if implementation is DynamicModuleLookup:
            selection._model.helper_name = 'f17_later_probabilities'
        if implementation is BoundLookup:
            selection._model.policy = {'adjust': later_probabilities}.get
        if implementation is Factory:
            selection._model.policy = defaultdict(later_probabilities)
        if implementation is PartialState:
            selection._model.policy = partial(identity_probabilities)
            selection._model.policy.adjust = later_probabilities
        with patch('limen.calibration.f17_later_probabilities', later_probabilities, create=True), patch('tests.test_holdout_research.f17_later_probabilities', identity_probabilities, create=True), patch.object(manifest_core, 'f17_later_probabilities', later_probabilities, create=True):
            selected = _freeze(selection)
            with patch.dict(_LATE_RULES, {'offset': 0.1}), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
                with pytest.raises(ValueError, match='prediction dependency changed'):
                    _report(selected, assessment_case)
                assert loader.call_count == 0
    from limen.experiment.manifest_core import CalibrationConfig
    class FittingProvider:
        def __init__(self, config):
            self.prediction_calibration_config = config
        def adjust(self, values):
            self.prediction_calibration_config.threshold_func(values)
            return values
    from limen.experiment._holdout_prediction import readonly_prediction
    class NumericCallbacks(np.ndarray):
        prediction_calibration_config = None
    callback_calls = []
    def nested_threshold(*args):
        callback_calls.append('threshold')
    nested = copy.deepcopy(assessment_case[2][0][0])
    nested._model.policy = nested._model.model.coef_.copy().view(NumericCallbacks)
    nested._model.policy.prediction_calibration_config = CalibrationConfig(calibration_func=None, threshold_func=nested_threshold)
    with pytest.raises(ValueError, match='attempted fitting'), readonly_prediction([nested]):
        nested._model.policy.prediction_calibration_config.threshold_func()
    assert callback_calls == []
    class ScalarCallbacks(float):
        prediction_calibration_config = None
    nested._model.policy = ScalarCallbacks(expected_offset)
    nested._model.policy.prediction_calibration_config = CalibrationConfig(calibration_func=None, threshold_func=nested_threshold)
    with pytest.raises(ValueError, match='attempted fitting'), readonly_prediction([nested]):
        nested._model.policy.prediction_calibration_config.threshold_func()
    assert callback_calls == []
    class StaticThreshold:
        __call__ = staticmethod(nested_threshold)
    class ClassThreshold:
        @classmethod
        def __call__(cls, *args):
            nested_threshold(*args)
    for callback in (partial(StaticThreshold()), partial(ClassThreshold())):
        nested._model.policy.prediction_calibration_config.threshold_func = callback
        with pytest.raises(ValueError, match='attempted fitting'), readonly_prediction([nested]):
            callback()
        assert callback_calls == []
    configured_provider = FittingProvider(CalibrationConfig(calibration_func=None, threshold_func=nested_threshold))
    def default_provider(provider=configured_provider):
        provider.prediction_calibration_config.threshold_func()
    def closed_provider(provider):
        def invoke():
            provider.prediction_calibration_config.threshold_func()
        return invoke
    class BoundProvider:
        prediction_calibration_config = configured_provider.prediction_calibration_config
        def invoke(self):
            self.prediction_calibration_config.threshold_func()
    for provider in (default_provider, closed_provider(configured_provider), BoundProvider().invoke):
        nested._model.policy = provider
        with pytest.raises(ValueError, match='attempted fitting'), readonly_prediction([nested]):
            provider()
        assert callback_calls == []
    nested._model.policy = FittingProvider(CalibrationConfig(threshold_func=partial(max)))
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='callbacks without observable Python call protection'), readonly_prediction([nested]):
            nested._model.policy.adjust(nested._model.model.coef_.ravel())
        assert loader.call_count == 0
    class GlobalCallbackModel(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            if not np.array_equal(np.asarray(data['x_test']), self.research_x):
                result['_probs'] = _GLOBAL_FITTED_PROVIDER.adjust(result['_probs'])
            return result
    globally_configured = copy.deepcopy(assessment_case[2][0][0])
    globally_configured._model.__class__ = GlobalCallbackModel
    globally_configured._model.research_x = late._model.research_x.copy()
    with patch.dict(globals(), {'_GLOBAL_FITTED_PROVIDER': configured_provider}):
        global_handle = _freeze(globally_configured)
        with pytest.raises(ValueError, match='attempted fitting') as global_failure:
            _report(global_handle, assessment_case)
        assert callback_calls == []
        with history.connection(assessment_case[0] / history._HISTORY) as db:
            record = db.execute('SELECT * FROM attempts ORDER BY assessment_index DESC LIMIT 1').fetchone()
            assert record['state'] == 'failed' and record['id'] in str(global_failure.value)
            assert json.loads(record['error_json'])['stage'] == 'predict'
    class ThreadedCallbackModel(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            if not np.array_equal(np.asarray(data['x_test']), self.research_x):
                worker = Thread(target=self.policy.prediction_calibration_config.threshold_func)
                worker.start()
                worker.join()
            return result
    threaded = copy.deepcopy(assessment_case[2][0][0])
    threaded._model.__class__ = ThreadedCallbackModel
    threaded._model.research_x = late._model.research_x.copy()
    threaded._model.policy = configured_provider
    threaded_handle = _freeze(threaded)
    with pytest.raises(ValueError, match='attempted thread dispatch') as thread_failure:
        _report(threaded_handle, assessment_case)
    assert callback_calls == []
    with history.connection(assessment_case[0] / history._HISTORY) as db:
        record = db.execute('SELECT * FROM attempts ORDER BY assessment_index DESC LIMIT 1').fetchone()
        assert record['state'] == 'failed' and record['id'] in str(thread_failure.value)
        assert json.loads(record['error_json'])['stage'] == 'predict'
    nested._model.policy = configured_provider
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(bool, True).result() is True
        with pytest.raises(ValueError, match='attempted thread dispatch'), readonly_prediction([nested]):
            pool.submit(nested_threshold).result()
        assert callback_calls == []
    def descriptor_get(self, instance, owner=None):
        return lambda values: np.clip(later_probabilities(values) + self.offset, 0, 1)
    class StaticDescriptor(staticmethod):
        __get__ = descriptor_get
    class ClassDescriptor(classmethod):
        __get__ = descriptor_get
    for descriptor_type in (StaticDescriptor, ClassDescriptor):
        class DescriptorModel(LogRegBinary):
            adjust = descriptor_type(identity_probabilities)
            def predict(self, data):
                result = super().predict(data)
                result['_probs'] = self.adjust(result['_probs'])
                return result
        descriptor = vars(DescriptorModel)['adjust']
        descriptor.offset = expected_offset
        described = copy.deepcopy(assessment_case[2][0][0])
        described._model.__class__ = DescriptorModel
        descriptor_handle = _freeze(described)
        with patch.dict(_LATE_RULES, {'offset': 0.1}), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
            with pytest.raises(ValueError, match='prediction dependency changed'):
                _report(descriptor_handle, assessment_case)
            assert loader.call_count == 0
        with patch.object(descriptor, 'offset', expected_offset + 0.1), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
            with pytest.raises(ValueError, match='prediction dependency changed'):
                _report(descriptor_handle, assessment_case)
            assert loader.call_count == 0
    class ExternalEntropy(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            if not np.array_equal(np.asarray(data['x_test']), self.research_x):
                result['_probs'] = np.clip(result['_probs'] + int.from_bytes(os.urandom(1), 'big') / 255, 0, 1)
            return result
    entropic = copy.deepcopy(assessment_case[2][0][0])
    entropic._model.__class__ = ExternalEntropy
    entropic._model.research_x = late._model.research_x.copy()
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='random dependency'):
            _freeze(entropic)
        assert loader.call_count == 0
    with pytest.raises(ValueError, match='unowned randomness'), readonly_prediction([nested]):
        os.urandom(1)
    class NativePartial(partial):
        __call__ = staticmethod(max)
    unsafe_callback = copy.deepcopy(assessment_case[2][0][0])
    unsafe_callback._model.callback_provider = FittingProvider(CalibrationConfig(calibration_func=None, threshold_func=NativePartial(identity_probabilities)))
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='callbacks without observable Python call protection'):
            _freeze(unsafe_callback)
        assert loader.call_count == 0
    class NativeThreshold:
        __call__ = staticmethod(max)
    class DynamicStaticMethod(staticmethod):
        def __get__(self, instance, owner=None):
            return max
    class DynamicThreshold:
        __call__ = DynamicStaticMethod(identity_probabilities)
    class DynamicClassMethod(classmethod):
        def __get__(self, instance, owner=None):
            return max
    class DynamicClassThreshold:
        __call__ = DynamicClassMethod(identity_probabilities)
    class DynamicProperty(property):
        def __get__(self, instance, owner=None):
            return max
    class DynamicPropertyThreshold:
        __call__ = DynamicProperty(lambda self: identity_probabilities)
    misleading = NativeThreshold()
    misleading.__wrapped__ = identity_probabilities
    for callback in (misleading, DynamicThreshold(), DynamicClassThreshold(), DynamicPropertyThreshold()):
        unsafe_callback._model.callback_provider.prediction_calibration_config.threshold_func = callback
        with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
            with pytest.raises(ValueError, match='callbacks without observable Python call protection'):
                _freeze(unsafe_callback)
            assert loader.call_count == 0
    class ReadonlyCallable:
        def __call__(self, values):
            return identity_probabilities(values)
    class UsesReadonlyCallable(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = self.policy(result['_probs'])
            return result
    distinct = copy.deepcopy(assessment_case[2][0][0])
    distinct._model.__class__ = UsesReadonlyCallable
    distinct._model.policy = ReadonlyCallable()
    distinct._model.callback_provider = FittingProvider(CalibrationConfig(calibration_func=None, threshold_func=partial(ReadonlyCallable())))
    distinct_handle = _freeze(distinct)
    assert _report(distinct_handle, assessment_case)['rows_sha256']
    with pytest.raises(ValueError, match='attempted fitting'), readonly_prediction([distinct]):
        distinct._model.callback_provider.prediction_calibration_config.threshold_func(distinct._model.model.coef_.ravel())
    def positional_default(offset):
        def invoke(values, adjustment=offset):
            return np.clip(values + adjustment, 0, 1)
        return invoke
    def keyword_default(offset):
        def invoke(values, *, adjustment=offset):
            return np.clip(values + adjustment, 0, 1)
        return invoke
    for factory, callback_offset in ((positional_default, expected_offset + 0.1), (keyword_default, expected_offset + 0.1), (positional_default, expected_offset)):
        prediction = factory(expected_offset)
        callback = factory(callback_offset)
        original_code = callback.__code__
        distinct._model.policy = prediction
        distinct._model.callback_provider.prediction_calibration_config.threshold_func = callback
        distinct_handle = _freeze(distinct)
        assert _report(distinct_handle, assessment_case)['rows_sha256']
        assert callback.__code__ is original_code
        with pytest.raises(ValueError, match='attempted fitting'), readonly_prediction([distinct]):
            callback(distinct._model.model.coef_.ravel(), adjustment=expected_offset)
        assert callback.__code__ is original_code
    class FirstAdjustment:
        def adjust(self, values):
            return values
    class SecondAdjustment:
        def adjust(self, values):
            return np.clip(values + expected_offset, 0, 1)
    class OrderedPrediction(FirstAdjustment, SecondAdjustment, LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = self.adjust(result['_probs'])
            return result
    ordered = copy.deepcopy(assessment_case[2][0][0])
    ordered_model = object.__new__(OrderedPrediction)
    vars(ordered_model).update(copy.deepcopy(vars(ordered._model)))
    ordered._model = ordered_model
    ordered_handle = _freeze(ordered)
    original_bases = OrderedPrediction.__bases__
    try:
        OrderedPrediction.__bases__ = (SecondAdjustment, FirstAdjustment, LogRegBinary)
        with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
            with pytest.raises(ValueError, match='prediction dependency changed'):
                _report(ordered_handle, assessment_case)
            assert loader.call_count == 0
    finally:
        OrderedPrediction.__bases__ = original_bases
    class NativeNewCallback:
        __new__ = staticmethod(getattr)
        def __init__(self, *args):
            raise AssertionError('unused initializer')
    unsafe_class = copy.deepcopy(assessment_case[2][0][0])
    unsafe_class._model.callback_provider = FittingProvider(CalibrationConfig(calibration_func=None, threshold_func=NativeNewCallback))
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='callbacks without observable Python call protection'):
            _freeze(unsafe_class)
        assert loader.call_count == 0
    class SharedInitializer:
        def __init__(self, values):
            self.values = values
    class SharedNew(SharedInitializer):
        def __new__(cls, values):
            return object.__new__(cls)
    for base in (SharedInitializer, SharedNew):
        class ThresholdClass(base):
            __slots__ = ()
        class ReadonlyClass(base):
            __slots__ = ()
        class UsesClass(LogRegBinary):
            def predict(self, data):
                result = super().predict(data)
                result['_probs'] = ReadonlyClass(result['_probs']).values
                return result
        for callback in (ThresholdClass, partial(ThresholdClass)):
            selected_class = copy.deepcopy(assessment_case[2][0][0])
            selected_class._model.__class__ = UsesClass
            selected_class._model.callback_provider = FittingProvider(CalibrationConfig(calibration_func=None, threshold_func=callback))
            class_handle = _freeze(selected_class)
            assert _report(class_handle, assessment_case)['rows_sha256']
            with pytest.raises(ValueError, match='attempted fitting'), readonly_prediction([selected_class]):
                callback(selected_class._model.model.coef_.ravel())
    for library_function in (LogisticRegression.predict_proba, LogisticRegression.fit):
        @wraps(library_function)
        def decorated_adjust(values):
            return np.clip(values + _LATE_RULES['offset'], 0, 1)
        decorated = copy.deepcopy(assessment_case[2][0][0])
        decorated._model.__class__ = UsesReadonlyCallable
        decorated._model.policy = decorated_adjust
        decorated_handle = _freeze(decorated)
        with patch.dict(_LATE_RULES, {'offset': 0.1}), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
            with pytest.raises(ValueError, match='prediction dependency changed'):
                _report(decorated_handle, assessment_case)
            assert loader.call_count == 0
    threshold_selections = []
    def bound_new(marker, cls, values):
        threshold_selections.append(cls)
        return np.max(values)
    def variadic_new(marker, *args):
        return bound_new(marker, *args)
    class NewProvider:
        def __call__(self, cls, values):
            return bound_new(expected_offset, cls, values)
    for factory in (partial(bound_new, expected_offset), partial(variadic_new, expected_offset), NewProvider()):
        class BoundConstructor:
            __new__ = factory
        for callback in (BoundConstructor, partial(BoundConstructor)):
            unsafe_class._model.callback_provider.prediction_calibration_config.threshold_func = callback
            _freeze(unsafe_class)
            with pytest.raises(ValueError, match='attempted fitting'), readonly_prediction([unsafe_class]):
                callback(unsafe_class._model.model.coef_.ravel())
            assert threshold_selections == []
    def target_copy(self, memo):
        clone = object.__new__(type(self))
        vars(clone).update(copy.deepcopy(vars(self), memo))
        _LATE_RULES['offset'] += 0.1
        return clone
    with history.connection(assessment_case[0] / history._HISTORY) as db:
        before_target_copy = db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
    with patch.dict(_LATE_RULES), patch.object(NextBarUpTarget, '__deepcopy__', target_copy, create=True), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='copy changed prediction dependencies before observation access'):
            _report(copying_handle, assessment_case)
        assert loader.call_count == 0
    with history.connection(assessment_case[0] / history._HISTORY) as db:
        assert db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0] == before_target_copy
    def bound_initializer(marker, cls, values):
        threshold_selections.append(cls)
    class BoundInitializer:
        __init__ = classmethod(partial(bound_initializer, expected_offset))
    for callback in (BoundInitializer, partial(BoundInitializer)):
        unsafe_class._model.callback_provider.prediction_calibration_config.threshold_func = callback
        _freeze(unsafe_class)
        with pytest.raises(ValueError, match='attempted fitting'), readonly_prediction([unsafe_class]):
            callback(unsafe_class._model.model.coef_.ravel())
        assert threshold_selections == []
    class HiddenFunction(staticmethod):
        @property
        def __func__(self):
            return identity_probabilities
    class HiddenThreshold:
        __call__ = HiddenFunction(max)
    unsafe_class._model.callback_provider.prediction_calibration_config.threshold_func = HiddenThreshold()
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='callbacks without observable Python call protection'):
            _freeze(unsafe_class)
        assert loader.call_count == 0
    def bound_adjust(offset, values, *, scale):
        return np.clip(values * scale + offset, 0, 1)
    class HiddenPartial(partial):
        @property
        def func(self):
            return identity_probabilities
        @property
        def args(self):
            return ()
        @property
        def keywords(self):
            return {}
        def adjust(self, values):
            return self(values)
        def __deepcopy__(self, memo):
            values = partial.__dict__['args'].__get__(self, type(self))
            keywords = partial.__dict__['keywords'].__get__(self, type(self))
            return type(self)(bound_adjust, *values, **keywords)
    class NonpreservingHiddenPartial(HiddenPartial):
        def __deepcopy__(self, memo):
            values = partial.__dict__['args'].__get__(self, type(self))
            keywords = partial.__dict__['keywords'].__get__(self, type(self))
            return type(self)(bound_adjust, values[0] + 0.1, **keywords)
    hidden_partial = copy.deepcopy(assessment_case[2][0][0])
    hidden_partial._model.__class__ = PartialState
    hidden_partial._model.policy = HiddenPartial(bound_adjust, expected_offset, scale=1.0)
    _freeze(hidden_partial)
    hidden_partial._model.policy = NonpreservingHiddenPartial(bound_adjust, expected_offset, scale=1.0)
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='copy changed'):
            _freeze(hidden_partial)
        assert loader.call_count == 0
    class ThresholdReceiver:
        def choose(self, values):
            threshold_selections.append(self)
            return np.max(values)
    threshold_receiver = ThresholdReceiver()
    class BoundMeta(type):
        __call__ = threshold_receiver.choose
    class BoundMetaCallback(metaclass=BoundMeta):
        pass
    for callback in (BoundMetaCallback, partial(BoundMetaCallback)):
        unsafe_class._model.callback_provider.prediction_calibration_config.threshold_func = callback
        _freeze(unsafe_class)
        with pytest.raises(ValueError, match='attempted fitting'), readonly_prediction([unsafe_class]):
            callback(unsafe_class._model.model.coef_.ravel())
        assert threshold_selections == []
    class HiddenDtype(np.ndarray):
        @property
        def dtype(self):
            return np.dtype(float)
        def __deepcopy__(self, memo):
            return np.ndarray.copy(self)
    keyed._model.__class__ = Keyed
    keyed._model.adjustments = np.asarray([later_probabilities], dtype=object).view(HiddenDtype)
    hidden_dtype_handle = _freeze(keyed)
    with patch.dict(_LATE_RULES, {'offset': 0.1}), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='prediction dependency changed'):
            _report(hidden_dtype_handle, assessment_case)
        assert loader.call_count == 0
    class HiddenFactory(defaultdict):
        @property
        def default_factory(self):
            return identity_probabilities
        def adjust(self, values):
            return defaultdict.__dict__['default_factory'].__get__(self, type(self))()(values)
        def __deepcopy__(self, memo):
            return type(self)(defaultdict.__dict__['default_factory'].__get__(self, type(self)))
    class NonpreservingFactory(HiddenFactory):
        def __deepcopy__(self, memo):
            return type(self)(later_probabilities)
    mapped._model.policy = HiddenFactory(lambda: identity_probabilities)
    _freeze(mapped)
    mapped._model.policy = NonpreservingFactory(lambda: identity_probabilities)
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='copy changed'):
            _freeze(mapped)
        assert loader.call_count == 0
    class HiddenCapacity(deque):
        @property
        def maxlen(self):
            return 1
        def adjust(self, values):
            return np.clip(values + deque.__dict__['maxlen'].__get__(self, type(self)) / 100, 0, 1)
        def __deepcopy__(self, memo):
            return type(self)(maxlen=deque.__dict__['maxlen'].__get__(self, type(self)))
    class NonpreservingCapacity(HiddenCapacity):
        def __deepcopy__(self, memo):
            return type(self)(maxlen=deque.__dict__['maxlen'].__get__(self, type(self)) + 1)
    mapped._model.policy = HiddenCapacity(maxlen=1)
    _freeze(mapped)
    mapped._model.policy = NonpreservingCapacity(maxlen=1)
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='copy changed'):
            _freeze(mapped)
        assert loader.call_count == 0
    def global_lookup(values):
        return np.clip(values + globals()['_LATE_RULES']['offset'], 0, 1)
    distinct._model.__class__ = UsesReadonlyCallable
    distinct._model.policy = global_lookup
    global_handle = _freeze(distinct)
    with patch.dict(_LATE_RULES, {'offset': 0.1}), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='prediction dependency changed'):
            _report(global_handle, assessment_case)
        assert loader.call_count == 0
    class ConstructorDescriptor(staticmethod):
        def __call__(self, *args):
            return identity_probabilities(args[-1])
    def descriptor_new(cls, values):
        threshold_selections.append(cls)
        return np.max(values)
    class DescriptorConstructor:
        __new__ = ConstructorDescriptor(descriptor_new)
    class InitializerDescriptor(classmethod):
        def __call__(self, *args):
            return identity_probabilities(args[-1])
    class DescriptorInitializer:
        __init__ = InitializerDescriptor(partial(bound_initializer, expected_offset))
    for constructor in (DescriptorConstructor, DescriptorInitializer):
        for callback in (constructor, partial(constructor)):
            unsafe_class._model.callback_provider.prediction_calibration_config.threshold_func = callback
            _freeze(unsafe_class)
            with pytest.raises(ValueError, match='attempted fitting'), readonly_prediction([unsafe_class]):
                callback(unsafe_class._model.model.coef_.ravel())
            assert threshold_selections == []
    for native_base, field in ((staticmethod, '__func__'), (classmethod, '__func__'), (property, 'fget')):
        class DirectDescriptor(native_base):
            __call__ = staticmethod(max)
            def __deepcopy__(self, memo, descriptor=native_base.__dict__[field]):
                return type(self)(descriptor.__get__(self, type(self)))
        unsafe_class._model.callback_provider.prediction_calibration_config.threshold_func = DirectDescriptor(identity_probabilities)
        with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
            with pytest.raises(ValueError, match='callbacks without observable Python call protection'):
                _freeze(unsafe_class)
            assert loader.call_count == 0
    class SentinelPolicy:
        DEFAULT = object()
        def __init__(self):
            self.policy = self.DEFAULT
        def adjust(self, values):
            return np.clip(values + (0 if self.policy is self.DEFAULT else 0.1), 0, 1)
    mapped._model.policy = SentinelPolicy()
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='copy changed'):
            _freeze(mapped)
        assert loader.call_count == 0
    class GlobalSentinelPolicy:
        def __init__(self):
            self.policy = _ALIAS_SENTINEL
        def adjust(self, values):
            return np.clip(values + (0 if self.policy is _ALIAS_SENTINEL else 0.1), 0, 1)
    class NestedSentinelPolicy(GlobalSentinelPolicy):
        def adjust(self, values):
            return np.clip(values + (0 if self.policy is _ALIAS_REGISTRY['default'] else 0.1), 0, 1)
    for implementation in (GlobalSentinelPolicy, NestedSentinelPolicy):
        mapped._model.policy = implementation()
        with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
            with pytest.raises(ValueError, match='copy changed'):
                _freeze(mapped)
            assert loader.call_count == 0
    class PreservingGlobalSentinel(GlobalSentinelPolicy):
        def __deepcopy__(self, memo):
            clone = object.__new__(type(self))
            clone.policy = self.policy
            return clone
    mapped._model.policy = PreservingGlobalSentinel()
    assert _report(_freeze(mapped), assessment_case)['rows_sha256']
    class StructuredPolicy(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = self.policy['adjust'](result['_probs'])
            return result
    structured = copy.deepcopy(assessment_case[2][0][0])
    structured._model.__class__ = StructuredPolicy
    structured._model.policy = np.asarray((later_probabilities,), dtype=[('adjust', object)])[()]
    structured_handle = _freeze(structured)
    with patch.dict(_LATE_RULES, {'offset': 0.1}), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='prediction dependency changed'):
            _report(structured_handle, assessment_case)
        assert loader.call_count == 0
    class FreshCallbackConfig:
        calibration_func = None
        @property
        def threshold_func(self):
            def choose(values):
                callback_calls.append('fresh threshold')
                return np.max(values)
            return choose
    unsafe_class._model.callback_provider.prediction_calibration_config = FreshCallbackConfig()
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='stable observable binding'):
            _freeze(unsafe_class)
        assert loader.call_count == 0 and callback_calls == []
    class StableCallbackConfig:
        calibration_func = None
        def __init__(self, callback):
            self.callback = callback
        @property
        def threshold_func(self):
            return self.callback
    class MethodCallbackConfig(StableCallbackConfig):
        @property
        def threshold_func(self):
            return self.choose
        def choose(self, values):
            callback_calls.append('bound threshold')
            return np.max(values)
    class PartialCallbackConfig(StableCallbackConfig):
        @property
        def threshold_func(self):
            return partial(self.callback)
    for config in (StableCallbackConfig(nested_threshold), MethodCallbackConfig(nested_threshold), PartialCallbackConfig(nested_threshold)):
        unsafe_class._model.callback_provider.prediction_calibration_config = config
        _freeze(unsafe_class)
        with pytest.raises(ValueError, match='attempted fitting'), readonly_prediction([unsafe_class]):
            config.threshold_func(unsafe_class._model.model.coef_.ravel())
        assert callback_calls == []
    class AnnotationState(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = np.clip(result['_probs'] + self.policy.__annotations__['offset'], 0, 1)
            return result
    annotated = copy.deepcopy(assessment_case[2][0][0])
    annotated._model.__class__ = AnnotationState
    annotated._model.policy = identity_probabilities
    with patch.object(identity_probabilities, '__annotations__', {'offset': expected_offset}):
        annotation_handle = _freeze(annotated)
        with patch.dict(identity_probabilities.__annotations__, {'offset': expected_offset + 0.1}), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
            with pytest.raises(ValueError, match='changed'):
                _report(annotation_handle, assessment_case)
            assert loader.call_count == 0
    class AnnotationCallableState(LogRegBinary):
        def predict(self, data):
            result = super().predict(data)
            result['_probs'] = self.policy.__annotations__['adjust'](result['_probs'])
            return result
    annotated._model.__class__ = AnnotationCallableState
    for policy in (identity_probabilities, LogisticRegression.predict_proba):
        annotated._model.policy = policy
        with patch.object(policy, '__annotations__', {'adjust': later_probabilities}):
            annotation_handle = _freeze(annotated)
            with patch.dict(_LATE_RULES, {'offset': 0.1}), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
                with pytest.raises(ValueError, match='prediction dependency changed'):
                    _report(annotation_handle, assessment_case)
                assert loader.call_count == 0
    class NativeMeta(type):
        __call__ = staticmethod(max)
    class NativeClassCallback(metaclass=NativeMeta):
        def __init__(self, values):
            self.values = values
    unsafe_class = copy.deepcopy(assessment_case[2][0][0])
    unsafe_class._model.callback_provider = FittingProvider(CalibrationConfig(calibration_func=None, threshold_func=NativeClassCallback))
    with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='callbacks without observable Python call protection'):
            _freeze(unsafe_class)
        assert loader.call_count == 0
    original_loader = assessment.load_admitted_source
    def changed_during_loading(*args, **kwargs):
        result = original_loader(*args, **kwargs)
        _LATE_RULES['offset'] = 0.1
        return result
    with patch.dict(_LATE_RULES), patch.object(assessment, 'load_admitted_source', side_effect=changed_during_loading), patch.object(assessment, '_tape', side_effect=AssertionError('changed predictions')) as tape:
        with pytest.raises(ValueError, match='changed prediction dependencies during loading') as failed:
            _report(branch_handle, assessment_case)
        assert tape.call_count == 0
    with history.connection(assessment_case[0] / history._HISTORY) as db:
        record = db.execute('SELECT * FROM attempts ORDER BY assessment_index DESC LIMIT 1').fetchone()
        assert record['state'] == 'failed' and record['id'] in str(failed.value)
        assert json.loads(record['error_json'])['stage'] == 'predict'
    changed = copy.deepcopy(assessment_case[2][0][0])
    changed._model.model = LogisticRegression()
    with pytest.raises(ValueError, match='not fitted'):
        _freeze(changed)

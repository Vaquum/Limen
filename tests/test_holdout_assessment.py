import copy
from contextlib import ExitStack, contextmanager
import json
import logging
import multiprocessing
from importlib.metadata import version
from pathlib import Path
from threading import Thread
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
from limen.targets.next_bar_up import NextBarUpTarget
from limen.yaml import CompiledSFD
from limen.yaml.compiler import build_search_strategy
from tests.test_holdout_research import _configuration, _project

_SCENARIO = {'fee_bps': 5.0, 'slip_bps': 2.0, 'notional_rate': 0.25, 'execution_lag_bars': 1, 'take_profit_bps': 20.0, 'stop_loss_bps': 20.0}
_REPEAT_KEYS = ('spec', 'spec_sha256', 'state_sha256', 'scenarios_sha256', 'source', 'source_schema', 'source_column_map', 'planned', 'classification', 'scenarios', 'rows_sha256')


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
    def __init__(self, estimator, x, y):
        self.estimator = estimator
        self.offset = float(np.mean(np.asarray(y)) - np.mean(estimator.predict_proba(np.asarray(x))[:, 1]))
    def predict_proba(self, x):
        positive = np.clip(self.estimator.predict_proba(np.asarray(x))[:, 1] + self.offset, 0, 1)
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


def mixed_architecture(data, C=1.0, prediction_calibration_config=None):
    from limen.sfd.reference_architecture.logreg_binary import logreg_binary
    from limen.sfd.reference_architecture.lightgbm_binary import lightgbm_binary
    if prediction_calibration_config is not None and prediction_calibration_config.threshold_func is not None:
        return logreg_binary(data, C=C, prediction_calibration_config=prediction_calibration_config)
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
    model.model = DecisionTreeClassifier(max_depth=2, random_state=42).fit(np.asarray(data['x_train']), np.asarray(data['y_train']))
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
    aliases.enter_context(patch.object(calibration, 'f17_calibrator', external_calibrator, create=True))
    aliases.enter_context(patch.object(scalers, 'f17_scaler', ExternalRobust, create=True))
    aliases.enter_context(patch.object(architectures, 'f17_mixed', mixed_architecture, create=True))
    aliases.enter_context(patch.object(architectures, 'f17_tree', tree_architecture, create=True))
    trained, sources = [], []
    for architecture in ('logreg', 'lightgbm', 'warmup', 'earlier_test'):
        current = copy.deepcopy(config)
        if architecture == 'logreg':
            current['sfd']['manifest']['reference_architecture'] = 'limen.sfd.reference_architecture.f17_mixed'
            current['sfd']['manifest']['calibration'] = {'probability_calibration': {'func': 'limen.calibration.f17_calibrator', 'params': {'method': 'sigmoid'}}, 'threshold_function': {'func': 'limen.calibration.grid_threshold_optimizer'}}
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
            stack.enter_context(patch(path, side_effect=lambda *args, **kwargs: calls.append('fit')))
        for sensor in trained[0] + trained[1]:
            for value in sensor._fitted_params.values():
                if type(value).__module__.startswith(('limen.scalers.', 'limen.targets.')):
                    stack.enter_context(patch.object(type(value), '__init__', side_effect=lambda *args, **kwargs: calls.append('fitted-transform')))
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
    for selected in [sensor for group in trained for sensor in group] + [cohort]:
        frozen = _freeze(selected, context='2024-01-01')
        first = _report(frozen, assessment_case)
        repeat = _report(frozen, assessment_case)
        assert first['state'] == repeat['state'] == 'completed'
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
    raw = sensor._fitted_params['_holdout_probe']
    expected = sensor.predict(raw)
    clone = owned_copy(sensor)
    second = owned_copy(clone)
    assert expected.reason is None
    assert expected == clone.predict(raw) == second.predict(raw)
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
    class Nonpreserving(LogRegBinary):
        def __deepcopy__(self, memo):
            result = type(self).__new__(type(self))
            vars(result).update(copy.deepcopy(vars(self), memo))
            result.extra = 'changed selected state'
            return result
    frozen = _freeze(assessment_case[2][0][0])
    with patch.dict(scaler_module.SCALING_RULES, dict.fromkeys(('open', 'high', 'low', 'close'), 'none')), patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
        with pytest.raises(ValueError, match='prediction dependency changed'):
            _report(frozen, assessment_case)
        assert loader.call_count == 0
    for kind, reason in ((Lazy, 'fixed fitted'), (Random, 'random dependency'), (Mutating, 'mutated fitted state'), (Nonpreserving, 'copy changed')):
        changed = copy.deepcopy(assessment_case[2][0][0])
        changed._model.__class__ = kind
        with patch.object(assessment, 'load_admitted_source', side_effect=AssertionError('final access')) as loader:
            with pytest.raises(ValueError, match=reason):
                _freeze(changed)
            assert loader.call_count == 0
    changed = copy.deepcopy(assessment_case[2][0][0])
    changed._model.model = LogisticRegression()
    with pytest.raises(ValueError, match='not fitted'):
        _freeze(changed)

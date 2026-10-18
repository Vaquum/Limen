from limen.experiment import _round_storage as _storage
import copy
import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression, Ridge
from threadpoolctl import threadpool_limits

from limen import __version__
from limen.calibration import apply_calibrated_predict, grid_threshold_optimizer, sklearn_probability_calibrator
from limen.experiment import CalibrationConfig, UniversalExperimentLoop
from limen.inference import Trainer
from limen.metrics.binary_metrics import binary_metrics
from limen.metrics.continuous_metrics import continuous_metrics
from limen.sfd.reference_architecture import LogRegBinary, RidgeRegressor
from limen.yaml import CompiledSFD, build_search_strategy
from tests.benchmark_sklearn_input_boundaries import _configuration, _manifest
from tests.test_record_model_outputs import _config as _recording_config

FIXTURE = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'
_SPLITS = ('train', 'val', 'test')
_LOGREG_PARAMS = {'C': 0.3, 'solver': 'lbfgs', 'max_iter': 300,
                  'random_state': 42, 'n_jobs': 1, 'tol': 1e-7}


@pytest.fixture(autouse=True)
def _single_native_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope='module')
def recorded_boundary_bars():
    return pl.read_parquet(FIXTURE, n_rows=336)


def _config(kind='logreg', method=None):
    config = _recording_config(kind, calibrated=method is not None)
    config['metadata'].update(name='sklearn_input_boundary', limen_version=__version__)
    manifest = config['sfd']['manifest']
    manifest['data_source'] = {
        'method': 'limen.data.HistoricalData.get_any_file',
        'params': {'file_path_or_url': str(FIXTURE)},
    }
    manifest['split_dates'] = {
        'train_start': '2024-01-01', 'train_end': '2024-01-08',
        'val_start': '2024-01-08', 'val_end': '2024-01-10',
        'test_start': '2024-01-10', 'test_end': '2024-01-12',
        'test_predict_guard': False,
    }
    manifest['backtest'] = {'fee_bps': 5.0, 'slip_bps': 3.0}
    params = _LOGREG_PARAMS if kind == 'logreg' else {'alpha': 0.3, 'solver': 'svd'}
    config['sfd']['params'] = {name: [value] for name, value in params.items()}
    if method is not None:
        manifest['calibration']['probability_calibration']['params']['method'] = method
    config['uel'].update(n_permutations=1, search_strategy={'type': 'grid'})
    return config


def _prepared(bars, *, kind='logreg', method=None):
    config = _config(kind, method)
    manifest = CompiledSFD(config).manifest()
    params = {name: values[0] for name, values in config['sfd']['params'].items()}
    return manifest, params, manifest.prepare_data(bars, params)


def _array_inputs(data):
    return {**data, **{f'{axis}_{split}': np.asarray(data[f'{axis}_{split}'])
                      for split in _SPLITS for axis in ('x', 'y')}}


def _no_interchange(self, *args, **kwargs):
    raise AssertionError('Sklearn boundary must not consume the dataframe interchange protocol')


def _assert_boundary_warnings(caught, *, allow_interchange=False, allow_feature_names=False):
    tokens = () if allow_interchange else ('__dataframe__', 'interchange')
    if not allow_feature_names:
        tokens += ('feature names',)
    assert not [str(item.message) for item in caught
                if any(token in str(item.message).lower() for token in tokens)]


def _assert_input_frames_unchanged(data, objects, snapshots, keys, names):
    assert set(data) == keys
    assert data['_feature_names'] == names
    for key, value in objects.items():
        assert data[key] is value
        assert value.equals(snapshots[key]), key
    assert all(data[f'x_{split}'].columns == data['x_train'].columns for split in _SPLITS)


def _direct_logreg(data, params, *, method=None, use_calibration=False, use_threshold=False,
                   threshold_params=None):
    model = LogisticRegression(**params).fit(data['x_train'], data['y_train'])
    fitted = model
    if use_calibration:
        fitted = CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(
            data['x_val'], data['y_val'],
        )
    score, threshold = None, 0.5
    if use_threshold:
        threshold, score = grid_threshold_optimizer(
            data['y_val'], fitted.predict_proba(data['x_val'])[:, 1], **threshold_params,
        )
    probabilities = fitted.predict_proba(data['x_test'])[:, 1]
    configured = use_calibration or use_threshold
    predictions = ((probabilities >= threshold).astype(np.int8)
                   if configured else model.predict(data['x_test']))
    return {'_preds': predictions, '_probs': probabilities,
            'optimal_threshold': threshold if configured else None, 'val_score': score}


@pytest.mark.parametrize('method,use_calibration,use_threshold', (
    (None, False, False), (None, False, True),
    ('sigmoid', True, True), ('sigmoid', True, False),
    ('sigmoid', False, True), ('sigmoid', False, False),
    ('isotonic', True, True),
))
def test_recorded_logreg_boundary_matches_direct_sklearn(
    recorded_boundary_bars, monkeypatch, method, use_calibration, use_threshold,
):
    manifest, params, data = _prepared(recorded_boundary_bars, method=method)
    threshold_params = {'threshold_min': 0.4, 'threshold_max': 0.6, 'threshold_step': 0.05}
    if method is None and use_threshold:
        manifest.with_calibration().threshold_function(
            func=grid_threshold_optimizer, **threshold_params,
        ).done()
    manifest._apply_backtest_cost(data, params)
    objects = {f'{axis}_{split}': data[f'{axis}_{split}']
               for split in _SPLITS for axis in ('x', 'y')}
    snapshots = {key: value.clone() for key, value in objects.items()}
    keys, names = set(data), list(data['_feature_names'])
    arrays = _array_inputs(data)
    expected = [_direct_logreg(inputs, _LOGREG_PARAMS, method=method,
                               use_calibration=use_calibration, use_threshold=use_threshold,
                               threshold_params=threshold_params) for inputs in (data, arrays)]
    params.update(use_calibration=use_calibration, use_threshold=use_threshold)
    with monkeypatch.context() as guard, warnings.catch_warnings(record=True) as caught:
        if not use_calibration:
            guard.setattr(pl.DataFrame, '__dataframe__', _no_interchange)
        warnings.simplefilter('always')
        actual = manifest.run_model(data, params)
        native = manifest.run_model(arrays, params)
        inferences = [result['_model'].predict({'x_test': inputs['x_test']})
                      for result, inputs in ((actual, data), (native, arrays))]
    _assert_boundary_warnings(caught, allow_interchange=use_calibration)
    for result, inference, baseline, inputs in zip((actual, native), inferences, expected, (data, arrays), strict=True):
        np.testing.assert_array_equal(result['_preds'], baseline['_preds'])
        np.testing.assert_array_equal(inference['_preds'], baseline['_preds'])
        np.testing.assert_allclose(inference['_probs'], baseline['_probs'], rtol=1e-10, atol=1e-12)
        assert result['optimal_threshold'] == baseline['optimal_threshold']
        assert result['val_score'] == baseline['val_score']
        expected_metrics = binary_metrics(inputs, baseline['_preds'], baseline['_probs'])
        expected_metrics.update(result['_model']._compute_confusion(
            baseline['_preds'], inputs['y_test'], inputs['price_data_for_backtest'],
        ))
        expected_metrics.update(result['_model']._compute_backtest(baseline['_preds'], inputs))
        for name, value in expected_metrics.items():
            assert result[name] == pytest.approx(value, nan_ok=True)
    if not use_calibration:
        np.testing.assert_array_equal(actual['_preds'], native['_preds'])
        np.testing.assert_allclose(inferences[0]['_probs'], inferences[1]['_probs'], rtol=1e-10, atol=1e-12)
    _assert_input_frames_unchanged(data, objects, snapshots, keys, names)


def test_recorded_isotonic_calibration_preserves_original_decisions():
    manifest = _manifest(True)
    config = _configuration('isotonic')
    manifest.prediction_calibration_config = config
    params = {'C': 1.0, 'solver': 'lbfgs', 'tol': 1e-8, 'max_iter': 200, 'random_state': 42}
    data = manifest.prepare_data(pl.read_parquet(FIXTURE), params)
    manifest._apply_backtest_cost(data, params)
    objects = {f'{axis}_{split}': data[f'{axis}_{split}']
               for split in _SPLITS for axis in ('x', 'y')}
    snapshots = {key: value.clone() for key, value in objects.items()}
    keys, names = set(data), list(data['_feature_names'])
    baseline = _direct_logreg(data, params, method='isotonic', use_calibration=True,
                              use_threshold=True, threshold_params=config.threshold_params)
    actual = manifest.run_model(data, params)
    replay = actual['_model'].predict({'x_test': data['x_test']})
    assert actual['optimal_threshold'] == baseline['optimal_threshold']
    assert actual['val_score'] == baseline['val_score']
    assert replay['optimal_threshold'] == baseline['optimal_threshold']
    assert replay['val_score'] == baseline['val_score']
    for result in (actual, replay):
        np.testing.assert_array_equal(result['_preds'], baseline['_preds'])
    np.testing.assert_allclose(replay['_probs'], baseline['_probs'], rtol=1e-10, atol=1e-12)
    _assert_input_frames_unchanged(data, objects, snapshots, keys, names)


@pytest.mark.parametrize('copy_x', (False, True))
def test_recorded_ridge_boundary_matches_direct_sklearn(recorded_boundary_bars, monkeypatch, copy_x):
    _, params, data = _prepared(recorded_boundary_bars, kind='ridge')
    params.update(copy_X=copy_x, fit_intercept=True)
    objects = {f'{axis}_{split}': data[f'{axis}_{split}']
               for split in _SPLITS for axis in ('x', 'y')}
    snapshots = {key: value.clone() for key, value in objects.items()}
    keys, names = set(data), list(data['_feature_names'])
    legacy = Ridge(**params).fit(data['x_train'], data['y_train'])
    expected = legacy.predict(data['x_test'])
    with monkeypatch.context() as guard, warnings.catch_warnings(record=True) as caught:
        guard.setattr(pl.DataFrame, '__dataframe__', _no_interchange)
        warnings.simplefilter('always')
        actual_model = RidgeRegressor().train(data, **params)
        actual = actual_model.evaluate(data)
        arrays = _array_inputs(data)
        array_x = np.array(arrays['x_train'], copy=True)
        arrays['x_train'] = array_x
        native = RidgeRegressor().train(arrays, **params).evaluate(arrays)
    _assert_boundary_warnings(caught)
    np.testing.assert_allclose(actual_model.model.coef_, legacy.coef_, rtol=1e-10, atol=1e-12)
    for result in (actual, native):
        np.testing.assert_allclose(result['_preds'], expected, rtol=1e-10, atol=1e-12)
        for name, value in continuous_metrics(data, expected).items():
            assert result[name] == pytest.approx(value, nan_ok=True)
    _assert_input_frames_unchanged(data, objects, snapshots, keys, names)


@pytest.mark.parametrize('method', ('sigmoid', 'isotonic'))
def test_builtin_probability_calibration_matches_each_input_baseline(recorded_boundary_bars, method):
    _, _, data = _prepared(recorded_boundary_bars)
    objects = {f'{axis}_{split}': data[f'{axis}_{split}']
               for split in _SPLITS for axis in ('x', 'y')}
    snapshots = {key: value.clone() for key, value in objects.items()}
    keys, names = set(data), list(data['_feature_names'])
    config = CalibrationConfig(
        calibration_func=sklearn_probability_calibrator, calibration_params={'method': method},
        threshold_func=grid_threshold_optimizer,
        threshold_params={'threshold_min': 0.4, 'threshold_max': 0.6, 'threshold_step': 0.05},
    )
    for inputs in (data, _array_inputs(data)):
        classifier = LogisticRegression(**_LOGREG_PARAMS).fit(inputs['x_train'], inputs['y_train'])
        baseline = _direct_logreg(inputs, _LOGREG_PARAMS, method=method, use_calibration=True,
                                  use_threshold=True, threshold_params=config.threshold_params)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            actual = apply_calibrated_predict(classifier, config, inputs)
        _assert_boundary_warnings(caught, allow_interchange=isinstance(inputs['x_val'], pl.DataFrame))
        np.testing.assert_array_equal(actual['_preds'], baseline['_preds'])
        np.testing.assert_allclose(actual['_probs'], baseline['_probs'], rtol=1e-10, atol=1e-12)
        assert actual['optimal_threshold'] == baseline['optimal_threshold']
        assert actual['val_score'] == baseline['val_score']
    _assert_input_frames_unchanged(data, objects, snapshots, keys, names)


def test_custom_calibration_keeps_original_frame_interface(recorded_boundary_bars, monkeypatch):
    _, _, data = _prepared(recorded_boundary_bars)
    calls = []
    predictions = []

    class FramePredictor:
        def __init__(self, model):
            self.model = model

        def predict_proba(self, frame):
            assert isinstance(frame, pl.DataFrame)
            assert frame.columns == data['x_train'].columns
            predictions.append(frame)
            return self.model.predict_proba(frame.to_numpy())

    def calibrate(model, x_val, y_val, *, marker):
        assert x_val is data['x_val']
        assert y_val is data['y_val']
        assert marker == 'recorded-frame-interface'
        calls.append((x_val, y_val))
        return FramePredictor(model)

    def threshold(y_val, probabilities):
        assert y_val is data['y_val']
        return grid_threshold_optimizer(y_val, probabilities)

    config = CalibrationConfig(
        calibration_func=calibrate, calibration_params={'marker': 'recorded-frame-interface'},
        threshold_func=threshold,
    )
    model = LogRegBinary(prediction_calibration_config=config).train(data, **_LOGREG_PARAMS)
    actual = model.predict(data)
    replay = model.predict({'x_test': data['x_test']})
    standalone = apply_calibrated_predict(model.model, config, data)
    with monkeypatch.context() as guard:
        guard.setattr(pl.DataFrame, '__dataframe__', _no_interchange)
        threshold_model = LogRegBinary(prediction_calibration_config=CalibrationConfig(threshold_func=threshold)).train(
            data, **_LOGREG_PARAMS,
        )
        threshold_only = apply_calibrated_predict(
            threshold_model.model, CalibrationConfig(threshold_func=threshold), data,
        )
    assert len(calls) == 2
    assert all(frame is data['x_val'] or frame is data['x_test'] for frame in predictions)
    assert sum(frame is data['x_val'] for frame in predictions) == 2
    for output in (replay, standalone, threshold_only):
        np.testing.assert_array_equal(output['_preds'], actual['_preds'])
        np.testing.assert_array_equal(output['_probs'], actual['_probs'])
        assert output['optimal_threshold'] == actual['optimal_threshold']
        assert output['val_score'] == actual['val_score']

    delegated_inputs = []

    def delegating_calibration(classifier, features, labels, **params):
        assert features is data['x_val']
        assert labels is data['y_val']
        delegated_inputs.append((features, labels))
        return sklearn_probability_calibrator(classifier, features, labels, **params)

    delegated_config = CalibrationConfig(
        calibration_func=delegating_calibration, calibration_params={'method': 'sigmoid'},
        threshold_func=threshold,
    )
    builtin_config = CalibrationConfig(
        calibration_func=sklearn_probability_calibrator, calibration_params={'method': 'sigmoid'},
        threshold_func=threshold,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        delegated_model = LogRegBinary(prediction_calibration_config=delegated_config).train(
            data, **_LOGREG_PARAMS,
        )
        delegated = delegated_model.predict(data)
        delegated_replay = delegated_model.predict({'x_test': data['x_test']})
        delegated_standalone = apply_calibrated_predict(delegated_model.model, delegated_config, data)
        builtin = apply_calibrated_predict(delegated_model.model, builtin_config, data)
    _assert_boundary_warnings(caught, allow_interchange=True)
    assert len(delegated_inputs) == 2
    for output in (delegated, delegated_replay, delegated_standalone):
        np.testing.assert_array_equal(output['_preds'], builtin['_preds'])
        np.testing.assert_allclose(output['_probs'], builtin['_probs'], rtol=1e-10, atol=1e-12)
        assert output['optimal_threshold'] == builtin['optimal_threshold']
        assert output['val_score'] == builtin['val_score']

    passthrough_inputs = []

    def passthrough_calibration(classifier, features, labels):
        assert features is data['x_val']
        assert labels is data['y_val']
        passthrough_inputs.append((features, labels))
        return classifier

    passthrough_config = CalibrationConfig(calibration_func=passthrough_calibration,
                                           threshold_func=threshold)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        passthrough_model = LogRegBinary(prediction_calibration_config=passthrough_config).train(
            data, **_LOGREG_PARAMS,
        )
        passthrough = passthrough_model.predict(data)
        passthrough_replay = passthrough_model.predict({'x_test': data['x_test']})
        passthrough_standalone = apply_calibrated_predict(passthrough_model.model, passthrough_config, data)
    _assert_boundary_warnings(caught, allow_interchange=True)
    assert len(passthrough_inputs) == 2
    for output in (passthrough, passthrough_replay, passthrough_standalone):
        np.testing.assert_array_equal(output['_preds'], threshold_only['_preds'])
        np.testing.assert_array_equal(output['_probs'], threshold_only['_probs'])
        assert output['optimal_threshold'] == threshold_only['optimal_threshold']
        assert output['val_score'] == threshold_only['val_score']


@pytest.mark.parametrize('kind,method', (('logreg', None), ('logreg', 'sigmoid'),
                                        ('logreg', 'isotonic'), ('ridge', None)))
def test_recorded_trainer_sensor_replays_sklearn_boundaries(
    recorded_boundary_bars, tmp_path, monkeypatch, kind, method,
):
    config = _config(kind, method)
    directory = tmp_path / 'recorded'
    loop = UniversalExperimentLoop(
        sfd=CompiledSFD(config), data=recorded_boundary_bars, experiment_dir=directory,
        yaml_reference=config, search_strategy=build_search_strategy(config), checkpoint_interval=1,
    )
    manifest, params, data = _prepared(recorded_boundary_bars, kind=kind, method=method)
    before = copy.deepcopy(config)
    with monkeypatch.context() as guard, warnings.catch_warnings(record=True) as caught:
        if method is None:
            guard.setattr(pl.DataFrame, '__dataframe__', _no_interchange)
        warnings.simplefilter('always')
        loop.run('sklearn_input_boundary', n_permutations=1, prep_each_round=True,
                 progress_bar=False, record_execution=True, record_model_outputs=True,
                 recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True))
        record = json.loads(_storage.text(_storage.round_path(directory)))
        sensor = Trainer(directory, data=recorded_boundary_bars).train([record['round_id']])[0]
        replay = sensor.predict_all(recorded_boundary_bars)
        expected = manifest.run_model(data, params)
    _assert_boundary_warnings(caught, allow_interchange=method is not None,
                               allow_feature_names=method is not None)
    assert config == before
    assert json.loads((directory / 'metadata.json').read_text())['yaml_reference'] == config
    saved = pl.read_csv(directory / 'results.csv').row(0, named=True)
    for name, value in expected.items():
        if name.startswith('_'):
            continue
        if value is None:
            assert saved[name] is None
        else:
            assert float(saved[name]) == pytest.approx(value, nan_ok=True)
    np.testing.assert_allclose(record['preds'], expected['_preds'], rtol=1e-6, atol=1e-9)
    by_time = {item.datetime: item for item in replay if item.reason is None}
    times = data['price_data_for_backtest']['datetime'].to_list()
    assert len(times) == len(record['preds'])
    for index, dt in enumerate(times):
        assert by_time[dt].prediction == pytest.approx(record['preds'][index], rel=1e-6, abs=1e-9)
        if kind == 'logreg':
            assert by_time[dt].probability == pytest.approx(record['probs'][index], rel=1e-6, abs=1e-9)
    if method is not None:
        assert sensor._model._calibration_threshold == pytest.approx(record['optimal_threshold'])
        assert sensor._model._val_score == pytest.approx(float(saved['val_score']))
    elif kind == 'ridge':
        assert record['probs'] is None
        assert all(item.probability is None for item in by_time.values())

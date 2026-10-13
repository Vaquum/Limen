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


def _assert_boundary_warnings(caught):
    assert not [str(item.message) for item in caught
                if any(token in str(item.message).lower()
                       for token in ('__dataframe__', 'interchange', 'feature names'))]


def _assert_input_frames_unchanged(data, objects, snapshots, keys, names):
    assert set(data) == keys
    assert data['_feature_names'] == names
    for key, value in objects.items():
        assert data[key] is value
        assert value.equals(snapshots[key]), key
    assert all(data[f'x_{split}'].columns == data['x_train'].columns for split in _SPLITS)


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
    if method is None and use_threshold:
        manifest.with_calibration().threshold_function(
            func=grid_threshold_optimizer, threshold_min=0.4, threshold_max=0.6, threshold_step=0.05,
        ).done()
    params.update(use_calibration=use_calibration, use_threshold=use_threshold)
    manifest._apply_backtest_cost(data, params)
    objects = {f'{axis}_{split}': data[f'{axis}_{split}']
               for split in _SPLITS for axis in ('x', 'y')}
    snapshots = {key: value.clone() for key, value in objects.items()}
    keys, names = set(data), list(data['_feature_names'])
    legacy = LogisticRegression(**_LOGREG_PARAMS).fit(data['x_train'], data['y_train'])
    fitted = legacy
    configured = use_calibration or use_threshold
    if use_calibration:
        fitted = CalibratedClassifierCV(FrozenEstimator(legacy), method=method).fit(
            np.asarray(data['x_val']), np.asarray(data['y_val']),
        )
    score, threshold = None, 0.5
    if use_threshold:
        threshold, score = grid_threshold_optimizer(
            data['y_val'], fitted.predict_proba(data['x_val'])[:, 1],
            threshold_min=0.4, threshold_max=0.6, threshold_step=0.05,
        )
    probabilities = fitted.predict_proba(data['x_test'])[:, 1]
    predictions = ((probabilities >= threshold).astype(np.int8)
                   if configured else legacy.predict(data['x_test']))
    with monkeypatch.context() as guard, warnings.catch_warnings(record=True) as caught:
        guard.setattr(pl.DataFrame, '__dataframe__', _no_interchange)
        warnings.simplefilter('always')
        actual = manifest.run_model(data, params)
        native = manifest.run_model(_array_inputs(data), params)
        inference = actual['_model'].predict({'x_test': data['x_test']})
    _assert_boundary_warnings(caught)
    for output in (actual, native, inference):
        np.testing.assert_array_equal(output['_preds'], predictions)
    for model in (actual['_model'], native['_model']):
        np.testing.assert_allclose(model.predict({'x_test': np.asarray(data['x_test'])})['_probs'],
                                   probabilities, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(inference['_probs'], probabilities, rtol=1e-10, atol=1e-12)
    assert actual['optimal_threshold'] == (pytest.approx(threshold) if configured else None)
    assert actual['val_score'] == (pytest.approx(score) if score is not None else None)
    expected_metrics = binary_metrics(data, predictions, probabilities)
    expected_metrics.update(actual['_model']._compute_confusion(
        predictions, data['y_test'], data['price_data_for_backtest'],
    ))
    expected_metrics.update(actual['_model']._compute_backtest(predictions, data))
    for name, value in expected_metrics.items():
        assert actual[name] == pytest.approx(value, nan_ok=True)
        assert native[name] == pytest.approx(value, nan_ok=True)
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
def test_builtin_probability_calibration_preserves_frame_array_parity(
    recorded_boundary_bars, monkeypatch, method,
):
    _, _, data = _prepared(recorded_boundary_bars)
    classifier = LogisticRegression(**_LOGREG_PARAMS).fit(
        np.asarray(data['x_train']), np.asarray(data['y_train']),
    )
    config = CalibrationConfig(
        calibration_func=sklearn_probability_calibrator, calibration_params={'method': method},
        threshold_func=grid_threshold_optimizer,
        threshold_params={'threshold_min': 0.4, 'threshold_max': 0.6, 'threshold_step': 0.05},
    )
    with monkeypatch.context() as guard, warnings.catch_warnings(record=True) as caught:
        guard.setattr(pl.DataFrame, '__dataframe__', _no_interchange)
        warnings.simplefilter('always')
        from_frames = apply_calibrated_predict(classifier, config, data)
        from_arrays = apply_calibrated_predict(classifier, config, _array_inputs(data))
    _assert_boundary_warnings(caught)
    assert set(from_frames) == set(from_arrays)
    for key in ('_preds', '_probs'):
        np.testing.assert_allclose(from_frames[key], from_arrays[key], rtol=1e-10, atol=1e-12)
    assert from_frames['optimal_threshold'] == from_arrays['optimal_threshold']
    assert from_frames['val_score'] == from_arrays['val_score']


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
    with monkeypatch.context() as guard:
        guard.setattr(pl.DataFrame, '__dataframe__', _no_interchange)
        model = LogRegBinary(prediction_calibration_config=config).train(data, **_LOGREG_PARAMS)
        actual = model.predict(data)
        replay = model.predict({'x_test': data['x_test']})
        standalone = apply_calibrated_predict(model.model, config, data)
    assert len(calls) == 2
    assert all(frame is data['x_val'] or frame is data['x_test'] for frame in predictions)
    assert sum(frame is data['x_val'] for frame in predictions) == 2
    for output in (replay, standalone):
        np.testing.assert_array_equal(output['_preds'], actual['_preds'])
        np.testing.assert_array_equal(output['_probs'], actual['_probs'])
        assert output['optimal_threshold'] == actual['optimal_threshold']
        assert output['val_score'] == actual['val_score']


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
        guard.setattr(pl.DataFrame, '__dataframe__', _no_interchange)
        warnings.simplefilter('always')
        loop.run('sklearn_input_boundary', n_permutations=1, prep_each_round=True,
                 progress_bar=False, record_execution=True, record_model_outputs=True)
        record = json.loads((directory / 'round_data.jsonl').read_text())
        sensor = Trainer(directory, data=recorded_boundary_bars).train([record['round_id']])[0]
        replay = sensor.predict_all(recorded_boundary_bars)
        expected = manifest.run_model(data, params)
    _assert_boundary_warnings(caught)
    assert config == before
    assert json.loads((directory / 'metadata.json').read_text())['yaml_reference'] == config
    saved = pl.read_csv(directory / 'results.csv').row(0, named=True)
    for name, value in expected.items():
        if name.startswith('_'):
            continue
        assert saved[name] == (pytest.approx(value, nan_ok=True) if value is not None else None)
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
        assert sensor._model._val_score == pytest.approx(saved['val_score'])
    elif kind == 'ridge':
        assert record['probs'] is None
        assert all(item.probability is None for item in by_time.values())

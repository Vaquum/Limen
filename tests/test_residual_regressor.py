from importlib import import_module
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from ruamel.yaml import YAML

from limen import Trainer, UniversalExperimentLoop
from limen.experiment import _round_storage
from limen.sfd.foundational_sfd import residual_regressor as foundational
from limen.sfd.reference_architecture import ResidualRegressor, residual_regressor
from limen.yaml import CompiledSFD, build_search_strategy, validate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def recorded():
    bars = pl.read_parquet(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet', n_rows=1024)
    return bars, foundational.manifest().prepare_data(bars, {})


@pytest.mark.parametrize('scale,alpha,intercept,early', [(0., .1, True, None), (.5, 1., False, None), (1., 1., True, 3)])
def test_observed_residual_fit_and_independent_addition(recorded, monkeypatch, scale, alpha, intercept, early):
    import xgboost
    from sklearn.linear_model import Ridge
    _, data = recorded
    x, y = np.asarray(data['x_train']), np.asarray(data['y_train'])
    center_x, center_y = (x.mean(axis=0), y.mean()) if intercept else (np.zeros(x.shape[1]), 0.)
    coefficients = np.linalg.solve((x-center_x).T @ (x-center_x) + alpha*np.eye(x.shape[1]), (x-center_x).T @ (y-center_y))
    expected = y - ((x-center_x) @ coefficients + center_y)
    calls = []
    ridge_fit, correction_fit = Ridge.fit, xgboost.XGBRegressor.fit

    def primary_fit(model, features, truth, **kwargs):
        np.testing.assert_array_equal(features, x)
        np.testing.assert_array_equal(truth, y)
        return ridge_fit(model, features, truth, **kwargs)

    def observed_fit(model, features, residual, **kwargs):
        calls.append(kwargs)
        np.testing.assert_array_equal(features, x)
        np.testing.assert_allclose(residual, expected, atol=1e-12)
        assert not np.array_equal(residual, y)
        assert model.get_params()['n_estimators'] == 24
        if early is None:
            assert 'eval_set' not in kwargs
        else:
            val_x, val_y = kwargs['eval_set'][0]
            np.testing.assert_array_equal(val_x, data['x_val'])
            np.testing.assert_allclose(val_y, np.asarray(data['y_val']) - ((np.asarray(val_x)-center_x) @ coefficients + center_y), atol=1e-12)
        return correction_fit(model, features, residual, **kwargs)

    monkeypatch.setattr(Ridge, 'fit', primary_fit)
    monkeypatch.setattr(xgboost.XGBRegressor, 'fit', observed_fit)
    model = ResidualRegressor().train(data | {'y_test': object()}, ridge_params={'alpha': alpha, 'fit_intercept': intercept, 'solver': 'svd'},
        xgboost_params={'n_estimators': 24, 'max_depth': 2, 'early_stopping_rounds': early}, correction_scale=scale)
    assert len(calls) == 1
    test_x = np.asarray(data['x_test'])
    primary = (test_x-center_x) @ coefficients + center_y
    correction = model.correction.model.get_booster().inplace_predict(test_x, iteration_range=(0, model.correction.model.best_iteration + 1) if early else (0, 0))
    prediction = model.predict({'x_test': data['x_test']})['_preds']
    np.testing.assert_allclose(prediction, primary + scale*correction, atol=1e-12)
    result = model.evaluate(data)
    np.testing.assert_allclose(result['rmse'], np.sqrt(np.mean((np.asarray(data['y_test'])-prediction)**2)))
    assert result['linkage_kind'] == 'residual' and result['linkage_training_rows'] == len(y)
    for name in ('features', 'observed', 'residual'):
        assert len(result['linkage_'+name+'_sha256']) == 64
    for index in range(5):
        np.testing.assert_allclose(model.predict({'x_test': data['x_test'].slice(index, 1)})['_preds'], prediction[index:index+1], atol=1e-12)
    np.testing.assert_allclose(model.predict({'x_test': data['x_test'].head(5)})['_preds'], prediction[:5], atol=1e-12)
    assert not any(key.startswith(('backtest_', 'confusion_')) for key in result)


def test_residual_native_source_logs_and_reconstruction(tmp_path, recorded, monkeypatch):
    bars, _ = recorded
    config = YAML(typ='safe').load(ROOT / 'limen/yaml/templates/residual_regressor.yaml')
    assert validate(config).valid
    monkeypatch.setattr(import_module('limen.data.historical_data'), '_read_any_file', lambda *_args, **_kwargs: bars)
    compiled = CompiledSFD(config)
    source = compiled.manifest().fetch_data()
    loop = UniversalExperimentLoop(sfd=compiled, data=source, experiment_dir=tmp_path,
                                   search_strategy=build_search_strategy(config), yaml_reference=config)
    loop.run('recorded_residual', n_permutations=1, prep_each_round=True, progress_bar=False,
             recording=dict.fromkeys(('predictions', 'alignment', 'fold_details'), True))
    log = pl.read_csv(tmp_path / 'results.csv')
    assert log.height == 1 and log['strict_mode_error'].drop_nulls().len() == 0
    record = json.loads(_round_storage.text(_round_storage.round_path(tmp_path)).splitlines()[0])
    sensor = Trainer(tmp_path, data=bars).train([record['round_id']])[0]
    model = sensor._model
    assert isinstance(model, ResidualRegressor)
    provenance = {name: log[name][0] for name in log.columns if name.startswith('linkage_')}
    assert provenance == model._provenance
    prepared = compiled.manifest().prepare_data(source, record['round_params'])
    features = prepared['x_train']
    header = json.dumps([features.shape, features.columns], separators=(',', ':')).encode()
    assert provenance['linkage_features_sha256'] == hashlib.sha256(header + np.asarray(features, dtype='<f8').tobytes(order='C')).hexdigest()
    rows = [row for row in sensor.predict_all(bars) if row.reason is None]
    assert rows and all(row.probability is None for row in rows)


def test_residual_controls_and_explicit_validation_failures(recorded):
    _, data = recorded
    config = YAML(typ='safe').load(ROOT / 'limen/yaml/templates/residual_regressor.yaml')
    controls = set(inspect.signature(residual_regressor).parameters) - {'data'}
    assert validate(config).valid and controls == set(config['sfd']['params']) == set(foundational.params())
    assert build_search_strategy(config).domain.total_combinations == 8
    with pytest.raises(ValueError, match='not fitted'):
        ResidualRegressor().predict({'x_test': data['x_test']})
    with pytest.raises(ValueError, match='finite'):
        ResidualRegressor().train(data, correction_scale=float('nan'))
    with pytest.raises(ValueError, match='objective'):
        ResidualRegressor().train(data, xgboost_params={'objective': 'binary:logistic'})
    with pytest.raises(ValueError, match='validation'):
        ResidualRegressor().train({name:value for name,value in data.items() if name not in ('x_val','y_val')}, xgboost_params={'early_stopping_rounds': 3})
    with pytest.raises(ValueError):
        ResidualRegressor().train(data | {'x_val': data['x_val'].head(0), 'y_val': np.asarray(data['y_val'])[:0]}, xgboost_params={'early_stopping_rounds': 3})
    with pytest.raises(ValueError, match='observed'):
        ResidualRegressor().train(data | {'y_train': np.asarray(data['y_train'])[:, None]})

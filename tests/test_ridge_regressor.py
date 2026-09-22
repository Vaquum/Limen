import inspect
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from ruamel.yaml import YAML
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge

from limen.backtest.backtest_snapshot import backtest_snapshot
from limen.cli.commands.run import run_experiment
from limen.data import HistoricalData
from limen.experiment import MLManifest
from limen.inference import Trainer
from limen.sfd import foundational_sfd, reference_architecture
from limen.sfd.foundational_sfd import ridge_regressor as ridge_sfd
from limen.sfd.reference_architecture import RidgeRegressor, ridge_regressor
from limen.yaml import CompiledSFD, build_search_strategy, parse, validate

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'limen/yaml/templates/ridge_regressor.yaml'


@pytest.fixture(scope='module')
def market_bars():
    # Retained market observations; no generated inputs or network data.
    return pl.read_parquet(ROOT / 'tests/fixtures/dollar_bar_crash_reversal_15m.parquet')


@pytest.fixture(scope='module')
def ridge_data(market_bars):
    params = {key: values[0] for key, values in ridge_sfd.params().items()}
    return ridge_sfd.manifest().prepare_data(market_bars, params)


@pytest.mark.parametrize('alpha', [0.1, 1.0, 10.0])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('solver', ['auto', 'cholesky', 'svd'])
def test_ridge_matches_sklearn(ridge_data, alpha, fit_intercept, solver):
    params = dict(alpha=alpha, fit_intercept=fit_intercept, solver=solver)
    expected = Ridge(**params).fit(ridge_data['x_train'], ridge_data['y_train'])
    actual = RidgeRegressor().train(ridge_data, **params)
    np.testing.assert_allclose(actual.model.coef_, expected.coef_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.model.intercept_, expected.intercept_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        actual.predict({'x_test': ridge_data['x_test']})['_preds'],
        expected.predict(ridge_data['x_test']), rtol=1e-12, atol=1e-12,
    )


@pytest.mark.parametrize('inline_metrics', [True, False])
@pytest.mark.parametrize('with_price', [True, False])
def test_ridge_evaluation_contract(ridge_data, inline_metrics, with_price):
    data = dict(ridge_data)
    if not with_price:
        del data['price_data_for_backtest']
    data.update(backtest_fee_bps=5.0, backtest_slip_bps=2.0)
    model = RidgeRegressor().train(data, alpha=1.0)
    result = model.evaluate(data, inline_metrics=inline_metrics)
    preds = result['_preds']
    assert preds.shape == (len(data['y_test']),)
    assert np.isfinite(preds).all()
    assert {'bias', 'mae', 'rmse', 'r2', 'mape'} <= result.keys()
    assert result['mae'] == round(float(np.mean(np.abs(preds - data['y_test']))), 3)
    assert '_probs' not in result
    assert any(key.startswith('confusion_') for key in result) == inline_metrics
    assert any(key.startswith('backtest_') for key in result) == (inline_metrics and with_price)
    if inline_metrics:
        positive = preds > 0
        actual_positive = np.asarray(data['y_test']) > 0
        assert result['confusion_tp'] == int((positive & actual_positive).sum())
        assert result['confusion_fp'] == int((positive & ~actual_positive).sum())
    if inline_metrics and with_price:
        price = data['price_data_for_backtest']
        opened, closed = price['open'].to_numpy(), price['close'].to_numpy()
        expected = backtest_snapshot(
            {'predictions': (preds > 0).astype(int), 'open': opened,
             'close': closed, 'price_change': closed - opened},
            execution_lag_bars=1, fee_bps=5.0, slip_bps=2.0,
        )
        for key, value in expected.items():
            np.testing.assert_equal(result[f'backtest_{key}'], value)


@pytest.mark.parametrize('solver', ['cholesky', 'svd', 'lsqr'])
def test_ridge_wrapper_preserves_model_and_native_params(ridge_data, solver):
    params = dict(alpha=10.0, fit_intercept=False, copy_X=True, max_iter=1000,
                  tol=1e-6, solver=solver, positive=False, random_state=7)
    result = ridge_regressor(ridge_data, **params)
    model = result['_model']
    assert isinstance(model, RidgeRegressor)
    assert model.model.get_params() == params
    np.testing.assert_array_equal(result['_preds'], model.predict(ridge_data)['_preds'])


def test_ridge_train_uses_only_training_split_and_refits(ridge_data):
    model = RidgeRegressor().train(ridge_data)
    first = model.model
    altered = dict(ridge_data)
    altered['y_test'] = np.asarray(ridge_data['y_test'])[::-1]
    altered['y_val'] = np.asarray(ridge_data['y_val'])[::-1]
    model.train(altered)
    assert model.model is not first
    np.testing.assert_array_equal(model.model.coef_, first.coef_)
    np.testing.assert_array_equal(model.model.intercept_, first.intercept_)
    assert RidgeRegressor.deterministic is False


def test_ridge_svd_accepts_rank_deficiency(ridge_data):
    data = dict(ridge_data)
    for split in ['train', 'test']:
        x = np.asarray(data[f'x_{split}'])
        data[f'x_{split}'] = np.column_stack([x, x[:, 0]])
    expected = Ridge(alpha=0.0, solver='svd').fit(data['x_train'], data['y_train'])
    actual = RidgeRegressor().train(data, alpha=0.0, solver='svd')
    np.testing.assert_allclose(actual.predict(data)['_preds'], expected.predict(data['x_test']))


@pytest.mark.parametrize('params', [
    {'alpha': -1.0}, {'solver': 'not-a-solver'},
    {'positive': True, 'solver': 'cholesky'},
])
def test_ridge_preserves_sklearn_validation(ridge_data, params):
    with pytest.raises(ValueError):
        RidgeRegressor().train(ridge_data, **params)


def test_ridge_rejects_unknown_parameters_and_unfitted_prediction(ridge_data):
    with pytest.raises(TypeError, match='solver_eps'):
        RidgeRegressor().train(ridge_data, solver_eps=1e-12)
    with pytest.raises(NotFittedError):
        RidgeRegressor().predict({'x_test': ridge_data['x_test']})


def test_ridge_template_exports_and_parameter_surface():
    config, errors = parse(TEMPLATE.read_text())
    assert errors == []
    result = validate(config)
    assert result.valid, [error.message for error in result.errors]
    build_search_strategy(config)
    compiled = CompiledSFD(config)
    assert isinstance(compiled.manifest(), MLManifest)
    assert compiled.manifest().strict_mode is True
    assert ridge_sfd.manifest().strict_mode is True
    assert 'RidgeRegressor' in reference_architecture.__all__
    assert 'ridge_regressor' in reference_architecture.__all__
    assert 'ridge_regressor' in foundational_sfd.__all__
    model_params = set(inspect.signature(ridge_regressor).parameters) - {'data'}
    assert model_params == set(ridge_sfd.params()) == set(compiled.params())
    assert compiled.params() == ridge_sfd.params()


def test_ridge_yaml_experiment_reconstructs(tmp_path, monkeypatch, market_bars):
    config, errors = parse(TEMPLATE.read_text())
    assert errors == []
    config['sfd']['manifest']['split_dates'] = {
        'train_start': '2026-01-01', 'train_end': '2026-01-23',
        'val_start': '2026-01-24', 'val_end': '2026-01-30',
        'test_start': '2026-01-31', 'test_end': '2026-02-06',
    }
    config['sfd']['params'] = {key: [values[0]] for key, values in ridge_sfd.params().items()}
    config['sfd']['params']['alpha'] = [0.1, 1.0]
    experiment_dir = tmp_path / 'experiment'
    config['uel'].update(n_permutations=2, output_path=str(experiment_dir))
    yaml_path = tmp_path / 'ridge.yaml'
    with yaml_path.open('w') as handle:
        YAML().dump(config, handle)

    monkeypatch.setattr(HistoricalData, 'get_spot_klines', staticmethod(lambda **kwargs: market_bars))
    assert run_experiment(yaml_path, progress_bar=False)
    entries = [json.loads(line) for line in (experiment_dir / 'round_data.jsonl').read_text().splitlines()]
    assert len(entries) == 2
    log = pl.read_csv(experiment_dir / 'results.csv')
    assert log.height == 2
    assert 'mae' in log.columns
    assert log['mae'].is_not_null().all()
    assert np.isfinite(log['mae'].to_numpy()).all()

    ids = [str(entry['round_id']) for entry in entries]
    sensors = Trainer(experiment_dir, data=market_bars).train(ids)
    assert len(sensors) == 2
    for sensor in sensors:
        assert isinstance(sensor._model, RidgeRegressor)
        prediction = sensor.predict(market_bars)
        assert prediction.reason is None
        assert np.isfinite(prediction.prediction)
        assert prediction.probability is None

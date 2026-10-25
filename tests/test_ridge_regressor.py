from limen.experiment import _round_storage as _storage
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
from limen.cohort import Cohort
from limen.data import HistoricalData
from limen.experiment.manifest_core import DataSourceResolver
from limen.experiment import MLManifest
from limen.inference import Trainer
from limen.sfd import foundational_sfd, reference_architecture
from limen.sfd.foundational_sfd import ridge_regressor as ridge_sfd
from limen.sfd.reference_architecture import RidgeRegressor, ridge_regressor
from limen.yaml import CompiledSFD, build_search_strategy, parse, validate

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'limen/yaml/templates/ridge_regressor.yaml'


def _recorded_data_source(config):
    assert config.method is HistoricalData.get_spot_klines
    return _load_market_bars(**config.params)


def _load_market_bars(**kwargs):
    return pl.read_parquet(ROOT / 'tests/fixtures/dollar_bar_crash_reversal_15m.parquet')


@pytest.fixture(scope='module')
def market_bars():
    return _load_market_bars()


@pytest.fixture(scope='module')
def ridge_data(market_bars):
    params = {key: values[0] for key, values in ridge_sfd.params().items()}
    return ridge_sfd.manifest().prepare_data(market_bars, params)


@pytest.mark.parametrize('alpha', [0.1, 1.0, 10.0])
@pytest.mark.parametrize('fit_intercept', [True, False])
@pytest.mark.parametrize('solver', ['auto', 'cholesky', 'svd'])
def test_ridge_matches_sklearn(ridge_data, alpha, fit_intercept, solver):
    params = {'alpha': alpha, 'fit_intercept': fit_intercept, 'solver': solver}
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
    assert result['mae'] == float(np.mean(np.abs(preds - np.asarray(data['y_test']))))
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
    params = {
        'alpha': 10.0, 'fit_intercept': False, 'copy_X': True, 'max_iter': 1000,
        'tol': 1e-6, 'solver': solver, 'positive': False, 'random_state': 7,
    }
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
    config['sfd']['params']['alpha'] = [0.1, 1.0, 1_000_000.0]
    experiment_dir = tmp_path / 'results/dev/experiment'
    config['uel'].update(n_permutations=3, output_path='experiment')
    yaml_path = tmp_path / 'ridge.yaml'
    with yaml_path.open('w') as handle:
        YAML().dump(config, handle)

    monkeypatch.setattr(DataSourceResolver, 'resolve', staticmethod(_recorded_data_source))
    assert run_experiment(yaml_path, results_base=tmp_path, progress_bar=False)
    entries = [json.loads(line) for line in _storage.text(_storage.round_path(experiment_dir)).splitlines()]
    assert len(entries) == 3
    log = pl.read_csv(experiment_dir / 'results.csv')
    assert log.height == 3
    assert 'mae' in log.columns
    assert log['mae'].is_not_null().all()
    assert np.isfinite(log['mae'].to_numpy()).all()

    ids = [str(entry['round_id']) for entry in entries]
    sensors = sorted(Trainer(experiment_dir, data=market_bars).train(ids), key=lambda s: s.round_params['alpha'])
    assert [sensor.round_params['alpha'] for sensor in sensors] == [0.1, 1.0, 1_000_000.0]
    assert all(sensor.round_params['fit_intercept'] is True for sensor in sensors)
    for sensor in sensors:
        assert isinstance(sensor._model, RidgeRegressor)
        prediction = sensor.predict(market_bars)
        assert prediction.reason is None
        assert np.isfinite(prediction.prediction)
        assert prediction.probability is None

    member_batches = [sensor.predict_all(market_bars) for sensor in sensors]
    datetimes = [bar.datetime for bar in member_batches[0]]
    rows = [i for i, bar in enumerate(member_batches[0]) if bar.reason is None]
    assert rows
    assert all(datetimes[i] is not None for i in rows)
    for batch in member_batches:
        assert len(batch) == market_bars.height
        assert [bar.datetime for bar in batch] == datetimes
        assert [i for i, bar in enumerate(batch) if bar.reason is None] == rows
        assert all(bar.probability is None for bar in batch)

    predictions = np.array([[batch[i].prediction for i in rows] for batch in member_batches], dtype=float)
    assert np.isfinite(predictions).all()
    votes = predictions > 0
    down_rows = [rows[i] for i in np.flatnonzero(predictions[2] < 0)]
    tie_rows = [rows[i] for i in np.flatnonzero(votes[0] != votes[2])]
    assert votes[2].any()
    assert down_rows
    assert tie_rows

    for member_indices in ([2], [0, 2], [0, 1, 2]):
        members = [sensors[i] for i in member_indices]
        if len(members) == 3:
            cohort = Cohort(rounds=[(experiment_dir, sensor.permutation_id) for sensor in members])
        else:
            cohort = Cohort(experiment_log_path=str(experiment_dir),
                            permutation_ids=[sensor.permutation_id for sensor in members])
        cohort.set_members(members)
        assert cohort.architecture_id == 'limen.sfd.reference_architecture.ridge_regressor.ridge_regressor'
        assert cohort._fallback_vote_threshold() == 0.0
        assert cohort.aggregation_mode == 'majority_vote'
        assert cohort.supports_probabilities is False

        batch = cohort.predict_all(market_bars)
        assert len(batch) == market_bars.height
        assert [bar.datetime for bar in batch] == datetimes
        assert [i for i, bar in enumerate(batch) if bar.reason is None] == rows
        assert all(bar.probability is None for bar in batch)
        expected = (votes[member_indices].sum(axis=0) > len(members) / 2).astype(int)
        np.testing.assert_array_equal([batch[i].prediction for i in rows], expected)
        if len(members) == 1:
            assert all(batch[i].prediction == 0 for i in down_rows)
        elif len(members) == 2:
            assert all(batch[i].prediction == 0 for i in tie_rows)

        for row in {len(batch) - 1, down_rows[0], tie_rows[0]}:
            prefix = market_bars.head(row + 1)
            assert cohort.predict(prefix) == cohort.predict_all(prefix)[-1] == batch[row]

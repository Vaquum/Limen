from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from ruamel.yaml import YAML
from sklearn.linear_model import QuantileRegressor

from limen import Trainer, UniversalExperimentLoop
from limen.experiment import _round_storage
from limen.sfd.foundational_sfd import entropy_var_regressor as foundational
from limen.sfd.reference_architecture import EntropyVaRRegressor, entropy_var_regressor
from limen.yaml import CompiledSFD, build_search_strategy, validate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def recorded():
    bars = pl.read_parquet(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet', n_rows=1024)
    params = {key: values[0] for key, values in foundational.params().items()}
    manifest = foundational.manifest()
    return bars, params, manifest, manifest.prepare_data(bars, params)


def _entropy(frame):
    windows = frame.select([column for column in frame.columns if column.startswith('ret_1_lag_')]).to_numpy()
    return np.asarray([sum(-p * np.log2(p) for p in (np.mean(row < 0), np.mean(row >= 0)) if p > 0)
                       for row in windows])[:, None]


@pytest.mark.parametrize('quantile,fit_intercept,alpha,solver_options', [(0.01, True, 0., None), (0.05, True, .1, {'presolve': False}), (0.1, False, 0., None)])
def test_recorded_entropy_quantile_and_independent_losses(recorded, quantile, fit_intercept, alpha, solver_options):
    _, params, manifest, data = recorded
    result = manifest.run_model(data, params | {'quantile': quantile, 'fit_intercept': fit_intercept, 'alpha': alpha, 'solver_options': solver_options})
    model = result['_model']
    native = QuantileRegressor(quantile=quantile, alpha=alpha, fit_intercept=fit_intercept, solver='highs', solver_options=solver_options).fit(_entropy(data['x_train']), data['y_train'])
    assert isinstance(model, EntropyVaRRegressor)
    np.testing.assert_allclose(model._entropy(data['x_train']), _entropy(data['x_train']), atol=1e-15)
    expected = _entropy(data['x_test'])[:, 0] * native.coef_[0] + native.intercept_
    np.testing.assert_allclose(result['_preds'], expected, atol=1e-12)
    truth = np.asarray(data['y_test'])
    residual = truth - expected
    np.testing.assert_allclose(result['pinball_loss'], np.mean([quantile * r if r >= 0 else (quantile - 1) * r for r in residual]))
    assert result['violation_rate'] == float(np.mean(truth < expected))
    assert not any(key.startswith(('confusion_', 'backtest_')) for key in result)
    assert '_probs' not in result
    bars = recorded[0]
    alignment = data['_alignment']
    future = bars.with_columns((((pl.col('close').shift(-params['horizon']) - pl.col('close')) / pl.col('close')) * 100).alias('target'))
    aligned = future.filter(pl.col('datetime').is_between(alignment['first_test_datetime'], alignment['last_test_datetime']))
    np.testing.assert_allclose(truth, aligned['target'].to_numpy(), rtol=1e-12, atol=1e-12)
    for index in range(5):
        np.testing.assert_allclose(model.predict({'x_test': data['x_test'].slice(index, 1)})['_preds'], expected[index:index + 1])
    np.testing.assert_allclose(model.predict({'x_test': data['x_test'].head(5).to_numpy()})['_preds'], expected[:5])
    altered = deepcopy(data)
    altered['y_test'] = np.asarray(data['y_test'])[::-1]
    altered['y_val'] = np.asarray(data['y_val'])[::-1]
    repeated = EntropyVaRRegressor().train(altered, quantile=quantile, fit_intercept=fit_intercept, alpha=alpha, solver_options=solver_options)
    np.testing.assert_array_equal(model.model.coef_, repeated.model.coef_)


def test_entropy_compiled_recording_and_continuous_reconstruction(tmp_path, recorded):
    bars, _, _, _ = recorded
    config = YAML(typ='safe').load(ROOT / 'limen/yaml/templates/entropy_var_regressor.yaml')
    config['sfd']['manifest']['data_source'] = {'method': 'limen.data.HistoricalData.get_any_file',
        'params': {'file_path_or_url': str(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet')}}
    loop = UniversalExperimentLoop(sfd=CompiledSFD(config), data=bars, experiment_dir=tmp_path,
        search_strategy=build_search_strategy(config), yaml_reference=config)
    loop.run('recorded_entropy', n_permutations=1, prep_each_round=True, progress_bar=False,
             record_model_outputs=True, recording={'predictions': True, 'alignment': True})
    assert loop.experiment_log.height == 1 and loop.experiment_log['strict_mode_error'].null_count() == 1
    record = json.loads(_round_storage.text(_round_storage.round_path(tmp_path)).splitlines()[0])
    assert {'pinball_loss', 'violation_rate'} <= set(loop.experiment_log.columns)
    sensor = Trainer(tmp_path, data=bars).train([record['round_id']])[0]
    rows = sensor.predict_all(bars)
    assert sensor._model.inference_output == ('continuous', False)
    actionable = [row for row in rows if row.reason is None]
    assert actionable and all(row.probability is None for row in actionable)
    assert all(np.isfinite(row.prediction) for row in actionable)


def test_entropy_guards_and_native_template_controls(recorded):
    _, _, _, data = recorded
    for quantile in (0, .5, 1, float('nan')):
        with pytest.raises(ValueError, match='quantile'):
            EntropyVaRRegressor().train(data, quantile=quantile)
    for key in ('_scaler', '_pca', '_trade_policy'):
        with pytest.raises(ValueError, match='raw returns'):
            EntropyVaRRegressor().train(data | {key: object()})
    with pytest.raises(ValueError, match='contiguous'):
        EntropyVaRRegressor().train(data | {'x_train': data['x_train'].drop('ret_1_lag_0')})
    with pytest.raises(ValueError, match='not fitted'):
        EntropyVaRRegressor().predict(data)
    config = YAML(typ='safe').load(ROOT / 'limen/yaml/templates/entropy_var_regressor.yaml')
    assert validate(config).valid
    controls = set(inspect.signature(entropy_var_regressor).parameters) - {'data'}
    assert controls <= config['sfd']['params'].keys() and controls <= foundational.params().keys()
    assert build_search_strategy(config).domain.total_combinations == 4


def test_recorded_sign_entropy_endpoint_and_balanced_windows(recorded):
    _, _, _, data = recorded
    model = EntropyVaRRegressor().train(data)
    for end in (0, 1):
        columns = ['ret_1_lag_' + str(lag) for lag in range(end + 1)]
        fitted = EntropyVaRRegressor().train(data | {'x_train': data['x_train'].select(columns)})
        observed = data['x_test'].select(columns)
        actual = fitted._entropy(observed)[:, 0]
        if end == 0:
            assert np.array_equal(actual, np.zeros(observed.height))
        else:
            values = observed.to_numpy()
            expected = (np.signbit(values[:, 0]) != np.signbit(values[:, 1])).astype(float)
            np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match='fitted feature order'):
        model.predict({'x_test': data['x_test'].to_numpy()[:, :-1]})

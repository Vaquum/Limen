import importlib
import inspect
import json
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from xgboost import XGBRegressor

from limen.data import HistoricalData
from limen.experiment.manifest_core import _apply_feature_transforms
from limen.sfd.foundational_sfd import xgboost_regressor as sfd
from limen.sfd.reference_architecture.xgboost_regressor import xgboost_regressor

ROOT = Path(__file__).resolve().parents[1]
GROUP_COLUMNS = {'momentum': 'stoch_k', 'trend': 'sma_10_ratio', 'volume': 'trade_intensity',
                 'volatility': 'bollinger_position', 'structure': 'price_range_position'}


def _params():
    return {key: values[0] for key, values in sfd.params().items()}


@pytest.fixture(scope='module')
def bars():
    return pl.read_parquet(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet', n_rows=2048)


@pytest.fixture(scope='module')
def prepared(bars):
    return sfd.manifest().prepare_data(bars, _params())


def test_wrapper_forwards_controls(prepared, monkeypatch):
    received = {}

    def constructor(**kwargs):
        received.update(kwargs)
        return XGBRegressor(**kwargs)

    module = importlib.import_module('limen.sfd.reference_architecture.xgboost_regressor')
    monkeypatch.setattr(module, 'require_optional', lambda *args: SimpleNamespace(XGBRegressor=constructor))
    controls = {
        'eval_metric': ['mae', 'rmse'], 'max_leaves': 16, 'max_bin': 128,
        'grow_policy': 'lossguide', 'sampling_method': 'uniform',
        'colsample_bylevel': 0.9, 'colsample_bynode': 0.8, 'max_delta_step': 1.0,
        'tree_method': 'hist', 'num_parallel_tree': 2, 'base_score': 0.0,
        'scale_pos_weight': 1.0, 'n_jobs': 1, 'verbosity': 0, 'device': 'cpu',
        'enable_categorical': False, 'max_cat_to_onehot': 4, 'max_cat_threshold': 64,
        'importance_type': 'gain', 'interaction_constraints': [[0, 1]],
        'monotone_constraints': tuple(0 for _ in prepared['x_train'].columns),
        'multi_strategy': 'one_output_per_tree', 'validate_parameters': True,
    }
    manifest = sfd.manifest()
    resolved = manifest.resolve_model_kwargs({**controls, 'feature_groups': 'all', 'lag': 2})
    for key, value in controls.items():
        assert resolved[key] == value
    assert 'lag' not in resolved and 'feature_groups' not in resolved
    result = manifest.run_model(prepared, {**controls, 'n_estimators': 8})
    for key, value in controls.items():
        assert received[key] == value
    assert np.isfinite(result['_preds']).all()
    assert {'mae', 'rmse'} == set(result['_model'].model.evals_result()['validation_0'])


@pytest.mark.parametrize('validation', ['present', 'absent', 'empty'])
@pytest.mark.parametrize('early_stopping', [None, 3])
def test_training_preserves_optional_defaults(prepared, monkeypatch, validation, early_stopping):
    received = {}

    def constructor(**kwargs):
        received.update(kwargs)
        return XGBRegressor(**kwargs)

    module = importlib.import_module('limen.sfd.reference_architecture.xgboost_regressor')
    monkeypatch.setattr(module, 'require_optional', lambda *args: SimpleNamespace(XGBRegressor=constructor))
    data = dict(prepared)
    if validation == 'absent':
        del data['x_val'], data['y_val']
    elif validation == 'empty':
        data['x_val'], data['y_val'] = data['x_val'].head(0), data['y_val'].head(0)
    result = xgboost_regressor(data, n_estimators=8, n_jobs=1, early_stopping_rounds=early_stopping)
    assert received['eval_metric'] == ['rmse']
    assert 'device' not in received and 'max_bin' not in received
    assert received['early_stopping_rounds'] is early_stopping
    expected = early_stopping if validation == 'present' else None
    assert result['_model'].model.get_params()['early_stopping_rounds'] == expected
    assert np.isfinite(result['_preds']).all()
    assert {'rmse', 'mae', 'r2'} <= result.keys()


@pytest.mark.parametrize('group', ['all', *GROUP_COLUMNS])
@pytest.mark.parametrize('scaler', ['robust', 'linear'])
def test_research_feature_pipeline(bars, group, scaler):
    params = {**_params(), 'feature_groups': group, 'scaler_type': scaler}
    manifest = sfd.manifest()
    prepared = manifest.prepare_data(bars, params)
    assert manifest.strict_mode
    for split in ['train', 'val', 'test']:
        features = prepared[f'x_{split}']
        assert features.height > 0
        assert features.columns == prepared['x_train'].columns
        assert np.isfinite(features.to_numpy()).all()
        assert np.isfinite(prepared[f'y_{split}'].to_numpy()).all()
    for name, column in GROUP_COLUMNS.items():
        assert (column in prepared['x_train'].columns) == (group in ('all', name))
        assert (f'{column}_lag_1' in prepared['x_train'].columns) == (group in ('all', name))


def test_research_windows_horizon_and_ablation(bars):
    params = {key: values[-1] for key, values in sfd.params().items()}
    params.update(feature_groups='all', feature_drop_count=0)
    manifest = sfd.manifest()
    prepared = manifest.prepare_data(bars, params)
    for col in ['ret_2_lag_3', 'ret_40_lag_3', 'rsi_21_lag_3', 'close_sma_100_lag_3',
                'volume_sma_40_lag_3', 'ret_1_volatility_40_lag_3', 'volume_regime_lag_3']:
        assert col in prepared['x_train'].columns
    expected = bars.with_columns(
        ((pl.col('close').shift(-6) / pl.col('close') - 1) * 100).alias('expected')
    )
    dates = prepared['price_data_for_backtest'].select('datetime')
    target = dates.join(expected.select('datetime', 'expected'), on='datetime', maintain_order='left')['expected']
    np.testing.assert_allclose(prepared['y_test'].to_numpy(), target.to_numpy(), atol=1e-12)
    params['feature_drop_count'] = 2
    first = manifest.prepare_data(bars, params)
    dropped = params['_dropped_features']
    second = manifest.prepare_data(bars, dict(params))
    assert len(dropped) == 2
    assert not set(dropped) & set(first['x_train'].columns)
    assert first['x_train'].width == prepared['x_train'].width - 2
    assert_frame_equal(first['x_train'], second['x_train'])


def test_research_features_are_causal_and_do_not_share_sma_aliases(bars):
    manifest, params = sfd.manifest(), _params()
    params.update(price_sma_fast=20, volume_window=20, flow_window=20)
    full = _apply_feature_transforms(manifest, bars.lazy(), params).collect()
    prefix = _apply_feature_transforms(manifest, bars.head(400).lazy(), params).collect()
    assert_frame_equal(full.head(400), prefix)
    np.testing.assert_allclose(full['close_sma_20'].tail(100), bars['close'].rolling_mean(20).tail(100))
    np.testing.assert_allclose(full['volume_sma_20'].tail(100), bars['volume'].rolling_mean(20).tail(100))
    assert 'sma_20' not in full.columns
    model_params = set(inspect.signature(xgboost_regressor).parameters) - {'data'}
    assert model_params <= sfd.params().keys()


@pytest.mark.parametrize('environment', ['test', 'prod'])
def test_sweep_entrypoint(bars, monkeypatch, tmp_path, environment):
    fetched = []

    def load(**kwargs):
        fetched.append(kwargs)
        return bars

    space = {key: [values[0]] for key, values in sfd.params().items()}
    space['n_estimators'] = [8]
    monkeypatch.setattr(sfd, 'params', lambda: space)
    monkeypatch.setattr(HistoricalData, 'get_spot_klines', staticmethod(load))
    output = tmp_path / 'sweep'
    monkeypatch.setattr(sys, 'argv', ['xgboost_sweep.py', '--env', environment,
                                     '--permutations', '1', '--output', str(output)])
    runpy.run_path(str(ROOT / 'docs/examples/xgboost_sweep.py'), run_name='__main__')
    assert fetched[0]['kline_size'] == (7200 if environment == 'test' else 3600)
    log = pl.read_csv(output / 'results.csv')
    assert log.height == 1 and log['mae'].is_not_null().all()
    assert (output / 'metadata.json').is_file()
    assert len((output / 'round_data.jsonl').read_text().splitlines()) == 1


@pytest.mark.skipif(os.getenv('LIMEN_TEST_CUDA') != '1', reason='requires an explicit CUDA integration environment')
def test_cuda_backend_is_verified(prepared):
    result = xgboost_regressor(prepared, device='cuda:0', tree_method='hist', n_estimators=8)
    config = json.loads(result['_model'].model.get_booster().save_config())
    assert config['learner']['generic_param']['device'] == 'cuda:0'

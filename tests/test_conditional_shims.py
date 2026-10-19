import json
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from ruamel.yaml import YAML
from threadpoolctl import threadpool_limits

from limen import __version__
from limen.experiment import UniversalExperimentLoop, _round_storage
from limen.experiment._column_shims import apply_column_shims
from limen.scalers import RobustScaler
from limen.sfd.reference_architecture.direction_sizing import direction_sizing
from limen.yaml import CompiledSFD, build_search_strategy
from tests.test_direction_sizing import YAML_SPEC, native_manifest, recorded_source


SHIMS = [{'columns': ['close'], 'method': 'fractional_diff', 'd': 'auto', 'threshold': 0.01}]
PARAMS = {'conditional_size': True, 'folds': 3, 'min_train_samples': 2,
          'direction_params': {'max_iter': 1000}, 'seed': 42}


def wrapped_direction_sizing(data, **params):
    return direction_sizing(data, **params)


wrapped_direction_sizing.requires_trade_outcomes = True


def _manifest():
    return native_manifest().set_scaler(RobustScaler).set_feature_shims(SHIMS)


def _config():
    config = YAML(typ='safe').load(YAML_SPEC)
    config['metadata']['limen_version'] = __version__
    config['sfd']['manifest']['feature_shims'] = SHIMS
    config['sfd']['manifest']['scaler'] = {'class': 'limen.scalers.RobustScaler'}
    config['sfd']['params'] = {key: [value] for key, value in {**PARAMS, 'holding': 1800, 'funding_rate': 0.0001}.items()}
    config['uel']['n_permutations'] = 1
    return config


@pytest.fixture(autouse=True)
def _single_thread():
    with threadpool_limits(1):
        yield


@pytest.mark.parametrize('entry', ['native', 'staged', 'wrapper', 'yaml'])
def test_conditional_shims_native_and_compiled(entry, tmp_path):
    if entry == 'yaml':
        config = _config()
        loop = UniversalExperimentLoop(sfd=CompiledSFD(config), data=recorded_source(limit=288),
            experiment_dir=tmp_path, search_strategy=build_search_strategy(config), yaml_reference=config)
        loop.run('conditional', n_permutations=1, prep_each_round=True, post_processing=True, progress_bar=False)
        assert loop.experiment_log.height == 1
        assert loop.experiment_log['strict_mode_error'].drop_nulls().is_empty()
        with _round_storage.open_text(_round_storage.round_path(tmp_path)) as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        assert records[0]['learning_binding']['model']['conditional_size'] is True
        assert np.isfinite(records[0]['preds']).all()
        return
    manifest = _manifest()
    if entry == 'wrapper':
        manifest.with_reference_architecture(wrapped_direction_sizing)
    data = manifest.prepare_data(recorded_source(limit=120), {} if entry == 'staged' else dict(PARAMS))
    result = manifest.run_model(data, dict(PARAMS))
    assert result['_model'].conditional
    assert result['_preds'].shape == (data['x_test'].height,)
    assert np.isfinite(result['_preds']).all()


@pytest.mark.parametrize('count', [120, 336])
def test_inner_shim_prefix_identity_and_causality(count):
    manifest = _manifest()
    raw = recorded_source(limit=count)
    data = manifest.prepare_data(raw, dict(PARAMS))
    preparation = data['_fold_preparation']
    train_raw = raw.head(count * 6 // 10)
    assert len(preparation.row_ids) == train_raw.height
    blocks = np.array_split(np.arange(len(preparation.row_ids)), 3)
    earlier = [preparation.row_ids[index] for index in blocks[0]]
    predicted = [preparation.row_ids[index] for index in blocks[1]]
    actual = preparation.fit_transform(earlier, predicted)
    prefix = train_raw.head(len(earlier))
    _, state = apply_column_shims(prefix, SHIMS, {}, None, training=True, target='outcome')
    transformed = state.transform(train_raw.head(len(earlier) + len(predicted))).drop_nulls()
    expected_train = transformed.filter(pl.col('datetime') < train_raw['datetime'][len(earlier)])
    expected_predict = transformed.filter(pl.col('datetime') >= train_raw['datetime'][len(earlier)])
    scaler = RobustScaler(expected_train)
    np.testing.assert_allclose(actual.train, scaler.transform(expected_train).drop('datetime'), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.predict, scaler.transform(expected_predict).drop('datetime'), rtol=1e-12, atol=1e-12)
    identities = preparation.identities
    expected_ids = expected_train.select('datetime').join(identities, on='datetime')['row_id'].to_list()
    assert actual.train_rows == tuple(expected_ids)
    assert set(actual.train_rows) <= set(earlier)
    assert actual.predict_rows == tuple(predicted)
    if count == 336:
        assert len(actual.train_rows) < len(earlier)
        assert state.orders != data['_fitted_params']['_column_shims'].orders


def test_conditional_shim_replay_and_outer_state():
    manifest = _manifest()
    raw = recorded_source(limit=120)
    data = manifest.prepare_data(raw, dict(PARAMS))
    state = data['_fitted_params']['_column_shims']
    orders = dict(state.orders)
    preparation = data['_fold_preparation']
    calls = []

    def observe(train, predict):
        features = preparation.transform(train, predict)
        calls.append(features)
        return features

    data['_fold_preparation'] = replace(preparation, transform=observe)
    result = manifest.run_model(data, dict(PARAMS))
    assert len(calls) == 2
    assert state.orders == orders
    with patch('limen.experiment._column_shims.fit_order', side_effect=AssertionError('inference refit')):
        prepared, _ = manifest.sensor_input_prep(raw, data['_fitted_params'], dict(PARAMS))
        repeated, _ = manifest.sensor_input_prep(raw, data['_fitted_params'], dict(PARAMS))
    assert prepared.equals(repeated)
    assert result['_model'].conditional

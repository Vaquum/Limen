import copy
import json
from dataclasses import replace
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from ruamel.yaml import YAML
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from threadpoolctl import threadpool_limits

from limen import __version__
from limen.experiment import UniversalExperimentLoop, _round_storage
from limen.experiment._column_shims import apply_column_shims
from limen.experiment._fold_preparation import FoldFeatures, PreparedFolds
from limen.experiment._walk_forward_split import resolve_walk_forward_split
from limen.indicators import roc
from limen.inference import Trainer
from limen.scalers import RobustScaler
from limen.sfd.reference_architecture.direction_sizing import direction_sizing
from limen.targets import TradeOutcomeTarget
from limen.yaml import CompiledSFD, build_search_strategy
from tests.test_direction_sizing import YAML_SPEC, native_manifest, recorded_source
from tests.test_column_shims import _ffd_expected


SHIMS = [{'columns': ['close'], 'method': 'fractional_diff', 'd': 'auto', 'threshold': 0.01}]
PARAMS = {'conditional_size': True, 'folds': 3, 'min_train_samples': 2,
          'direction_params': {'max_iter': 1000}, 'seed': 42}


def wrapped_direction_sizing(data, **params):
    return direction_sizing(data, **params)


wrapped_direction_sizing.requires_trade_outcomes = True


def _manifest(shims=SHIMS):
    return native_manifest().set_scaler(RobustScaler).set_feature_shims(shims)


def _config():
    config = YAML(typ='safe').load(YAML_SPEC)
    config['metadata']['limen_version'] = __version__
    config['sfd']['manifest']['feature_shims'] = [{**SHIMS[0], 'd': '{order}'}]
    config['sfd']['manifest']['scaler'] = {'class': 'limen.scalers.RobustScaler'}
    config['sfd']['params'] = {key: [value] for key, value in {**PARAMS, 'holding': 1800, 'funding_rate': 0.0001, 'order': 'auto'}.items()}
    config['uel']['n_permutations'] = 1
    config['uel']['recording'] = dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True)
    return config


class VolumeExcludedOutcome(TradeOutcomeTarget):
    source_columns: ClassVar[list[str]] = ['volume']


@pytest.mark.parametrize('entry', ['native', 'yaml'])
def test_inner_folds_preserve_target_source_exclusions(entry):
    params = {**PARAMS, 'holding': 1800, 'funding_rate': 0.0001, 'order': 'auto', 'auto_pca': True, 'pca_k': 2}
    if entry == 'yaml':
        config = _config()
        config['sfd']['manifest']['pca_compression'] = {}
        with patch('limen.targets.TradeOutcomeTarget', VolumeExcludedOutcome):
            manifest = CompiledSFD(config).manifest()
    else:
        manifest = _manifest().with_target_label('outcome', VolumeExcludedOutcome).set_pca_compression()
    raw = recorded_source(limit=288)
    data = manifest.prepare_data(raw, dict(params))
    preparation = data['_fold_preparation']
    assert 'volume' not in preparation.raw_features.columns
    assert 'outcome' not in preparation.raw_features.columns
    blocks = np.array_split(np.arange(len(preparation.row_ids)), 3)
    earlier, predicted = [tuple(preparation.row_ids[index] for index in block) for block in blocks[:2]]
    features = preparation.fit_transform(earlier, predicted)
    history = preparation.identities.select('datetime').join(raw.drop('volume'), on='datetime', how='left', maintain_order='left')
    prefix = history.head(len(earlier))
    _, state = apply_column_shims(prefix, SHIMS, {}, None, training=True, target='outcome')
    transformed = state.transform(history.head(len(earlier) + len(predicted))).drop_nulls()
    cutoff = history['datetime'][len(earlier)]
    train = transformed.filter(pl.col('datetime') < cutoff)
    predict = transformed.filter(pl.col('datetime') >= cutoff)
    scaler = RobustScaler(train)
    compression = PCA(n_components=2, svd_solver='full', whiten=False)
    expected_train = compression.fit_transform(scaler.transform(train).drop('datetime').to_numpy())
    expected_predict = compression.transform(scaler.transform(predict).drop('datetime').to_numpy())
    np.testing.assert_allclose(features.train, expected_train, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(features.predict, expected_predict, rtol=1e-9, atol=1e-9)
    result = manifest.run_model(data, dict(params))
    assert np.isfinite(result['_preds']).all()


def test_fold_identity_validation_uses_linear_membership():
    comparisons = 0

    class Identity(str):
        __hash__ = str.__hash__

        def __eq__(self, other):
            nonlocal comparisons
            comparisons += 1
            return super().__eq__(other)

    raw = recorded_source(limit=512)
    rows = tuple(Identity(str(value)) for value in raw['datetime'])
    train = tuple(Identity(row) for row in rows[:384])
    predict = tuple(Identity(row) for row in rows[384:])
    expected = FoldFeatures(raw.head(384), raw.tail(128), rows[:384], rows[384:])
    preparation = PreparedFolds(raw, rows, raw.select('datetime'), True, lambda *_: expected)
    assert preparation.fit_transform(train, predict) is expected
    assert comparisons <= 10 * len(rows)
    with pytest.raises(ValueError, match='ordered original row identities'):
        preparation.fit_transform((*train, train[-1]), predict)


@pytest.fixture(autouse=True)
def _single_thread():
    with threadpool_limits(1):
        yield


@pytest.fixture(scope='module')
def campaign(tmp_path_factory):
    directory = tmp_path_factory.mktemp('conditional_shims')
    config = _config()
    raw = recorded_source(limit=288)
    loop = UniversalExperimentLoop(sfd=CompiledSFD(config), data=raw,
        experiment_dir=directory, search_strategy=build_search_strategy(config), yaml_reference=config)
    with threadpool_limits(1):
        loop.run('conditional', n_permutations=1, prep_each_round=True, post_processing=False, progress_bar=False)
    return directory, config, raw, loop


@pytest.mark.parametrize('entry', ['native', 'staged', 'wrapper', 'yaml', 'none', 'fixed', 'stateless', 'insufficient'])
def test_conditional_shims_native_and_compiled(entry, campaign):
    if entry == 'yaml':
        directory, _, _, loop = campaign
        assert loop.experiment_log.height == 1
        assert loop.experiment_log['strict_mode_error'].drop_nulls().is_empty()
        with _round_storage.open_text(_round_storage.round_path(directory)) as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        assert records[0]['learning_binding']['model']['conditional_size'] is True
        assert np.isfinite(records[0]['preds']).all()
        return
    shims = copy.deepcopy(SHIMS)
    if entry == 'native':
        del shims[0]['d']
    elif entry == 'none':
        shims = []
    elif entry == 'fixed':
        shims[0]['d'] = 1.0
    elif entry == 'stateless':
        shims = [{'columns': ['volume'], 'method': 'log1p'}]
    manifest = _manifest(shims)
    if entry == 'wrapper':
        manifest.with_reference_architecture(wrapped_direction_sizing)
    data = manifest.prepare_data(recorded_source(limit=120), {} if entry == 'staged' else dict(PARAMS))
    if entry == 'insufficient':
        with pytest.raises(ValueError, match='Chronological fold 1:'):
            manifest.run_model(data, {**PARAMS, 'folds': 24})
        return
    result = manifest.run_model(data, dict(PARAMS))
    assert result['_model'].conditional
    assert result['_preds'].shape == (data['x_test'].height,)
    assert np.isfinite(result['_preds']).all()


@pytest.mark.parametrize('count,pca', [(120, False), (336, True)])
def test_inner_shim_prefix_identity_and_causality(count, pca):
    import limen.experiment._column_shims as module

    shims = [{**SHIMS[0], 'columns': ['close', 'volume']},
             {'columns': ['high'], 'method': 'ratio', 'reference': 'open'}]
    manifest = _manifest(shims).add_indicator(roc, period=2).set_feature_ablation()
    params = {**PARAMS, 'feature_drop_count': 1, 'feature_drop_seed': 42}
    if pca:
        manifest.set_pca_compression()
        params.update(auto_pca=True, pca_k=2)
    raw = recorded_source(limit=count)
    data = manifest.prepare_data(raw, dict(params))
    preparation = data['_fold_preparation']
    train_raw = raw.head(count * 6 // 10)
    assert len(preparation.row_ids) == train_raw.height
    blocks = np.array_split(np.arange(len(preparation.row_ids)), 3)
    earlier = [preparation.row_ids[index] for index in blocks[0]]
    predicted = [preparation.row_ids[index] for index in blocks[1]]
    observed = []
    original = module.fit_order

    def observe(frame, column, *args):
        order = original(frame, column, *args)
        observed.append((frame.clone(), column, order))
        return order

    with patch.object(module, 'fit_order', observe):
        actual = preparation.fit_transform(earlier, predicted)
    prefix = train_raw.head(len(earlier))
    _, state = apply_column_shims(prefix, shims, {}, None, training=True, target='outcome')
    assert len(observed) == 2
    assert {column for _, column, _ in observed} == {'close', 'volume'}
    for fitted_rows, column, order in observed:
        assert fitted_rows.select(prefix.columns).equals(prefix)
        assert order == state.orders[column]
        assert 'outcome' not in fitted_rows.columns
    through_block = train_raw.head(len(earlier) + len(predicted))
    replacements = []
    widths = {}
    for column in ('close', 'volume'):
        values, width = _ffd_expected(through_block[column].to_list(), state.orders[column], 0.01)
        replacements.append(pl.Series(column, values, dtype=pl.Float64))
        widths[column] = width
    replacements.append((through_block['high'] / through_block['open'] - 1).alias('high'))
    transformed = through_block.with_columns(replacements).fill_nan(None).drop_nulls()
    expected_train = transformed.filter(pl.col('datetime') < train_raw['datetime'][len(earlier)])
    expected_predict = transformed.filter(pl.col('datetime') >= train_raw['datetime'][len(earlier)])
    scaler = RobustScaler(expected_train)
    expected_x = scaler.transform(expected_train).drop('datetime').to_numpy()
    expected_p = scaler.transform(expected_predict).drop('datetime').to_numpy()
    if pca:
        compression = PCA(n_components=2, svd_solver='full', whiten=False)
        expected_x = compression.fit_transform(expected_x)
        expected_p = compression.transform(expected_p)
    np.testing.assert_allclose(actual.train, expected_x, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(actual.predict, expected_p, rtol=1e-9, atol=1e-9)
    identities = preparation.identities
    expected_ids = expected_train.select('datetime').join(identities, on='datetime')['row_id'].to_list()
    assert actual.train_rows == tuple(expected_ids)
    assert set(actual.train_rows) <= set(earlier)
    assert actual.predict_rows == tuple(predicted)
    assert widths['close'] != widths['volume']
    extended_data = manifest.prepare_data(recorded_source(limit=count + 80), dict(params))
    same_prefix = extended_data['_fold_preparation'].fit_transform(earlier, predicted)
    assert actual.train_rows == same_prefix.train_rows
    assert actual.predict_rows == same_prefix.predict_rows
    np.testing.assert_array_equal(actual.train, same_prefix.train)
    np.testing.assert_array_equal(actual.predict, same_prefix.predict)
    cutoff = identities.filter(pl.col('row_id') == predicted[0])['available_at_ns'][0]

    def prefix_prediction(features, prepared):
        labels = pl.DataFrame({'row_id': features.train_rows}).join(prepared['_trade_labels'].rows, on='row_id', how='left', maintain_order='left')
        completed = labels.select(pl.max_horizontal('long_label_available_ns', 'short_label_available_ns')).to_series()
        eligible = (labels['long_available'] & labels['short_available'] & (completed < cutoff)).fill_null(False)
        selected = labels.filter(eligible)
        long, short = selected['long_return'].to_numpy(), selected['short_return'].to_numpy()
        direction = np.where((long > 0) & (long > short), 1, np.where((short > 0) & (short > long), -1, 0))
        model = LogisticRegression(random_state=42, max_iter=1000).fit(features.train.to_numpy()[eligible.to_numpy()], direction)
        return model.predict(features.predict.to_numpy()), selected.select('row_id', 'long_return', 'short_return')

    first_prediction, first_labels = prefix_prediction(actual, data)
    later_prediction, later_labels = prefix_prediction(same_prefix, extended_data)
    assert first_labels.equals(later_labels)
    np.testing.assert_array_equal(first_prediction, later_prediction)
    with pytest.raises(ValueError, match='ordered original row identities'):
        preparation.fit_transform(earlier[::-1], predicted)
    if count == 336:
        assert len(actual.train_rows) < len(earlier)
        assert state.orders != data['_fitted_params']['_column_shims'].orders


@pytest.mark.parametrize('proof', ['state', 'trainer', 'walk-forward'])
def test_conditional_shim_replay_and_outer_state(proof, campaign, tmp_path):
    if proof == 'trainer':
        directory, _, raw, loop = campaign
        identity = loop.experiment_log['id'][0]
        sensor = Trainer(directory, data=raw).train([identity])[0]
        rows = recorded_source(limit=384)
        with patch('limen.experiment._column_shims.fit_order', side_effect=AssertionError('inference refit')):
            first = sensor.predict_all(rows)
            second = sensor.predict_all(rows)
        assert first == second
        valid = [item for item in first if item.reason is None]
        assert valid and all(np.isfinite(item.prediction) for item in valid)
        features, _ = sensor._manifest.sensor_input_prep(rows, sensor._fitted_params, sensor.round_params)
        times = pl.Series('datetime', [item.datetime for item in valid], dtype=features.schema['datetime'])
        expected_input = features.join(times.to_frame(), on='datetime', how='inner', maintain_order='left').select(sensor._model.feature_names)
        assert expected_input.height == len(valid)
        expected = sensor._model.predict({'x_test': expected_input})['_preds']
        np.testing.assert_array_equal([item.prediction for item in valid], expected)
        return
    if proof == 'walk-forward':
        import limen.experiment._column_shims as module

        config = _config()
        declaration = config['sfd']['manifest']
        del declaration['split_dates']
        declaration['split_walk_forward'] = {'n_folds': 2, 'test_bars': 24, 'purge_bars': 2, 'embargo_bars': 1, 'anchored': True}
        raw = recorded_source(limit=336)
        loop = UniversalExperimentLoop(sfd=CompiledSFD(config), data=raw, experiment_dir=tmp_path,
            search_strategy=build_search_strategy(config), yaml_reference=config)
        fits = []
        original = module.fit_order

        def observe(frame, column, *args):
            fits.append(frame.select('datetime', column))
            return original(frame, column, *args)

        with patch.object(module, 'fit_order', observe):
            loop.run('folds', n_permutations=1, prep_each_round=True, post_processing=False, progress_bar=False)
        assert loop.fold_results.height == 2
        assert len(fits) == 6
        for fold in range(2):
            earlier = resolve_walk_forward_split(raw, loop.manifest.split_walk_forward, fold, loop.manifest.split_config, require_validation=False)[0]
            assert fits[fold * 3].equals(earlier.select('datetime', 'close'))
            assert fits[fold * 3 + 1].height < fits[fold * 3 + 2].height < fits[fold * 3].height
        return
    manifest = _manifest()
    raw = recorded_source(limit=120)
    data = manifest.prepare_data(raw, dict(PARAMS))
    state = data['_fitted_params']['_column_shims']
    orders = dict(state.orders)
    preparation = data['_fold_preparation']
    calls = []

    def observe(train, predict):
        features = preparation.transform(train, predict)
        calls.append((tuple(train), tuple(predict), features))
        return features

    data['_fold_preparation'] = replace(preparation, transform=observe)
    direction_fits, size_fits = [], []
    original_direction, original_size = LogisticRegression.fit, Ridge.fit

    def fit_direction(model, x, y, *args, **kwargs):
        direction_fits.append((np.asarray(x).copy(), np.asarray(y).copy()))
        return original_direction(model, x, y, *args, **kwargs)

    def fit_size(model, x, y, *args, **kwargs):
        size_fits.append(np.asarray(x).copy())
        return original_size(model, x, y, *args, **kwargs)

    with patch.object(LogisticRegression, 'fit', fit_direction), patch.object(Ridge, 'fit', fit_size):
        result = manifest.run_model(data, dict(PARAMS))
    assert len(calls) == 2
    assert len(direction_fits) == 3 and len(size_fits) == 1
    labels = preparation.identities.join(data['_trade_labels'].rows, on='row_id', how='left', maintain_order='left')
    labels = labels.with_columns(pl.max_horizontal('long_label_available_ns', 'short_label_available_ns').alias('completed'))
    for index, (_, predict, features) in enumerate(calls):
        start = labels.filter(pl.col('row_id') == predict[0])['available_at_ns'][0]
        eligible = labels.filter(pl.col('row_id').is_in(features.train_rows) & pl.col('long_available') & pl.col('short_available') & (pl.col('completed') < start))
        local = [features.train_rows.index(row) for row in eligible['row_id']]
        np.testing.assert_array_equal(direction_fits[index][0], features.train.to_numpy()[local])
        long, short = eligible['long_return'].to_numpy(), eligible['short_return'].to_numpy()
        expected_direction = np.where((long > 0) & (long > short), 1, np.where((short > 0) & (short > long), -1, 0))
        np.testing.assert_array_equal(direction_fits[index][1], expected_direction)
        assert eligible['completed'].max() < start
    first_block = np.array_split(np.arange(len(preparation.row_ids)), 3)[0]
    assert size_fits[0].shape[0] <= len(preparation.row_ids) - len(first_block)
    assert state.orders == orders
    with patch('limen.experiment._column_shims.fit_order', side_effect=AssertionError('inference refit')):
        prepared, _ = manifest.sensor_input_prep(raw, data['_fitted_params'], dict(PARAMS))
        repeated, _ = manifest.sensor_input_prep(raw, data['_fitted_params'], dict(PARAMS))
    assert prepared.equals(repeated)
    assert result['_model'].conditional

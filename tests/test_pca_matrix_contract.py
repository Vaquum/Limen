import copy
import logging
from datetime import date

import numpy as np
import polars as pl
import pytest
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

import limen.experiment.manifest_core as core
from limen.scalers import LinearScaler, SCALER_REGISTRY
from limen.inference import Trainer
from limen.yaml import CompiledSFD
from tests.test_column_shims import FIXTURE, SPLITS, _assert_prepared_equal, _config, _native, _loop, _run, _records


SCALERS = ('robust', 'logreg', 'linear', 'rank_gauss', 'causal_rolling_robust')
CASES = [(name, selected) for name in SCALERS for selected in (False, True)] + [('caller', False)]
CASES += [('grid', False)] + [(name, 'conditional') for name in SCALERS] + [(name, 'conditional_auto') for name in SCALERS]
PARAMS = {'alpha': 0.3, 'solver': 'svd', 'auto_pca': True, 'pca_k': 1}


class CallerScaler:
    def __init__(self, train):
        self.fitted = LinearScaler(train)

    def transform(self, data):
        return self.fitted.transform(data)


@pytest.fixture(scope='module')
def recorded():
    return pl.read_parquet(FIXTURE, n_rows=336)


@pytest.fixture(autouse=True)
def single_thread():
    with threadpool_limits(limits=1):
        yield


def _manifest(name='robust', selected=False):
    extras = {'window': 20, 'min_samples': 5} if name == 'causal_rolling_robust' else {}
    manifest = _native().set_strict_mode(True)
    if selected:
        manifest.set_scaler_from_params(extra_params=extras)
    else:
        manifest.set_scaler(CallerScaler if name == 'caller' else SCALER_REGISTRY[name],
                            param_name='caller_scaler' if name == 'caller' else '_scaler', extra_params=extras)
    return manifest.set_pca_compression(scaler_param_name='caller_scaler' if name == 'caller' else '_scaler')


@pytest.mark.parametrize(('name', 'selected'), CASES)
def test_recorded_pca_matrix_contract(recorded, name, selected, monkeypatch, tmp_path):
    if name == 'grid':
        _retained_grid(tmp_path, monkeypatch)
        return
    if isinstance(selected, str):
        _conditional(name, selected == 'conditional_auto')
        return
    manifest = _manifest(name, selected)
    params = PARAMS | {'scaler_type': name}
    plain = manifest.prepare_data(recorded, params | {'auto_pca': False})
    cls = CallerScaler if name == 'caller' else SCALER_REGISTRY[name]
    init, fit = cls.__init__, PCA.fit_transform
    constructed, fitted = [], []

    def observe_init(instance, train, **kwargs):
        constructed.append(train.height)
        init(instance, train, **kwargs)

    def observe_fit(instance, matrix, **kwargs):
        fitted.append(matrix.shape[0])
        return fit(instance, matrix, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(cls, '__init__', observe_init)
        patch.setattr(PCA, 'fit_transform', observe_fit)
        actual = manifest.prepare_data(recorded, params)
    assert constructed == [168] and fitted == [167]
    assert [actual[f'x_{split}'].height for split in SPLITS] == [167, 47, 47]
    columns = plain['x_train'].columns
    assert actual['_pca_input_feature_names'] == columns
    matrix = plain['x_train'].to_numpy()
    assert np.isfinite(matrix).all() and np.all(np.std(matrix, axis=0) > 0)
    oracle = PCA(n_components=1, svd_solver='full', whiten=False)
    expected_train = oracle.fit_transform(matrix)
    for split in SPLITS:
        expected = expected_train if split == 'train' else oracle.transform(plain[f'x_{split}'].to_numpy())
        if name == 'robust':
            assert np.array_equal(actual[f'x_{split}'].to_numpy(), expected)
        else:
            np.testing.assert_allclose(actual[f'x_{split}'].to_numpy(), expected, rtol=1e-9, atol=1e-9)
        assert actual[f'y_{split}'].equals(plain[f'y_{split}'])
    assert actual['_alignment'] == plain['_alignment']
    assert actual['price_data_for_backtest'].equals(plain['price_data_for_backtest'])
    shorter = manifest.prepare_data(recorded.head(240), params)
    assert np.array_equal(shorter['_pca'].components_, actual['_pca'].components_)
    assert shorter['x_train'].equals(actual['x_train'])
    logging.getLogger(__name__).info('scaler=%s selected=%s std=%s loadings=%s', name, selected,
        dict(zip(columns, np.std(matrix, axis=0).tolist(), strict=True)), actual['_pca'].components_[0].tolist())
    if name != 'caller':
        config = _config()
        declaration = config['sfd']['manifest']
        declaration.update(strict_mode=True, pca_compression={})
        declaration['scaler'] = ({'from_params': 'scaler_type'} if selected else
                                  {'class': f'limen.scalers.{cls.__name__}'})
        if name == 'causal_rolling_robust':
            declaration['scaler']['params'] = {'window': 20, 'min_samples': 5}
        compiled = CompiledSFD(config).manifest().prepare_data(recorded, params)
        _assert_prepared_equal(actual, compiled)


def _retained_grid(tmp_path, monkeypatch):
    raw = pl.read_parquet(FIXTURE, n_rows=2856)
    config = _config()
    declaration = config['sfd']['manifest']
    dates = ('2024-01-01', '2024-03-01', '2024-03-01', '2024-04-12', '2024-04-12', '2024-04-29')
    declaration['split_dates'].update(dict(zip(declaration['split_dates'], dates, strict=False)))
    declaration['split_dates']['test_predict_guard'] = False
    declaration.update(strict_mode=True, scaler={'from_params': 'scaler_type'}, pca_compression={})
    config['sfd']['params'].update(scaler_type=list(SCALERS), auto_pca=[False, True], pca_k=[1])
    config['uel']['n_permutations'] = 10
    counts = [raw.filter((pl.col('datetime').dt.date() >= date.fromisoformat(dates[i])) &
                        (pl.col('datetime').dt.date() < date.fromisoformat(dates[i + 1]))).height for i in (0, 2, 4)]
    assert counts == [1440, 1008, 408]
    directory = tmp_path / 'mixed-grid'
    loop = _loop(config, raw, directory)
    _run(loop, 10)
    records = _records(directory)
    assert len(records) == 10 and loop.experiment_log.height == 10
    assert loop.experiment_log['strict_mode_error'].drop_nulls().is_empty()
    assert {(r['round_params']['scaler_type'], r['round_params']['auto_pca']) for r in records} == {
        (name, enabled) for name in SCALERS for enabled in (False, True)}
    trainer = Trainer(directory, data=raw)
    sensors = trainer.train([record['round_id'] for record in records])
    assert len(sensors) == 10
    for record, sensor in zip(records, sensors, strict=True):
        params = record['round_params']
        prepared = CompiledSFD(config).manifest().prepare_data(raw, params)
        assert [prepared[f'x_{split}'].height for split in SPLITS] == [1439, 1007, 407]
        assert len(record['preds']) == 407 and np.isfinite(record['preds']).all()
        scaler = sensor._fitted_params['_scaler']
        transform = type(scaler).transform
        rotation = PCA.transform
        observed = {'scaler': 0, 'pca': 0}

        def frozen_scale(instance, data, observed=observed, transform=transform):
            observed['scaler'] += 1
            return transform(instance, data)

        def frozen_rotate(instance, data, observed=observed, rotation=rotation):
            observed['pca'] += 1
            return rotation(instance, data)

        def no_fit(*args, **kwargs):
            raise AssertionError('Frozen prediction attempted construction or fitting')

        with monkeypatch.context() as patch:
            for cls in SCALER_REGISTRY.values():
                patch.setattr(cls, '__init__', no_fit)
            patch.setattr(type(scaler), 'transform', frozen_scale)
            patch.setattr(core, '_pca_transformer', no_fit)
            patch.setattr(PCA, 'fit', no_fit)
            patch.setattr(PCA, 'fit_transform', no_fit)
            patch.setattr(PCA, 'transform', frozen_rotate)
            predictions = sensor.predict_all(raw)
            replay, _ = sensor._manifest.sensor_input_prep(raw, sensor._fitted_params, params)
            if params['auto_pca'] and params['scaler_type'] == 'linear':
                instant = raw['datetime'][-10]
                damaged = raw.with_columns(pl.when(pl.col('datetime') == instant)
                    .then(float('nan')).otherwise(pl.col('open')).alias('open'))
                invalid = sensor.predict_all(damaged)
                assert [p.reason for p in invalid if p.datetime == instant] == ['null-features']
                damaged = damaged.with_columns(pl.col('open').fill_nan(float('inf')))
                with pytest.raises(ValueError, match='infinity'):
                    sensor._manifest.sensor_input_prep(damaged, sensor._fitted_params, params)
                assert all(p.reason == 'sensor-error' for p in sensor.predict_all(damaged))
        assert observed['scaler'] > 0
        assert (observed['pca'] > 0) is params['auto_pca']
        assert len(predictions) == raw.height == replay.height
        assert {p.datetime for p in predictions if p.reason is None} == set(raw.tail(408)['datetime'])
        assert {p.datetime for p in predictions if p.reason == 'inside-training-window'} == set(raw.head(2448)['datetime'])
        assert replay.drop('datetime').null_count().sum_horizontal()[0] == 0
        test_times = raw.tail(408)['datetime'].to_list()
        eligible = {p.datetime: p for p in predictions if p.datetime in set(test_times)}
        assert len(eligible) == 408 and all(p.reason is None for p in eligible.values())
        retained = prepared['price_data_for_backtest'].select('datetime')
        aligned = replay.join(retained, on='datetime', how='semi', maintain_order='left')
        assert aligned.height == 407 and aligned.drop('datetime').columns == prepared['x_test'].columns
        np.testing.assert_allclose(aligned.drop('datetime').to_numpy(), prepared['x_test'].to_numpy(), rtol=1e-8, atol=1e-10)
        assert aligned.drop('datetime').null_count().sum_horizontal()[0] == 0
        actual = [eligible[t].prediction for t in retained['datetime']]
        np.testing.assert_allclose(actual, record['preds'], rtol=1e-8, atol=1e-10)


def _conditional(name, automatic):
    from tests.test_conditional_shims import PARAMS as CONDITIONAL_PARAMS, SHIMS, _manifest as conditional_manifest
    from tests.test_direction_sizing import recorded_source
    from limen.experiment._column_shims import apply_column_shims

    raw = recorded_source(limit=288)
    shims = SHIMS if automatic else []
    cls = SCALER_REGISTRY[name]
    extras = {'window': 20, 'min_samples': 5} if name == 'causal_rolling_robust' else {}
    manifest = conditional_manifest(shims).set_strict_mode(True).set_scaler(cls, extra_params=extras).set_pca_compression()
    params = CONDITIONAL_PARAMS | PARAMS | {'sizing_params': {'alpha': 0.3, 'solver': 'svd'}}
    plain = manifest.prepare_data(raw, params | {'auto_pca': False})
    data = manifest.prepare_data(raw, params)
    oracle = PCA(n_components=1, svd_solver='full', whiten=False)
    np.testing.assert_allclose(data['x_train'], oracle.fit_transform(plain['x_train']), rtol=1e-9, atol=1e-9)
    for split in ('val', 'test'):
        np.testing.assert_allclose(data[f'x_{split}'], oracle.transform(plain[f'x_{split}']), rtol=1e-9, atol=1e-9)
    preparation = data['_fold_preparation']
    blocks = np.array_split(np.arange(len(preparation.row_ids)), 3)
    for block in (1, 2):
        earlier = tuple(preparation.row_ids[index] for index in np.concatenate(blocks[:block]))
        predicted = tuple(preparation.row_ids[index] for index in blocks[block])
        actual = preparation.fit_transform(earlier, predicted)
        history = preparation.identities.select('datetime').join(raw, on='datetime', how='left', maintain_order='left')
        prefix = history.head(len(earlier))
        _, fitted = apply_column_shims(prefix, shims, {}, None, training=True, target='outcome')
        through = history.head(len(earlier) + len(predicted))
        transformed = through if fitted is None else fitted.transform(through)
        transformed = transformed.fill_nan(None).drop_nulls()
        cutoff = history['datetime'][len(earlier)]
        train = transformed.filter(pl.col('datetime') < cutoff)
        future = transformed.filter(pl.col('datetime') >= cutoff)
        scaler = cls(train, **extras)
        x_train = scaler.transform(train).drop('datetime').to_numpy()
        context = int(getattr(scaler, 'context_rows', 0))
        future = scaler.transform(pl.concat([train.tail(context), future])).slice(min(context, train.height))
        train_scaled = pl.DataFrame(x_train).fill_nan(None).drop_nulls().to_numpy() if automatic else x_train
        future_scaled = future.drop('datetime').fill_nan(None).drop_nulls().to_numpy() if automatic else future.drop('datetime').to_numpy()
        oracle = PCA(n_components=1, svd_solver='full', whiten=False)
        expected = oracle.fit_transform(train_scaled)
        assert actual.train.height and actual.predict.height and np.isfinite(expected).all()
        np.testing.assert_allclose(actual.train, expected, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(actual.predict, oracle.transform(future_scaled), rtol=1e-9, atol=1e-9)
    result = manifest.run_model(data, params)
    assert np.isfinite(result['_preds']).all() and result['_preds'].shape == (data['x_test'].height,)


@pytest.fixture(scope='module')
def prepared_boundary(recorded):
    captured = []
    original = core._apply_pca_compression

    def observe(manifest, splits, params, fitted):
        captured.extend(frame.clone() for frame in splits)
        return original(manifest, splits, params, fitted)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(core, '_apply_pca_compression', observe)
        data = _manifest().prepare_data(recorded, PARAMS | {'auto_pca': False})
    return captured, data['_fitted_params']


@pytest.mark.parametrize('case', ('no_scaler', 'wrong_key', 'missing_key', 'noncallable', 'target_key',
                                  'nonnumeric', 'nonfinite', 'nan', 'empty_train', 'no_features', 'enable', 'invalid_k',
                                  'missing_k', 'zero_k', 'large_k', 'float_k', 'train_k', 'missing_column',
                                  'finite_order', 'dtype_order', 'empty_train_future', 'disabled', 'absent',
                                  'empty_future', 'integer', 'decimal', 'end_to_end', 'fold_policy'))
def test_pca_matrix_refusals(prepared_boundary, recorded, monkeypatch, case):
    source, fitted = prepared_boundary
    frames, fitted = [frame.clone() for frame in source], copy.deepcopy(fitted)
    manifest, params, error = _manifest(), dict(PARAMS), ValueError
    if case in ('disabled', 'absent'):
        if case == 'disabled':
            params['auto_pca'] = False
        else:
            manifest.pca_compression_config = None
        manifest.scaler = None
        actual, state = core._apply_pca_compression(manifest, frames, params, fitted)
        assert actual is frames and state is fitted and '_pca' not in state
        return
    if case in ('empty_future', 'integer', 'decimal'):
        if case == 'empty_future':
            frames[1:] = [frame.head(0) for frame in frames[1:]]
        else:
            dtype = pl.Int64 if case == 'integer' else pl.Decimal(precision=30, scale=5)
            frames = [frame.with_columns(pl.col('open').cast(dtype)) for frame in frames]
        actual, state = core._apply_pca_compression(manifest, frames, params, fitted)
        assert actual[0].height and state['_pca_n_components'] == 1
        assert [frame.height for frame in actual] == [frame.height for frame in frames]
        return
    if case == 'end_to_end':
        _ordinary_invalid_rows(recorded, monkeypatch)
        return
    if case == 'fold_policy':
        _fold_invalid_rows(recorded)
        return
    match = ''
    if case == 'no_scaler':
        manifest.scaler, match = None, 'configured fitted scaler'
    elif case in ('wrong_key', 'target_key'):
        manifest.set_pca_compression(scaler_param_name='absent' if case == 'wrong_key' else '_target_cls_next_return')
        match = 'not a configured scaler output'
    elif case == 'missing_key':
        fitted.pop('_scaler')
        match = 'could not find a fitted scaler'
    elif case == 'noncallable':
        fitted['_scaler'], match = 'not a scaler', 'callable transform'
    elif case == 'nonnumeric':
        frames[0] = frames[0].with_columns(pl.col('open').cast(pl.String))
        match = 'numeric feature columns; split 0'
    elif case in ('nonfinite', 'nan'):
        frames[1] = frames[1].with_columns(pl.lit(float('inf') if case == 'nonfinite' else float('nan')).alias('open'))
        match = r"finite feature columns; split 1 has nonfinite columns: \['open'\]"
    elif case == 'empty_train':
        frames[0], match = frames[0].head(0), 'train row count'
    elif case == 'no_features':
        frames[0] = frames[0].select('datetime', 'next_return')
        match = 'at least one feature'
    elif case == 'enable':
        params['auto_pca'], error, match = 1, TypeError, 'must be a bool'
    elif case == 'missing_k':
        params.pop('pca_k')
        manifest.scaler, match = None, 'is required'
    elif case in ('zero_k', 'large_k', 'float_k', 'train_k'):
        params['pca_k'] = {'zero_k': 0, 'large_k': 999, 'float_k': 1.0, 'train_k': 2}[case]
        match = 'must be an int' if case == 'float_k' else 'feature count'
        if case == 'train_k':
            frames[0], match = frames[0].head(1), 'train row count'
    elif case == 'missing_column':
        frames[1] = frames[1].drop('open')
        params['pca_k'], error, match = False, pl.exceptions.ColumnNotFoundError, 'open'
    elif case in ('finite_order', 'dtype_order', 'empty_train_future'):
        params['pca_k'] = False
        if case == 'dtype_order':
            frames[1] = frames[1].with_columns(pl.col('open').cast(pl.String), pl.lit(float('inf')).alias('high'))
            match = 'numeric feature columns; split 1'
        else:
            frames[1] = frames[1].with_columns(pl.lit(float('inf')).alias('open'), pl.lit(float('-inf')).alias('high'))
            match = r"finite feature columns; split 1 has nonfinite columns: \['open', 'high'\]"
            if case == 'empty_train_future':
                frames[0] = frames[0].head(0)
    else:
        params['pca_k'], match = True, 'must be an int'
    with pytest.raises(error, match=match):
        core._apply_pca_compression(manifest, frames, params, fitted)


def _ordinary_invalid_rows(recorded, monkeypatch):
    plain = _manifest().prepare_data(recorded, PARAMS)
    timestamp = recorded['datetime'][180]
    for value in (float('inf'), float('nan')):
        damaged = recorded.with_columns(pl.when(pl.col('datetime') == timestamp).then(value).otherwise(pl.col('open')).alias('open'))
        if np.isinf(value):
            with pytest.raises(ValueError, match='finite feature columns; split 1'):
                _manifest().prepare_data(damaged, PARAMS)
        else:
            from limen.experiment.errors import StrictModeError
            with pytest.raises(StrictModeError, match='Unexpected nulls in val split'):
                _manifest().prepare_data(damaged, PARAMS)
            actual = _manifest().set_strict_mode(False).prepare_data(damaged, PARAMS)
            assert actual['x_train'].equals(plain['x_train'])
            assert actual['x_val'].height == plain['x_val'].height - 1
            manifest = _manifest()
            replay, _ = manifest.sensor_input_prep(damaged, plain['_fitted_params'], PARAMS)
            row = replay.filter(pl.col('datetime') == timestamp)
            assert row.height == 1 and row['pc_0'].null_count() == 1
    with monkeypatch.context() as patch:
        def no_constructor(*args, **kwargs):
            raise AssertionError('PCA constructor must follow finalized input validation')
        patch.setattr(core, '_pca_transformer', no_constructor)
        damaged = recorded.with_columns(pl.when(pl.col('datetime') == timestamp).then(float('inf')).otherwise(pl.col('open')).alias('open'))
        with pytest.raises(ValueError, match='finite feature columns; split 1'):
            _manifest().prepare_data(damaged, PARAMS)


def _fold_invalid_rows(recorded):
    from functools import partial
    from limen.experiment._fold_preparation import _scaled

    train, future = recorded.head(168), recorded.slice(168, 48)
    manifest = _manifest()
    scale = partial(core._apply_scaler, manifest, round_params=PARAMS)
    compress = partial(core._apply_pca_compression, manifest, round_params=PARAMS)
    broken = future.with_columns(pl.lit(float('inf')).alias('open'))
    with pytest.raises(ValueError, match='finite feature columns; split 1'):
        _scaled(train, broken, scale, compress, available=False)
    with pytest.raises(ValueError, match='Fold preprocessing produced nonfinite features'):
        _scaled(train, broken, scale, compress, available=True)
    broken = future.with_columns(pl.when(pl.col('datetime') == future['datetime'][5]).then(float('nan')).otherwise(pl.col('open')).alias('open'))
    with pytest.raises(ValueError, match='finite feature columns; split 1'):
        _scaled(train, broken, scale, compress, available=False)
    _, actual = _scaled(train, broken, scale, compress, available=True)
    assert actual.height == future.height - 1 and np.isfinite(actual['pc_0'].to_numpy()).all()

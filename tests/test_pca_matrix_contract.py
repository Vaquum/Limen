import copy

import numpy as np
import polars as pl
import pytest
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

import limen.experiment.manifest_core as core
from limen.scalers import LinearScaler, SCALER_REGISTRY
from limen.yaml import CompiledSFD
from tests.test_column_shims import FIXTURE, SPLITS, _assert_prepared_equal, _config, _native


SCALERS = ('robust', 'logreg', 'linear', 'rank_gauss', 'causal_rolling_robust')
CASES = [(name, selected) for name in SCALERS for selected in (False, True)] + [('caller', False)]
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
def test_recorded_pca_matrix_contract(recorded, name, selected, monkeypatch):
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
    print(name, selected, dict(zip(columns, np.std(matrix, axis=0).tolist(), strict=True)),
          actual['_pca'].components_[0].tolist())
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
                                  'nonnumeric', 'nonfinite', 'nan', 'empty_train', 'no_features', 'enable', 'invalid_k'))
def test_pca_matrix_refusals(prepared_boundary, case):
    source, fitted = prepared_boundary
    frames, fitted = [frame.clone() for frame in source], copy.deepcopy(fitted)
    manifest, params, error = _manifest(), dict(PARAMS), ValueError
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
    else:
        params['pca_k'], match = True, 'must be an int'
    with pytest.raises(error, match=match):
        core._apply_pca_compression(manifest, frames, params, fitted)

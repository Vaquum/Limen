import copy
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from threadpoolctl import threadpool_limits

from limen import __version__
from limen.data import HistoricalData
from limen.experiment import MLManifest, UniversalExperimentLoop
from limen.experiment._column_shims import apply_column_shims, parse_shims
from limen.experiment._walk_forward_split import resolve_walk_forward_split
from limen.features import fractional_diff
from limen.indicators import window_return
from limen.inference import Trainer
from limen.scalers import RobustScaler
from limen.sfd.reference_architecture import ridge_regressor
from limen.targets import NextReturnTarget
from limen.yaml import CompiledSFD, build_search_strategy, validate


FIXTURE = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'
SPLITS = ('train', 'val', 'test')
DATES = ('2024-01-01', '2024-01-08', '2024-01-08', '2024-01-10',
         '2024-01-10', '2024-01-12')
DATE_KEYS = ('train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end')


@pytest.fixture(scope='module')
def recorded_shim_bars():
    return pl.read_parquet(FIXTURE, n_rows=336)


@pytest.fixture(autouse=True)
def _single_native_thread():
    with threadpool_limits(limits=1):
        yield


def _config(shims=None, *, scaler=False):
    manifest = {
        'type': 'ml',
        'data_source': {'method': 'limen.data.HistoricalData.get_any_file',
                        'params': {'file_path_or_url': str(FIXTURE)}},
        'split_dates': {**dict(zip(DATE_KEYS, DATES, strict=True)), 'test_predict_guard': False},
        'target': {'name': 'next_return', 'class': 'limen.targets.NextReturnTarget',
                   'transform_params': {'periods': 1, 'scale': 100.0}},
        'reference_architecture': 'limen.sfd.reference_architecture.ridge_regressor',
        'backtest': {'fee_bps': 5.0, 'slip_bps': 3.0},
    }
    if shims is not None:
        manifest['feature_shims'] = copy.deepcopy(shims)
    if scaler:
        manifest['scaler'] = {'class': 'limen.scalers.RobustScaler'}
    return {
        'schema_version': '1.0',
        'metadata': {'name': 'recorded_column_shims', 'limen_version': __version__, 'mode': 'development'},
        'sfd': {'manifest': manifest, 'params': {'alpha': [0.3], 'solver': ['svd']}},
        'uel': {'n_permutations': 1, 'prep_each_round': True, 'search_strategy': {'type': 'grid'}},
    }


def _native(shims=None, *, scaler=False):
    manifest = (MLManifest()
        .set_data_source(HistoricalData.get_any_file, params={'file_path_or_url': str(FIXTURE)})
        .set_split_dates(*[date.fromisoformat(value) for value in DATES], test_predict_guard=False)
        .with_target_label('next_return', NextReturnTarget, transform_params={'periods': 1, 'scale': 100.0})
        .with_reference_architecture(ridge_regressor)
        .set_backtest_config(fee_bps=5.0, slip_bps=3.0))
    if shims is not None:
        manifest.set_feature_shims(shims)
    if scaler:
        manifest.set_scaler(RobustScaler)
    return manifest


def _assert_prepared_equal(left, right):
    assert set(left) == set(right)
    for split in SPLITS:
        assert left[f'x_{split}'].equals(right[f'x_{split}'])
        assert left[f'y_{split}'].equals(right[f'y_{split}'])
    assert left['_feature_names'] == right['_feature_names']
    assert left['_alignment'] == right['_alignment']
    assert left['price_data_for_backtest'].equals(right['price_data_for_backtest'])


def _raw_split(bars, index):
    start, end = [date.fromisoformat(value) for value in DATES[2 * index:2 * index + 2]]
    return bars.filter((pl.col('datetime').dt.date() >= start) & (pl.col('datetime').dt.date() < end))


def _numeric_outputs(data):
    return data.with_columns(
        ((pl.col('high') + pl.col('low')) / 2).alias('custom_mid'),
        (pl.col('close') - pl.col('open')).alias('custom_move'),
    )


def _ffd_expected(values, order, threshold):
    weights = [1.0]
    for offset in range(1, len(values) + 1):
        candidate = weights[-1] * (offset - order - 1.0) / offset
        if abs(candidate) < threshold:
            break
        weights.append(candidate)
    width = len(weights)
    expected = [None] * min(width - 1, len(values))
    for index in range(width - 1, len(values)):
        expected.append(sum(weight * values[index - offset] for offset, weight in enumerate(weights)))
    return expected, width


def _loop(config, bars, directory):
    return UniversalExperimentLoop(sfd=CompiledSFD(config), data=bars, experiment_dir=directory,
        yaml_reference=config, search_strategy=build_search_strategy(config), checkpoint_interval=1)


def _run(loop, count=1, **kwargs):
    loop.run('recorded_column_shims', n_permutations=count, prep_each_round=True,
             progress_bar=False, record_model_outputs=True, **kwargs)


def _records(directory):
    return [json.loads(line) for line in (directory / 'round_data.jsonl').read_text().splitlines()]


def test_declared_column_shims_match_native_yaml_and_recorded_formulas(recorded_shim_bars):
    shims = [{'columns': ['open', 'high'], 'method': 'ratio', 'reference': 'close'},
             {'columns': ['low'], 'method': 'scale', 'reference': 'close'},
             {'columns': ['volume'], 'method': 'log1p'}]
    before = recorded_shim_bars.clone()
    config = _config(shims)
    assert validate(config).valid
    native = _native(shims).prepare_data(recorded_shim_bars, {})
    compiled = CompiledSFD(config).manifest().prepare_data(recorded_shim_bars, {})
    _assert_prepared_equal(native, compiled)
    for index, split in enumerate(SPLITS):
        raw = _raw_split(recorded_shim_bars, index).head(-1)
        features = native[f'x_{split}']
        assert features.columns == raw.drop('datetime').columns
        np.testing.assert_allclose(features['open'], raw['open'] / raw['close'] - 1, rtol=1e-13)
        np.testing.assert_allclose(features['high'], raw['high'] / raw['close'] - 1, rtol=1e-13)
        np.testing.assert_allclose(features['low'], raw['low'] / raw['close'], rtol=1e-13)
        np.testing.assert_allclose(features['volume'], np.log1p(raw['volume'].to_numpy()), rtol=1e-13)
        for name in set(features.columns) - {'open', 'high', 'low', 'volume'}:
            assert features[name].equals(raw[name])
        expected_labels = (_raw_split(recorded_shim_bars, index)['close'].shift(-1) /
                           _raw_split(recorded_shim_bars, index)['close'] - 1) * 100
        np.testing.assert_allclose(native[f'y_{split}'], expected_labels.head(-1), atol=1e-12)
    assert recorded_shim_bars.equals(before)
    assert native['_fitted_params']['_column_shims'].specs == compiled['_fitted_params']['_column_shims'].specs

    custom = (_native([{'columns': ['custom_mid', 'custom_move'], 'method': 'scale',
                       'reference': 'close'}]).add_feature(_numeric_outputs)
              .prepare_data(recorded_shim_bars, {}))
    raw_train = _raw_split(recorded_shim_bars, 0).head(-1)
    assert custom['x_train'].columns == raw_train.drop('datetime').columns + ['custom_mid', 'custom_move']
    np.testing.assert_allclose(custom['x_train']['custom_mid'],
                               (raw_train['high'] + raw_train['low']) / 2 / raw_train['close'])
    np.testing.assert_allclose(custom['x_train']['custom_move'],
                               (raw_train['close'] - raw_train['open']) / raw_train['close'])

    period_shims = [{'columns': ['ret_{period}'], 'method': 'scale', 'reference': 'close'}]
    period_config = _config(period_shims)
    period_config['sfd']['manifest']['indicators'] = [
        {'func': 'limen.indicators.window_return', 'params': {'period': '{period}'}}]
    period_config['sfd']['params']['period'] = [2]
    assert validate(period_config).valid
    period_native = (_native(period_shims).add_indicator(window_return, period='period')
                     .prepare_data(recorded_shim_bars, {'period': 2}))
    period_yaml = CompiledSFD(period_config).manifest().prepare_data(recorded_shim_bars, {'period': 2})
    _assert_prepared_equal(period_native, period_yaml)
    raw_period = _raw_split(recorded_shim_bars, 0)
    expected_return = (raw_period['close'] / raw_period['close'].shift(2) - 1) / raw_period['close']
    np.testing.assert_allclose(period_native['x_train']['ret_2'], expected_return.slice(2).head(-1))
    assert period_native['x_train'].columns == raw_train.drop('datetime').columns + ['ret_2']
    assert window_return(recorded_shim_bars, period=2)['close'].equals(before['close'])

    simultaneous, _ = apply_column_shims(recorded_shim_bars, parse_shims([
        {'columns': ['close'], 'method': 'log1p'},
        {'columns': ['open'], 'method': 'ratio', 'reference': 'close'},
    ]), {}, None, training=True, target='next_return')
    np.testing.assert_allclose(simultaneous['close'], np.log1p(recorded_shim_bars['close'].to_numpy()))
    np.testing.assert_allclose(simultaneous['open'], recorded_shim_bars['open'] / recorded_shim_bars['close'] - 1)
    for name in set(recorded_shim_bars.columns) - {'close', 'open'}:
        assert simultaneous[name].equals(recorded_shim_bars[name])


@pytest.mark.parametrize('order', (0.0, 0.5, 1.0))
def test_fractional_shim_matches_causal_fixed_width_definition(recorded_shim_bars, order):
    specs = parse_shims([{'columns': ['close', 'volume'], 'method': 'fractional_diff',
                          'd': order, 'threshold': 0.01}])
    actual, fitted = apply_column_shims(recorded_shim_bars, specs, {}, None, training=True, target='next_return')
    assert actual.columns == recorded_shim_bars.columns
    assert actual['datetime'].equals(recorded_shim_bars['datetime'])
    assert fitted.orders == {'close': order, 'volume': order}
    for column in ('close', 'volume'):
        expected, width = _ffd_expected(recorded_shim_bars[column].to_list(), order, 0.01)
        assert actual[column].null_count() == width - 1
        np.testing.assert_allclose(actual[column].to_numpy(), np.asarray(expected, dtype=float),
                                   rtol=1e-12, atol=1e-9, equal_nan=True)
    for column in set(recorded_shim_bars.columns) - {'close', 'volume'}:
        assert actual[column].equals(recorded_shim_bars[column])


def test_auto_fractional_shim_fits_train_only_and_keeps_past_outputs(recorded_shim_bars):
    shims = [{'columns': ['close'], 'method': 'fractional_diff', 'd': 'auto', 'threshold': 0.01}]
    manifest = _native(shims)
    first = manifest.prepare_data(recorded_shim_bars, {})
    cutoff = date(2024, 1, 10)
    changed = recorded_shim_bars.with_columns([
        pl.when(pl.col('datetime').dt.date() >= cutoff).then(pl.col(name) * 1.07)
          .otherwise(pl.col(name)).alias(name) for name in ('open', 'high', 'low', 'close')
    ])
    second = manifest.prepare_data(changed, {})
    state = first['_fitted_params']['_column_shims']
    assert state.orders == second['_fitted_params']['_column_shims'].orders
    assert 0 <= state.orders['close'] <= 1
    for split in ('train', 'val'):
        assert first[f'x_{split}'].equals(second[f'x_{split}'])
        assert first[f'y_{split}'].equals(second[f'y_{split}'])
    transformed = state.transform(recorded_shim_bars)
    perturbed = state.transform(changed)
    before = recorded_shim_bars['datetime'].dt.date() < cutoff
    assert transformed.filter(before).equals(perturbed.filter(before))
    with pytest.raises(ValueError):
        apply_column_shims(recorded_shim_bars.head(2), parse_shims(shims), {}, None,
                           training=True, target='next_return')


def test_column_shims_preserve_disabled_path_and_raw_feature_api(recorded_shim_bars):
    absent = _native().prepare_data(recorded_shim_bars, {})
    empty = _native([]).prepare_data(recorded_shim_bars, {})
    _assert_prepared_equal(absent, empty)
    assert '_column_shims' not in absent['_fitted_params']
    assert '_column_shims' not in empty['_fitted_params']
    raw = fractional_diff(recorded_shim_bars, d=0.5, cols=['close'], threshold=0.01)
    assert raw['close'].equals(recorded_shim_bars['close'])
    shims = [{'columns': ['close_fracdiff'], 'method': 'scale', 'reference': 'close'}]
    transformed, _ = apply_column_shims(raw, parse_shims(shims), {}, None, training=True, target='next_return')
    assert transformed['close'].equals(raw['close'])
    np.testing.assert_allclose(transformed['close_fracdiff'].to_numpy(),
        (raw['close_fracdiff'] / raw['close']).to_numpy(), equal_nan=True)


def test_column_shims_prepare_before_scaler_pca_and_preserve_alignment(recorded_shim_bars):
    shims = [{'columns': ['volume'], 'method': 'log1p'},
             {'columns': ['open', 'high', 'low'], 'method': 'ratio', 'reference': 'close'}]
    unscaled = _native(shims).prepare_data(recorded_shim_bars, {})
    manifest = _native(shims, scaler=True).set_pca_compression()
    prepared = manifest.prepare_data(recorded_shim_bars, {'auto_pca': True, 'pca_k': 2})
    fitted = prepared['_fitted_params']
    training_volume = np.log1p(_raw_split(recorded_shim_bars, 0)['volume'].to_numpy())
    assert fitted['_scaler'].medians['volume'] == pytest.approx(np.median(training_volume))
    assert fitted['_pca_input_feature_names'] == unscaled['x_train'].columns
    for split in SPLITS:
        assert prepared[f'x_{split}'].columns == ['pc_0', 'pc_1']
        assert prepared[f'y_{split}'].equals(unscaled[f'y_{split}'])
        assert prepared[f'x_{split}'].height == unscaled[f'x_{split}'].height
    assert prepared['_alignment'] == unscaled['_alignment']
    assert prepared['price_data_for_backtest'].equals(unscaled['price_data_for_backtest'])
    fractional = _native([{'columns': ['close'], 'method': 'fractional_diff', 'd': 1.0}]).prepare_data(recorded_shim_bars, {})
    assert fractional['x_train'].height == unscaled['x_train'].height - 1
    assert fractional['x_test'].height == unscaled['x_test'].height
    assert fractional['price_data_for_backtest']['datetime'].is_sorted()
    assert fractional['price_data_for_backtest']['datetime'].n_unique() == fractional['x_test'].height
    assert fractional['price_data_for_backtest']['datetime'].equals(unscaled['price_data_for_backtest']['datetime'])


def test_column_shim_sweep_records_distinct_parameters_and_outputs(recorded_shim_bars, tmp_path):
    config = _config([{'columns': ['close'], 'method': 'fractional_diff', 'd': '{order}'}], scaler=True)
    config['sfd']['params']['order'] = [0.0, 1.0]
    config['uel']['n_permutations'] = 2
    assert validate(config).valid
    directory = tmp_path / 'orders'
    loop = _loop(config, recorded_shim_bars, directory)
    _run(loop, count=2)
    records = _records(directory)
    assert loop.experiment_log['order'].to_list() == [0.0, 1.0]
    assert loop.experiment_log['_round_index'].to_list() == [0, 1]
    assert len(set(loop.experiment_log['id'])) == 2
    assert [entry['round_id'] for entry in records] == loop.experiment_log['id'].to_list()
    expected = []
    for record in records:
        params = record['round_params']
        manifest = CompiledSFD(config).manifest()
        data = manifest.prepare_data(recorded_shim_bars, params)
        assert data['_fitted_params']['_column_shims'].orders['close'] == params['order']
        result = manifest.run_model(data, params)
        np.testing.assert_allclose(record['preds'], result['_preds'], rtol=1e-8, atol=1e-10)
        expected.append(data['x_test']['close'].to_numpy())
    assert not np.allclose(*expected)
    saved = pl.read_csv(directory / 'results.csv')
    assert saved['id'].to_list() == loop.experiment_log['id'].to_list()
    assert saved['order'].to_list() == [0.0, 1.0]
    assert json.loads((directory / 'metadata.json').read_text())['yaml_reference'] == config


@pytest.mark.parametrize('order', (0.5, 'auto'))
def test_column_shims_trainer_sensor_replays_fitted_state(recorded_shim_bars, tmp_path, order):
    config = _config([{'columns': ['close'], 'method': 'fractional_diff', 'd': order, 'threshold': 0.01},
                      {'columns': ['volume'], 'method': 'log1p'}], scaler=True)
    directory = tmp_path / str(order)
    loop = _loop(config, recorded_shim_bars, directory)
    _run(loop)
    record = _records(directory)[0]
    trainer = Trainer(directory, data=recorded_shim_bars)
    sensor = trainer.train([record['round_id']])[0]
    prepared = CompiledSFD(config).manifest().prepare_data(recorded_shim_bars, record['round_params'])
    fitted = sensor._fitted_params['_column_shims']
    assert fitted.orders == prepared['_fitted_params']['_column_shims'].orders
    assert fitted.specs == prepared['_fitted_params']['_column_shims'].specs
    by_time = {prediction.datetime: prediction for prediction in sensor.predict_all(recorded_shim_bars)
               if prediction.reason is None}
    times = prepared['price_data_for_backtest']['datetime'].to_list()
    assert len(times) == len(record['preds'])
    for instant, prediction in zip(times, record['preds'], strict=True):
        assert by_time[instant].prediction == pytest.approx(prediction, rel=1e-6, abs=1e-9)
    replayed = sensor._manifest.sensor_input_prep(recorded_shim_bars, sensor._fitted_params, record['round_params'])[0]
    expected = prepared['x_test']
    aligned = replayed.join(prepared['price_data_for_backtest'].select('datetime'),
                            on='datetime', how='semi', maintain_order='left')
    assert aligned.drop('datetime').equals(expected)


def test_column_shims_walk_forward_refits_on_each_training_fold(recorded_shim_bars, tmp_path, monkeypatch):
    import limen.experiment._shim_fractional_diff as shims_module

    config = _config([{'columns': ['close'], 'method': 'fractional_diff', 'd': 'auto', 'threshold': 0.01}], scaler=True)
    declaration = config['sfd']['manifest']
    del declaration['split_dates']
    declaration['split_walk_forward'] = {'n_folds': 2, 'test_bars': 24,
        'purge_bars': 2, 'embargo_bars': 1, 'anchored': True}
    loop = _loop(config, recorded_shim_bars, tmp_path / 'folds')
    observed = []
    original = shims_module.find_min_d

    def fit_order(frame, col, **kwargs):
        observed.append(frame[col].drop_nulls().to_list())
        return original(frame, col=col, **kwargs)

    monkeypatch.setattr(shims_module, 'find_min_d', fit_order)
    _run(loop)
    assert len(observed) == 2
    for fold, values in enumerate(observed):
        fit = resolve_walk_forward_split(recorded_shim_bars, loop.manifest.split_walk_forward,
            fold, loop.manifest.split_config, require_validation=False)[0]
        assert values == fit['close'].to_list()
    assert len(observed[0]) != len(observed[1])
    assert loop.fold_results.height == 2
    assert len(_records(tmp_path / 'folds')[0]['folds']) == 2


@pytest.mark.parametrize('shims', (
    [{'columns': [], 'method': 'log1p'}],
    [{'columns': 'close', 'method': 'log1p'}],
    [{'columns': '{selected}', 'method': 'log1p'}],
    [{'columns': ['close', 3], 'method': 'log1p'}],
    [{'columns': ['close', 'close'], 'method': 'log1p'}],
    [{'columns': ['close'], 'method': 'unknown'}],
    [{'columns': ['close'], 'method': 'ratio'}],
    [{'columns': ['close'], 'method': 'scale', 'reference': 3}],
    [{'columns': ['close'], 'method': 'log1p', 'unknown': 1}],
    [{'columns': ['close'], 'method': 'fractional_diff', 'd': True}],
    [{'columns': ['close'], 'method': 'fractional_diff', 'd': -0.1}],
    [{'columns': ['close'], 'method': 'fractional_diff', 'd': 1.1}],
    [{'columns': ['close'], 'method': 'fractional_diff', 'd': 'missing'}],
    [{'columns': ['close'], 'method': 'fractional_diff', 'threshold': 0}],
    [{'columns': ['close'], 'method': 'fractional_diff', 'significance_level': 1}],
    [{'columns': ['close'], 'method': 'log1p'}, {'columns': ['close'], 'method': 'log1p'}],
))
def test_column_shims_reject_invalid_configuration(recorded_shim_bars, shims):
    with pytest.raises((TypeError, ValueError)):
        _native(shims).prepare_data(recorded_shim_bars, {})
    assert not validate(_config(shims)).valid


@pytest.mark.parametrize('case', ('missing', 'nonnumeric', 'reference', 'zero', 'log_domain',
                                   'datetime', 'target', 'missing_parameter'))
def test_column_shims_reject_invalid_data_domains(recorded_shim_bars, case):
    data = recorded_shim_bars
    spec = {'columns': ['volume'], 'method': 'log1p'}
    if case == 'missing':
        spec['columns'] = ['absent_recorded_column']
    elif case == 'nonnumeric':
        data = data.with_columns(pl.col('volume').cast(pl.String))
    elif case == 'reference':
        spec = {'columns': ['open'], 'method': 'ratio', 'reference': 'absent_reference'}
    elif case == 'zero':
        data = data.with_columns((pl.col('close') - pl.col('close')).alias('close'))
        spec = {'columns': ['open'], 'method': 'ratio', 'reference': 'close'}
    elif case == 'log_domain':
        data = data.with_columns((-pl.col('close')).alias('volume'))
    elif case == 'datetime':
        spec['columns'] = ['datetime']
    elif case == 'target':
        spec['columns'] = ['next_return']
    elif case == 'missing_parameter':
        spec = {'columns': ['close'], 'method': 'fractional_diff', 'd': '{absent_parameter}'}
    with pytest.raises((TypeError, ValueError)):
        _native([spec]).prepare_data(data, {})


def test_column_shims_reject_factorized_preparation(recorded_shim_bars, tmp_path):
    config = _config([{'columns': ['volume'], 'method': 'log1p'}], scaler=True)
    loop = _loop(config, recorded_shim_bars, tmp_path / 'factorized')
    with pytest.raises(ValueError, match='factorize'):
        _run(loop, factorize=True)

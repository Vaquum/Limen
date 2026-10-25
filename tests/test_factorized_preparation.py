from __future__ import annotations

from limen.experiment import _round_storage as _storage
import copy
import json
import weakref
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from limen import __version__
from limen.experiment import MLManifest, UniversalExperimentLoop
from limen.experiment._factorize import FactorizedRounds
from limen.experiment.errors import StrictModeError
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search.grid_strategy import GridStrategy
from limen.scalers import RobustScaler
from limen.sfd.reference_architecture.lightgbm_binary import LightGBMBinary
from limen.targets import QuantileBinaryTarget
from limen.yaml import CompiledSFD, build_search_strategy
from tests.test_factorize_equivalence import _numeric_equal

FIXTURE = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'


def _config():
    return {
        'schema_version': '1.0',
        'metadata': {'name': 'fitted_preparation', 'description': 'Recorded preparation reuse proof',
                     'limen_version': __version__, 'mode': 'development'},
        'sfd': {
            'manifest': {
                'type': 'ml',
                'data_source': {'method': 'limen.data.HistoricalData.get_any_file',
                                'params': {'file_path_or_url': str(FIXTURE)}},
                'split_dates': {
                    'train_start': '2024-01-01', 'train_end': '2024-01-04',
                    'val_start': '2024-01-04', 'val_end': '2024-01-05',
                    'test_start': '2024-01-05', 'test_end': '2024-01-07',
                    'test_predict_guard': False,
                },
                'indicators': [{'func': 'limen.indicators.window_return', 'params': {'period': 1}}],
                'features': [{'func': 'limen.features.lag_range',
                              'params': {'col': 'ret_1', 'start': 0, 'end': '{lookback}'}}],
                'target': {'name': 'up_next', 'class': 'limen.targets.QuantileBinaryTarget',
                           'fit_params': {'source_column': 'ret_1', 'quantile': '{quantile}'},
                           'transform_params': {'shift': -1}},
                'scaler': {'class': 'limen.scalers.RobustScaler'},
                'reference_architecture': 'limen.sfd.reference_architecture.lightgbm_binary',
                'backtest': {'fee_bps': '{fee}', 'slip_bps': 3., 'notional_rate': 0.5},
            },
            'params': {
                'lookback': [2], 'quantile': [0.5], 'fee': [4., 9.],
                'n_estimators': [4, 8], 'n_jobs': [1], 'random_state': [42],
                'deterministic': [True], 'force_row_wise': [True], 'early_stopping_rounds': [0],
                'subsample_freq': [0], 'subsample': [1.], 'colsample_bytree': [1.],
            },
        },
        'uel': {'n_permutations': 4, 'prep_each_round': True,
                'search_strategy': {'type': 'grid'}, 'record_execution': True,
                'record_model_outputs': True},
    }


def _bars():
    return pl.read_parquet(FIXTURE, n_rows=192).with_columns(pl.lit(3600).alias('base_interval'))


def _loop(config, path, *, bars=None, callback=None):
    return UniversalExperimentLoop(
        sfd=CompiledSFD(config), data=_bars() if bars is None else bars, experiment_dir=path,
        yaml_reference=config, search_strategy=build_search_strategy(config),
        checkpoint_interval=1, feedback_interval=1, intra_callback=callback,
    )


def _run(loop, *, enabled=None, count=4, resume=False, retained=False):
    kwargs = {} if enabled is None else {'factorize': enabled}
    loop.run('fitted_preparation', n_permutations=count, prep_each_round=True,
             progress_bar=False, resume=resume, post_processing=retained,
             record_execution=True, record_model_outputs=True,
             recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True), **kwargs)


def _rounds(config, bars=None):
    loop = _loop(config, None, bars=bars)
    domain = config['sfd']['params']
    return FactorizedRounds(
        manifest=loop.manifest, strategy=GridStrategy(ParamDomain(domain)), domain=domain,
        pruning=False, callback=False, context=None, prep=loop.manifest.prepare_data,
        model=loop.manifest.run_model, data=loop.data,
        record_execution=True, record_model_outputs=True,
    )


def _params(config):
    return {name: values[0] for name, values in config['sfd']['params'].items()}


def _instrument(monkeypatch):
    counts = {'target': 0, 'scaler': 0, 'model': 0, 'prepare': 0}
    for cls, name, label in ((QuantileBinaryTarget, '__init__', 'target'),
                             (RobustScaler, '__init__', 'scaler'),
                             (LightGBMBinary, 'train', 'model'),
                             (MLManifest, '_prepare_data_with_fitted', 'prepare')):
        original = getattr(cls, name)

        def counted(*args, _fn=original, _label=label, **kwargs):
            counts[_label] += 1
            return _fn(*args, **kwargs)

        monkeypatch.setattr(cls, name, counted)
    return counts


def _fitted(data):
    return {name: copy.deepcopy(vars(value)) for name, value in data['_fitted_params'].items()}


def _science(path):
    rows = pl.read_csv(path / 'results.csv', infer_schema_length=None).drop('execution_time').to_dicts()
    recorded = [json.loads(line) for line in _storage.text(_storage.round_path(path)).splitlines()]
    return [rows, recorded]


def test_factorized_preparation_equivalence(tmp_path, monkeypatch):
    captured = []
    for enabled in (False, True):
        config = _config()
        path = tmp_path / str(enabled)
        loop = _loop(config, path)
        prepared = []
        original = MLManifest._prepare_data_with_fitted

        def capture(*args, _original=original, _prepared=prepared, **kwargs):
            data = _original(*args, **kwargs)
            _prepared.append((args[2]['n_estimators'],
                             {key: data[key].clone() for key in
                              ('x_train', 'x_val', 'x_test', 'y_train', 'y_val', 'y_test')},
                             _fitted(data)))
            return data

        with monkeypatch.context() as hooks:
            counts = _instrument(hooks)
            hooks.setattr(MLManifest, '_prepare_data_with_fitted', capture)
            _run(loop, enabled=enabled, retained=True)
        # Retained row analysis independently reconstructs all four permutations.
        assert counts['target'] == counts['scaler'] == (1 if enabled else 4) + 4
        assert counts['model'] == (2 if enabled else 4)
        assert len(prepared) == (2 if enabled else 4) + 4
        assert loop.experiment_log['_round_index'].to_list() == list(range(4))
        assert len(loop.scalers) == len(loop.preds) == len(loop._alignment) == 4
        assert len({id(scaler) for scaler in loop.scalers}) == 4
        captured.append((loop, prepared, _science(path)))
        rows, records = _science(path)
        assert len(rows) == len(records) == 4
        assert [record['_round_index'] for record in records] == list(range(4))
        assert [record['round_id'] for record in records] == loop.experiment_log['id'].to_list()
        assert not any('preparation' in name.name for name in path.iterdir())
        metadata = json.loads((path / 'metadata.json').read_text())
        assert metadata.get('factorize', False) is enabled
    off, on = captured
    _numeric_equal(*[entry[0].experiment_log.drop('execution_time').to_dicts() for entry in captured])
    _numeric_equal(off[2], on[2])
    for model, values, artifacts in on[1]:
        reference = next(item for item in off[1] if item[0] == model)
        for key, frame in values.items():
            assert frame.equals(reference[1][key]), key
        _numeric_equal(artifacts, reference[2])
    for ordinal in range(4):
        np.testing.assert_array_equal(off[0].preds[ordinal], on[0].preds[ordinal])
        _numeric_equal(off[0]._alignment[ordinal], on[0]._alignment[ordinal])
        _numeric_equal(vars(off[0].scalers[ordinal]), vars(on[0].scalers[ordinal]))


@pytest.mark.parametrize('changed', ['feature', 'target', 'scaler', 'model_named_feature'])
def test_factorized_preparation_keys_and_fit_counts(monkeypatch, changed):
    config = _config()
    domain = config['sfd']['params']
    manifest = config['sfd']['manifest']
    if changed == 'feature':
        domain['lookback'] = [2, 4]
        key, value = 'lookback', 4
    elif changed == 'target':
        domain['quantile'] = [0.5, 0.25]
        key, value = 'quantile', 0.25
    elif changed == 'scaler':
        manifest['scaler']['params'] = {'quantile_range': '{scale_range}'}
        domain['scale_range'] = [(0.25, 0.75), (0.1, 0.9)]
        key, value = 'scale_range', (0.1, 0.9)
    else:
        manifest['features'][0]['params']['end'] = '{n_estimators}'
        key, value = 'n_estimators', 8
    rounds = _rounds(config)
    params = _params(config)
    counts = _instrument(monkeypatch)
    data, result, _, _ = rounds.evaluate(params)
    original = _fitted(data)
    del data, result
    data, result, _, _ = rounds.evaluate({**params, 'fee': 9.})
    assert counts == {'target': 1, 'scaler': 1, 'model': 1, 'prepare': 1}
    del data, result
    data, result, _, _ = rounds.evaluate({**params, key: value})
    assert counts == {'target': 2, 'scaler': 2, 'model': 2, 'prepare': 2}
    assert _fitted(data) != original
    del data, result
    if changed != 'model_named_feature':
        data, result, _, _ = rounds.evaluate({**params, key: value, 'n_estimators': 8})
        assert counts == {'target': 2, 'scaler': 2, 'model': 3, 'prepare': 3}
        del data, result


def test_factorized_preparation_cache_lifetime(tmp_path, monkeypatch):
    counts = _instrument(monkeypatch)
    config = _config()
    first = _rounds(config)
    params = _params(config)
    fitted = []
    for rounds in (first, _rounds(config), _rounds(config, _bars().slice(1))):
        data, result, _, _ = rounds.evaluate(params)
        fitted.append(_fitted(data))
        del data, result
    altered = copy.deepcopy(config)
    altered['sfd']['manifest']['split_dates']['train_end'] = '2024-01-03'
    data, result, _, _ = _rounds(altered).evaluate(_params(altered))
    fitted.append(_fitted(data))
    del data, result
    assert counts['target'] == counts['scaler'] == counts['model'] == 4
    assert fitted[0] == fitted[1]
    assert fitted[0] != fitted[2] and fitted[0] != fitted[3]

    source = _bars()
    frozen = _rounds(config, source)
    data, result, _, _ = frozen.evaluate(params)
    expected = {key: data[key].clone() for key in ('x_train', 'y_train', 'x_test', 'y_test')}
    del data, result
    source.drop_in_place('close')
    before = dict(counts)
    data, result, _, _ = frozen.evaluate({**params, 'n_estimators': 8})
    assert counts['target'] == before['target'] and counts['scaler'] == before['scaler']
    assert counts['model'] == before['model'] + 1
    assert 'close' in frozen.data.columns and 'close' not in source.columns
    for key, frame in expected.items():
        assert data[key].equals(frame)
    del data, result

    full = _loop(config, tmp_path / 'full')
    _run(full, enabled=True)
    split = _loop(config, tmp_path / 'split')
    checkpoint = split._checkpoint

    def stop(*args, **kwargs):
        checkpoint(*args, **kwargs)
        if args[4] == 1:
            split._shutdown_requested = True

    monkeypatch.setattr(split, '_checkpoint', stop)
    _run(split, enabled=True)
    before = dict(counts)
    resumed = _loop(config, tmp_path / 'split')
    _run(resumed, enabled=True, resume=True)
    assert counts['target'] == before['target'] + 1
    assert counts['scaler'] == before['scaler'] + 1
    _numeric_equal(_science(tmp_path / 'full'), _science(tmp_path / 'split'))

    for enabled in (None, False):
        def inject(log, msq):
            if log.height == 2:
                msq.inject({key: params[key] for key in sorted(params)})

        repeated = _loop(config, tmp_path / f'fresh_{enabled}', callback=inject)
        before = dict(counts)
        _run(repeated, enabled=enabled, count=5)
        assert counts['target'] == before['target'] + 5
        assert counts['scaler'] == before['scaler'] + 5
        assert counts['model'] == before['model'] + 5
        assert len(set(repeated.experiment_log['_id'].to_list())) == 4
        assert repeated.experiment_log['_round_index'].to_list() == list(range(5))


def test_factorized_preparation_failure_and_isolation(monkeypatch):
    config = _config()
    rounds = _rounds(config)
    params = _params(config)
    counts = _instrument(monkeypatch)
    original = MLManifest.run_model

    def fail(*args, **kwargs):
        raise StrictModeError('Recorded preparation proof rejects the first model attempt')

    monkeypatch.setattr(MLManifest, 'run_model', fail)
    rounds.model = rounds.manifest.run_model
    with pytest.raises(StrictModeError, match='first model attempt'):
        rounds.evaluate(params)
    assert not rounds.cache and not rounds.preparation.cache
    monkeypatch.setattr(MLManifest, 'run_model', original)
    rounds.model = rounds.manifest.run_model
    data, result, _, _ = rounds.evaluate(params)
    assert counts['target'] == counts['scaler'] == 2
    expected = _fitted(data)
    model_ref = weakref.ref(result['_model'])
    frame_ref = weakref.ref(data['x_train'])
    data['_scaler'].medians.clear()
    data['_target_cls_up_next'].cutoff = -1.
    del data, result
    assert model_ref() is None and frame_ref() is None
    data, result, _, _ = rounds.evaluate({**params, 'n_estimators': 8})
    assert counts['target'] == counts['scaler'] == 2
    assert _fitted(data) == expected
    data['_scaler'].medians.clear()
    data['_target_cls_up_next'].cutoff = -1.
    del data, result
    data, result, _, _ = rounds.evaluate({**params, 'fee': 9.})
    assert _fitted(data) == expected
    assert 'x_train' not in data and '_model' not in result
    assert all(set(entry) == {'_target_cls_up_next', '_scaler'}
               for entry in rounds.preparation.cache.values())


@pytest.mark.parametrize('changed', ['economic_target', 'economic_scaler'])
def test_factorized_preparation_admission(tmp_path, changed):
    config = _config()
    manifest = config['sfd']['manifest']
    if changed == 'economic_target':
        manifest['target']['fit_params']['quantile'] = '{fee}'
    elif changed == 'economic_scaler':
        manifest['scaler']['params'] = {'quantile_range': '{fee}'}
    elif changed == 'registry_scaler':
        manifest['scaler'] = {'from_params': 'scaler_type'}
        config['sfd']['params']['scaler_type'] = ['robust']
    else:
        manifest['scaler'] = {'class': 'limen.scalers.RankGaussScaler'}
    loop = _loop(config, tmp_path)
    with pytest.raises(ValueError, match='factorize requires'):
        _run(loop, enabled=True)
    for name in ('metadata.json', 'checkpoint.json', 'results.csv', 'round_data.jsonl'):
        assert not (tmp_path / name).exists()

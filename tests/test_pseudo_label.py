from datetime import datetime, timezone
import hashlib
from importlib import import_module
import inspect
import json
from pathlib import Path
import shutil

import numpy as np
import polars as pl
import pytest
from ruamel.yaml import YAML

from limen import Trainer, UniversalExperimentLoop
from limen.experiment import _round_storage
from limen.scalers import RobustScaler
from limen.sfd.foundational_sfd import pseudo_label_binary as foundational
from limen.sfd.reference_architecture import PseudoLabelBinary, pseudo_label_binary
from limen.sfd.reference_architecture.pseudo_label_binary import recorded_rows
from limen.yaml import CompiledSFD, build_search_strategy, validate

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet'


@pytest.fixture(scope='module')
def recorded():
    bars = pl.read_parquet(SOURCE, n_rows=1024)
    params = {key: values[0] for key, values in foundational.params().items()}
    params.update(pool_path=str(SOURCE), upstream_params={'C': 1000., 'max_iter': 500},
                  downstream_params={'C': 1.}, confidence_threshold=.8)
    return bars, params, foundational.manifest().prepare_data(bars, params)


def _controls(params):
    return {key: value for key, value in params.items() if key != 'lookback_end'}


def test_recorded_pool_selection_native_fit_populations_and_observed_metrics(recorded, monkeypatch):
    from sklearn.linear_model import LogisticRegression
    bars, params, data = recorded
    columns = ['ret_1_lag_' + str(index) for index in range(24)]
    x, y = data['x_train'].select(columns).to_numpy(), np.asarray(data['y_train'])
    original, calls = LogisticRegression.fit, []

    def observe(model, features, labels, **kwargs):
        assert not kwargs
        calls.append((np.asarray(features).copy(), np.asarray(labels).copy()))
        return original(model, features, labels, **kwargs)

    monkeypatch.setattr(LogisticRegression, 'fit', observe)
    model = PseudoLabelBinary().train(data | {'y_val': object(), 'y_test': object()}, **_controls(params))
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], x)
    np.testing.assert_array_equal(calls[0][1], y)
    closes, times = bars['close'].to_numpy(), bars['datetime'].to_list()
    returns = closes[1:] / closes[:-1] - 1
    pool_indices = [index for index, time in enumerate(times) if
                    datetime(2024, 1, 10, tzinfo=timezone.utc) <= time < datetime(2024, 1, 15, tzinfo=timezone.utc)]
    pool = np.asarray([[returns[index - lag - 1] for lag in range(24)] for index in pool_indices])
    upstream = original(LogisticRegression(C=1000., max_iter=500, random_state=42), x, y)
    probability = upstream.predict_proba(pool)[:, 1]
    confidence = np.maximum(probability, 1 - probability)
    selected = confidence >= .8
    assert selected.sum() == 6
    augmented, labels = np.vstack((x, pool[selected])), np.concatenate((y, (probability[selected] > .5).astype(np.int8)))
    np.testing.assert_allclose(calls[1][0], augmented, atol=1e-12)
    np.testing.assert_array_equal(calls[1][1], labels)
    downstream = original(LogisticRegression(C=1., max_iter=200, random_state=42), augmented, labels)
    expected = downstream.predict_proba(data['x_test'].select(columns).to_numpy())[:, 1]
    np.testing.assert_allclose(model.predict(data)['_probs'], expected, atol=1e-12)
    assert model.evaluate(data)['accuracy'] == np.mean((expected > .5) == np.asarray(data['y_test']))
    provenance = model._provenance
    assert provenance['linkage_observed_rows'] == 191 and provenance['linkage_pool_rows'] == 120
    assert provenance['linkage_pseudo_rows'] == 6 and provenance['linkage_downstream_rows'] == 197
    payload = provenance['linkage_selected_rows']
    assert hashlib.sha256(payload.encode()).hexdigest() == provenance['linkage_selected_sha256']
    records = json.loads(payload)
    expected_times = [times[index] for index, keep in zip(pool_indices, selected, strict=True) if keep]
    assert [row['datetime'] for row in records] == [time.isoformat() for time in expected_times]
    np.testing.assert_allclose([row['probability'] for row in records], probability[selected], atol=1e-12)
    assert all(row['origin'] == 'pseudo' and row['label'] == int(row['probability'] > .5) and
               row['confidence'] == max(row['probability'], 1 - row['probability']) and
               datetime.fromisoformat(row['available_at']) <= datetime.fromisoformat(params['training_cutoff']) for row in records)
    assert provenance['linkage_upstream_fit_cutoff_ns'] <= int(expected_times[0].timestamp() + 3600) * 1_000_000_000
    np.testing.assert_allclose(model.predict({'x_test': data['x_test'].to_numpy()})['_probs'], expected, atol=1e-12)
    np.testing.assert_allclose(model.predict({'x_test': data['x_test'].head(5)})['_probs'], expected[:5], atol=1e-12)
    for index in range(5):
        np.testing.assert_allclose(model.predict({'x_test': data['x_test'].slice(index, 1)})['_probs'], expected[index:index + 1], atol=1e-12)


def test_pseudo_native_source_recording_reconstruction_and_changed_pool(tmp_path, recorded, monkeypatch):
    bars, _, _ = recorded
    pool_path = tmp_path / 'recorded_pool.parquet'
    shutil.copyfile(SOURCE, pool_path)
    config = YAML(typ='safe').load(ROOT / 'limen/yaml/templates/pseudo_label_binary.yaml')
    config['sfd']['params']['pool_path'] = [str(pool_path)]
    assert validate(config).valid
    monkeypatch.setattr(import_module('limen.data.historical_data'), '_read_any_file', lambda *_args, **_kwargs: bars)
    compiled = CompiledSFD(config)
    source = compiled.manifest().fetch_data()
    output = tmp_path / 'experiment'
    loop = UniversalExperimentLoop(sfd=compiled, data=source, experiment_dir=output,
                                   search_strategy=build_search_strategy(config), yaml_reference=config)
    loop.run('recorded_pseudo', n_permutations=1, prep_each_round=True, progress_bar=False,
             record_model_outputs=True, recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True))
    log = pl.read_csv(output / 'results.csv')
    assert log.height == 1 and log['strict_mode_error'].drop_nulls().len() == 0
    record = json.loads(_round_storage.text(_round_storage.round_path(output)).splitlines()[0])
    np.testing.assert_array_equal(record['preds'], np.asarray(record['probs']) > .5)
    sensor = Trainer(output, data=bars).train([record['round_id']])[0]
    assert isinstance(sensor._model, PseudoLabelBinary)
    assert {key: log[key][0] for key in log.columns if key.startswith('linkage_')} == sensor._model._provenance
    rows = [row for row in sensor.predict_all(bars) if row.reason is None]
    assert rows and all(row.probability is not None and row.prediction == int(row.probability > .5) for row in rows)
    bars.head(512).write_parquet(pool_path)
    with pytest.raises(ValueError, match='SHA-256'):
        Trainer(output, data=bars).train([record['round_id']])


def test_pseudo_pool_cutoff_identity_selection_and_preprocessing_guards(recorded, tmp_path):
    bars, params, data = recorded
    for changed, message in [({'pool_sha256': '0' * 64}, 'SHA-256'),
                             ({'pool_start': '2024-01-09T00:00:00+00:00'}, 'fitting boundary'),
                             ({'training_cutoff': '2024-01-19T00:00:00+00:00'}, 'precede observed evaluation'),
                             ({'pool_end': '2024-01-16T00:00:00+00:00'}, 'training cutoff'),
                             ({'pool_start': '2024-01-10'}, 'timezone-aware'),
                             ({'observation_interval_seconds': 1800}, 'declared interval'),
                             ({'confidence_threshold': .5}, 'above 0.5'),
                             ({'confidence_threshold': 1.}, 'selected no pool'),
                             ({'upstream_params': {'prediction_calibration_config': object()}}, 'calibration')]:
        with pytest.raises(ValueError, match=message):
            PseudoLabelBinary().train(data, **(_controls(params) | changed))
    for name, pool in [('duplicate', pl.concat([bars, bars.slice(300, 1)])), ('short_history', bars.slice(216, 120))]:
        path = tmp_path / (name + '.parquet')
        pool.write_parquet(path)
        with pytest.raises(ValueError, match=r'chronological|complete recorded history'):
            PseudoLabelBinary().train(data, **(_controls(params) | {'pool_path': str(path), 'pool_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
    custom = foundational.manifest().set_scaler(RobustScaler, param_name='custom_scaler').prepare_data(bars, params)
    with pytest.raises(ValueError, match='unscaled'):
        PseudoLabelBinary().train(custom, **_controls(params))
    with pytest.raises(ValueError, match='protected holdout'):
        recorded_rows(data | {'_holdout_carriers': list(data['_linkage_rows'].values())}, list(data['_linkage_rows'].values()), {}, {})
    with pytest.raises(ValueError, match='not fitted'):
        PseudoLabelBinary().predict(data)
    with pytest.raises(FileNotFoundError):
        PseudoLabelBinary().train(data, **(_controls(params) | {'pool_path': str(tmp_path / 'missing.parquet')}))


def test_pseudo_native_and_yaml_controls(recorded):
    _, _, _ = recorded
    config = YAML(typ='safe').load(ROOT / 'limen/yaml/templates/pseudo_label_binary.yaml')
    controls = set(inspect.signature(pseudo_label_binary).parameters) - {'data'}
    assert validate(config).valid and controls <= config['sfd']['params'].keys() and controls <= foundational.params().keys()
    assert build_search_strategy(config).domain.total_combinations == 8

from datetime import timedelta
from importlib import import_module
import inspect
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from ruamel.yaml import YAML

from limen import Trainer, UniversalExperimentLoop
from limen.experiment import _round_storage
from limen.scalers import RobustScaler
from limen.sfd.foundational_sfd import feature_link_binary as foundational
from limen.sfd.reference_architecture import FeatureLinkedBinary, feature_link_binary
from limen.yaml import CompiledSFD, build_search_strategy, validate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def recorded():
    bars = pl.read_parquet(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet', n_rows=1024)
    params = {key:values[0] for key,values in foundational.params().items()}
    return bars, params, foundational.manifest().prepare_data(bars, params)


@pytest.mark.parametrize('splits,upstream_c,downstream_c', [(2,.5,1.), (3,1.,.5)])
def test_native_downstream_fits_out_of_sample_feature(recorded, monkeypatch, splits, upstream_c, downstream_c):
    from sklearn.linear_model import LogisticRegression
    _, params, data = recorded
    columns = ['ret_1_lag_' + str(index) for index in range(params['lookback_end'] + 1)]
    x, y = data['x_train'].select(columns).to_numpy(), np.asarray(data['y_train'])
    times = data['_linkage_rows']['train']['datetime'].to_list()
    original, calls = LogisticRegression.fit, []

    def observe(model, features, observed, **kwargs):
        assert not kwargs
        calls.append((np.asarray(features).copy(), np.asarray(observed).copy()))
        return original(model, features, observed, **kwargs)

    monkeypatch.setattr(LogisticRegression, 'fit', observe)
    model = FeatureLinkedBinary().train(data | {'y_val':object(), 'y_test':object()}, n_splits=splits,
        upstream_params={'C':upstream_c}, downstream_params={'C':downstream_c})
    assert len(calls) == splits + 2
    size, oof, indices = len(y)//(splits+1), [], []
    for index in range(splits):
        start = len(y) - (splits-index)*size
        cutoff = times[start] + timedelta(hours=1)
        prefix = [row for row in range(start) if times[row]+timedelta(hours=2) <= cutoff]
        covered = np.arange(start, start+size)
        np.testing.assert_array_equal(calls[index][0], x[prefix])
        np.testing.assert_array_equal(calls[index][1], y[prefix])
        upstream = original(LogisticRegression(C=upstream_c,max_iter=200,random_state=42), x[prefix], y[prefix])
        oof.extend(upstream.predict_proba(x[covered])[:,1])
        indices.extend(covered)
    covered = np.asarray(indices)
    augmented = np.column_stack((x[covered], oof))
    np.testing.assert_array_equal(calls[-2][0], x)
    np.testing.assert_array_equal(calls[-2][1], y)
    np.testing.assert_allclose(calls[-1][0], augmented, atol=1e-12)
    np.testing.assert_array_equal(calls[-1][1], y[covered])
    final_upstream = original(LogisticRegression(C=upstream_c,max_iter=200,random_state=42), x, y)
    assert not np.allclose(oof, final_upstream.predict_proba(x[covered])[:,1])
    downstream = original(LogisticRegression(C=downstream_c,max_iter=200,random_state=42), augmented, y[covered])
    test = data['x_test'].select(columns).to_numpy()
    inference = np.column_stack((test, final_upstream.predict_proba(test)[:,1]))
    probability = downstream.predict_proba(inference)[:,1]
    prediction = model.predict({'x_test':data['x_test']})
    np.testing.assert_allclose(prediction['_probs'], probability, atol=1e-12)
    np.testing.assert_array_equal(prediction['_preds'], probability > .5)
    assert model.evaluate(data)['accuracy'] == np.mean((probability>.5) == np.asarray(data['y_test']))
    assert json.loads(model._provenance['linkage_columns']) == [*columns, 'upstream_probability']
    assert model._provenance['linkage_downstream_rows'] == len(covered) < len(y)
    assert all(fold['fit_label_available_at_ns'] <= fold['prediction_available_at_ns'] for fold in json.loads(model._provenance['linkage_fold_cutoffs']))
    np.testing.assert_allclose(model.predict({'x_test':data['x_test'].to_numpy()})['_probs'], probability, atol=1e-12)
    np.testing.assert_allclose(model.predict({'x_test':data['x_test'].head(5)})['_probs'], probability[:5], atol=1e-12)
    for index in range(5):
        np.testing.assert_allclose(model.predict({'x_test':data['x_test'].slice(index,1)})['_probs'], probability[index:index+1], atol=1e-12)


def test_feature_link_native_source_logs_and_reconstruction(tmp_path, recorded, monkeypatch):
    bars, _, _ = recorded
    config = YAML(typ='safe').load(ROOT / 'limen/yaml/templates/feature_link_binary.yaml')
    assert validate(config).valid
    monkeypatch.setattr(import_module('limen.data.historical_data'), '_read_any_file', lambda *_args, **_kwargs:bars)
    compiled = CompiledSFD(config)
    source = compiled.manifest().fetch_data()
    loop = UniversalExperimentLoop(sfd=compiled, data=source, experiment_dir=tmp_path,
                                   search_strategy=build_search_strategy(config), yaml_reference=config)
    loop.run('recorded_features', n_permutations=1, prep_each_round=True, progress_bar=False,
             record_model_outputs=True, recording=dict.fromkeys(('predictions','alignment','trade_ledger','fold_details'),True))
    log = pl.read_csv(tmp_path / 'results.csv')
    assert log.height == 1 and log['strict_mode_error'].drop_nulls().len() == 0
    record = json.loads(_round_storage.text(_round_storage.round_path(tmp_path)).splitlines()[0])
    np.testing.assert_array_equal(record['preds'], np.asarray(record['probs']) > .5)
    sensor = Trainer(tmp_path,data=bars).train([record['round_id']])[0]
    model = sensor._model
    assert isinstance(model,FeatureLinkedBinary)
    assert {key:log[key][0] for key in log.columns if key.startswith('linkage_')} == model._provenance
    rows = [row for row in sensor.predict_all(bars) if row.reason is None]
    assert rows and all(row.probability is not None and row.prediction == int(row.probability>.5) for row in rows)


def test_feature_controls_and_fit_boundary_guards(recorded):
    bars, params, data = recorded
    config = YAML(typ='safe').load(ROOT / 'limen/yaml/templates/feature_link_binary.yaml')
    controls = set(inspect.signature(feature_link_binary).parameters) - {'data'}
    assert validate(config).valid and controls <= config['sfd']['params'].keys() and controls <= foundational.params().keys()
    assert build_search_strategy(config).domain.total_combinations == 8
    with pytest.raises(ValueError,match='not fitted'):
        FeatureLinkedBinary().predict({'x_test':data['x_test']})
    with pytest.raises(ValueError,match='row times'):
        FeatureLinkedBinary().train({key:value for key,value in data.items() if key != '_linkage_rows'})
    with pytest.raises(ValueError,match='observation interval'):
        FeatureLinkedBinary().train(data | {'_linkage_rows':data['_linkage_rows'] | {'train':data['_linkage_rows']['train'].reverse()}})
    custom = foundational.manifest().set_scaler(RobustScaler, param_name='custom_scaler').prepare_data(bars, params)
    with pytest.raises(ValueError,match='unscaled'):
        FeatureLinkedBinary().train(custom)
    scaled = foundational.manifest().set_scaler(RobustScaler).prepare_data(bars,params)
    with pytest.raises(ValueError,match='unscaled'):
        FeatureLinkedBinary().train(scaled)
    with pytest.raises(ValueError,match='calibration'):
        FeatureLinkedBinary().train(data,upstream_params={'prediction_calibration_config':object()})

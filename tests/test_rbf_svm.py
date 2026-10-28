import inspect
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from ruamel.yaml import YAML
from sklearn.svm import SVC

from limen import Trainer, UniversalExperimentLoop
from limen.sfd.foundational_sfd import rbf_svm_binary as foundational
from limen.sfd.reference_architecture import RBFSVMBinary, rbf_svm_binary
from limen.yaml import validate
from limen.yaml.compiler import CompiledSFD, build_search_strategy

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def recorded():
    bars = pl.read_parquet(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet', n_rows=768)
    parameters = {name: values[0] for name, values in foundational.params().items()}
    parameters.update(C=0.7, gamma=0.25, shrinking=False, tol=0.01, cache_size=32,
                      class_weight='balanced', random_state=7, max_iter=10000)
    manifest = foundational.manifest()
    return bars, manifest, parameters, manifest.prepare_data(bars, parameters)


@pytest.mark.parametrize('calibrated', [False, True])
def test_recorded_svc_fit_metrics_and_calibration(recorded, calibrated):
    _, manifest, parameters, data = recorded
    selected = parameters | {'use_calibration': calibrated, 'use_threshold': calibrated}
    result = rbf_svm_binary(data, **manifest.resolve_model_kwargs(selected))
    model = result['_model']
    assert isinstance(model, RBFSVMBinary) and isinstance(model.model, SVC)
    assert model.model.kernel == 'rbf' and model.model.probability is True
    controls = set(inspect.signature(rbf_svm_binary).parameters) - {'data', 'prediction_calibration_config'}
    assert controls <= foundational.params().keys()
    for name in controls:
        assert model.model.get_params()[name] == parameters[name]
    prediction = model.predict({'x_test': data['x_test']})
    np.testing.assert_array_equal(prediction['_preds'], result['_preds'])
    probabilities = prediction['_probs']
    truth = np.asarray(data['y_test'])
    assert set(truth) == {0, 1}
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    assert result['accuracy'] == float(np.mean(truth == prediction['_preds']))
    differences = probabilities[truth == 1, None] - probabilities[truth == 0]
    np.testing.assert_allclose(result['auc'], np.mean((differences > 0) + 0.5 * (differences == 0)), atol=1e-12)
    if calibrated:
        np.testing.assert_array_equal(prediction['_preds'], probabilities >= result['optimal_threshold'])
        assert result['optimal_threshold'] is not None and result['val_score'] is not None
    assert not any(key.startswith(('confusion_', 'backtest_')) for key in model.evaluate(data, inline_metrics=False))


def test_svc_refit_resets_calibration_and_requires_binary_labels(recorded):
    _, manifest, parameters, data = recorded
    model = rbf_svm_binary(data, **manifest.resolve_model_kwargs(parameters))['_model']
    model.train(data, C=0.5, gamma='scale', prediction_calibration_config=None)
    assert set(model.predict({'x_test': data['x_test']})) == {'_preds', '_probs'}
    truth = np.asarray(data['y_train'])
    indices = np.flatnonzero(truth == 0)
    with pytest.raises(ValueError, match='both numeric binary'):
        model.train({'x_train': np.asarray(data['x_train'])[indices], 'y_train': truth[indices]})
    with pytest.raises(TypeError, match='kernel'):
        model.train(data, kernel='linear')


def test_compiled_svc_and_trainer_use_recorded_parameters(tmp_path):
    source = ROOT / 'tests/fixtures/spot_15m_20250101_20250531.parquet'
    config = YAML(typ='safe').load(ROOT / 'docs/examples/compiled-research/family_rbf_svm_binary.yaml')
    config['sfd']['manifest']['data_source']['params']['file_path_or_url'] = str(source)
    directory = tmp_path / 'compiled-svc'
    loop = UniversalExperimentLoop(sfd=CompiledSFD(config), search_strategy=build_search_strategy(config),
                                   experiment_dir=directory, yaml_reference=config)
    loop.run('recorded-svc', n_permutations=1, prep_each_round=True, post_processing=False, progress_bar=False)
    assert loop.experiment_log.height == len(loop.models) == 1
    assert loop.experiment_log['strict_mode_error'].null_count() == 1
    model = loop.models[0]
    assert isinstance(model, RBFSVMBinary) and isinstance(model.model, SVC)
    prepared = loop.manifest.prepare_data(loop.data, loop.round_params[0])
    result = rbf_svm_binary(prepared, **loop.manifest.resolve_model_kwargs(loop.round_params[0]))
    np.testing.assert_array_equal(result['_preds'], model.predict(prepared)['_preds'])
    assert result['accuracy'] == loop.experiment_log['accuracy'][0]
    sensor = Trainer(directory).train(loop.experiment_log['id'].to_list())[0]
    rows = sensor.predict_all(pl.read_parquet(source))
    assert any(row.reason is None and row.prediction in (0, 1) for row in rows)
    assert all(row.probability is None or 0 <= row.probability <= 1 for row in rows)
    metadata = json.loads((directory / 'metadata.json').read_text())
    assert metadata['yaml_reference']['sfd']['params'] == config['sfd']['params']


def test_svc_template_covers_its_architecture():
    config = YAML(typ='safe').load(ROOT / 'limen/yaml/templates/rbf_svm_binary.yaml')
    outcome = validate(config)
    assert outcome.valid, outcome.errors
    controls = set(inspect.signature(rbf_svm_binary).parameters) - {'data', 'prediction_calibration_config'}
    assert controls <= config['sfd']['params'].keys()
    assert build_search_strategy(config).domain.total_combinations <= 10000

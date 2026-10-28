import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from limen import UniversalExperimentLoop
from limen.sfd.foundational_sfd import tabpfn_binary
from limen.sfd.reference_architecture import TabPFNBinary


pytestmark = pytest.mark.tabpfn_model


def test_tabpfn(tmp_path):
    if os.getenv('LIMEN_RUN_TABPFN_MODEL_TEST') != '1':
        pytest.skip('Set LIMEN_RUN_TABPFN_MODEL_TEST=1 for the real TabPFN CPU test')

    from tabpfn import TabPFNClassifier
    from inspect import signature

    upstream = set(signature(TabPFNClassifier).parameters) - {'n_jobs', 'n_estimators'}
    public = set(signature(tabpfn_binary.tabpfn_binary).parameters) - {'data', 'prediction_calibration_config'}
    assert public == upstream | {'n_ensemble_configurations'}

    source = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'
    bars = pl.read_parquet(source, n_rows=384)
    parameters = {name: [values[0]] for name, values in tabpfn_binary.params().items()}
    parameters['n_ensemble_configurations'] = [1]
    parameters.update(softmax_temperature=[1.1], balance_probabilities=[True],
                      average_before_softmax=[True])
    loop = UniversalExperimentLoop(data=bars, sfd=tabpfn_binary)
    loop.run(str(tmp_path / 'tabpfn'), n_permutations=1, prep_each_round=True,
             random_search=False, params=lambda: parameters, post_processing=True,
             progress_bar=False)

    assert loop.experiment_log.height == len(loop.models) == len(loop.preds) == 1
    assert loop.experiment_log['strict_mode_error'].null_count() == 1
    model = loop.models[0]
    assert isinstance(model, TabPFNBinary)
    assert isinstance(model.model, TabPFNClassifier)
    for name in ('softmax_temperature', 'balance_probabilities', 'average_before_softmax'):
        assert model.model.get_params()[name] == parameters[name][0]
    prepared = loop.manifest.prepare_data(bars, loop.round_params[0])
    result = model.predict(prepared)
    truth = np.asarray(prepared['y_test'])
    predictions, probabilities = result['_preds'], result['_probs']
    assert truth.size > 0 and set(truth) == {0, 1}
    assert predictions.shape == probabilities.shape == truth.shape
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_array_equal(predictions, probabilities >= result['optimal_threshold'])
    np.testing.assert_array_equal(loop.preds[0], predictions)
    assert loop.experiment_log['accuracy'][0] == float(np.mean(truth == predictions))
    differences = probabilities[truth == 1, None] - probabilities[truth == 0]
    auc = np.mean((differences > 0) + 0.5 * (differences == 0))
    np.testing.assert_allclose(loop.experiment_log['auc'][0], auc, atol=1e-12)


def test_tabpfn_compiled_parameters(tmp_path):
    if os.getenv('LIMEN_RUN_TABPFN_MODEL_TEST') != '1':
        pytest.skip('Set LIMEN_RUN_TABPFN_MODEL_TEST=1 for the real TabPFN CPU test')

    import json
    from ruamel.yaml import YAML
    from limen.experiment.param_domain import ParamDomain
    from limen.experiment.param_search import GridStrategy
    from limen.yaml.compiler import CompiledSFD

    root = Path(__file__).resolve().parents[1]
    source = root / 'tests/fixtures/spot_15m_20250101_20250531.parquet'
    config = YAML(typ='safe').load(root / 'docs/examples/compiled-research/tabpfn.yaml')
    config['sfd']['manifest']['data_source']['params']['file_path_or_url'] = str(source)
    selected = {'softmax_temperature': 1.1, 'balance_probabilities': True,
                'average_before_softmax': True, 'random_state': 7,
                'categorical_features_indices': [0],
                'inference_config': {'OUTLIER_REMOVAL_STD': 2.5}}
    config['sfd']['params'].update({name: [value] for name, value in selected.items()})
    sfd = CompiledSFD(config)
    loop = UniversalExperimentLoop(sfd=sfd,
             search_strategy=GridStrategy(ParamDomain(sfd.params())),
             experiment_dir=tmp_path, yaml_reference=config)
    loop.run('compiled-tabpfn', n_permutations=1, prep_each_round=True,
             post_processing=True, progress_bar=False)

    assert loop.experiment_log.height == len(loop.models) == 1
    assert loop.experiment_log['strict_mode_error'].null_count() == 1
    assert isinstance(loop.models[0], TabPFNBinary)
    for name, value in selected.items():
        assert loop.models[0].model.get_params()[name] == value
        assert loop.round_params[0][name] == value
        if not isinstance(value, (list, dict)):
            assert loop.experiment_log[name][0] == value
    saved = json.loads((tmp_path / 'metadata.json').read_text())
    for name, value in selected.items():
        assert saved['yaml_reference']['sfd']['params'][name] == [value]
    prepared = loop.manifest.prepare_data(loop.data, loop.round_params[0])
    predicted = loop.models[0].predict(prepared)
    truth = np.asarray(prepared['y_test'])
    assert set(truth) == {0, 1}
    assert np.isfinite(predicted['_probs']).all()
    assert ((predicted['_probs'] >= 0) & (predicted['_probs'] <= 1)).all()
    assert loop.experiment_log['accuracy'][0] == float(np.mean(truth == predicted['_preds']))
    differences = predicted['_probs'][truth == 1, None] - predicted['_probs'][truth == 0]
    np.testing.assert_allclose(loop.experiment_log['auc'][0], np.mean((differences > 0) + 0.5 * (differences == 0)), atol=1e-12)

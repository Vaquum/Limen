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

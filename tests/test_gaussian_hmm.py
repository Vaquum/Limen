import inspect
import json
from math import erf, sqrt
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hmmlearn.hmm import GaussianHMM
from ruamel.yaml import YAML

from limen import Trainer, UniversalExperimentLoop
from limen.experiment import _round_storage
from limen.sfd.foundational_sfd import gaussian_hmm_binary as foundational
from limen.sfd.reference_architecture import GaussianHMMBinary, gaussian_hmm_binary
from limen.yaml import CompiledSFD, build_search_strategy, validate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def recorded():
    bars = pl.read_parquet(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet', n_rows=768)
    parameters = {key: values[0] for key, values in foundational.params().items()}
    manifest = foundational.manifest()
    return bars, manifest, parameters, manifest.prepare_data(bars, parameters)


@pytest.mark.parametrize('covariance_type', ['diag', 'spherical', 'full', 'tied'])
@pytest.mark.parametrize('implementation', ['scaling', 'log'])
def test_recorded_hmm_probabilities_are_terminal_forward_mixtures(recorded, covariance_type, implementation):
    _, manifest, params, data = recorded
    selected = params | {'covariance_type': covariance_type, 'implementation': implementation}
    result = manifest.run_model(data, selected)
    model = result['_model']
    assert isinstance(model, GaussianHMMBinary) and isinstance(model.model, GaussianHMM)
    controls = set(inspect.signature(gaussian_hmm_binary).parameters) - {'data'}
    assert controls <= foundational.params().keys()
    for name in controls:
        assert getattr(model.model, name) == selected[name]
    prediction = model.predict({'x_test': data['x_test']})
    probability = prediction['_probs']
    assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
    np.testing.assert_array_equal(prediction['_preds'], probability > .5)
    truth = np.asarray(data['y_test'])
    assert result['accuracy'] == float(np.mean(truth == prediction['_preds']))
    differences = probability[truth == 1, None] - probability[truth == 0]
    np.testing.assert_allclose(result['auc'], np.mean((differences > 0) + .5 * (differences == 0)), atol=1e-12)
    x = data['x_test']
    window = x.select(model._columns).to_numpy()[0] * 100
    backend = model.model
    means, variance = backend.means_[:, 0], backend.covars_[:, 0, 0]
    alpha = backend.startprob_.copy()
    for index, value in enumerate(window):
        if index:
            alpha = alpha @ backend.transmat_
        emission = np.exp(-.5 * (value - means) ** 2 / variance) / np.sqrt(2 * np.pi * variance)
        alpha *= emission
        alpha /= alpha.sum()
    tail = np.asarray([.5 * (1 + erf(float(mu) / sqrt(2 * float(var)))) for mu, var in zip(means, variance, strict=True)])
    np.testing.assert_allclose(probability[0], alpha @ backend.transmat_ @ tail, atol=1e-12)
    prefix = model.predict({'x_test': x.head(5)})['_probs']
    np.testing.assert_array_equal(prefix, probability[:5])
    rows = [model.predict({'x_test': x.slice(i, 1)})['_probs'][0] for i in range(5)]
    np.testing.assert_array_equal(prefix, rows)
    np.testing.assert_array_equal(model.predict({'x_test': x.to_numpy()})['_probs'], probability)


def test_compiled_hmm_recording_and_native_reconstruction(tmp_path, recorded):
    bars, _, _, _ = recorded
    config = YAML(typ='safe').load(ROOT / 'limen/yaml/templates/gaussian_hmm_binary.yaml')
    config['sfd']['manifest']['data_source'] = {'method': 'limen.data.HistoricalData.get_any_file',
        'params': {'file_path_or_url': str(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet')}}
    assert validate(config).valid
    loop = UniversalExperimentLoop(sfd=CompiledSFD(config), data=bars, experiment_dir=tmp_path,
                                   yaml_reference=config, search_strategy=build_search_strategy(config))
    loop.run('recorded_hmm', n_permutations=1, prep_each_round=True, progress_bar=False,
             record_model_outputs=True, recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True))
    assert loop.experiment_log.height == 1 and loop.experiment_log['strict_mode_error'].null_count() == 1
    record = json.loads(_round_storage.text(_round_storage.round_path(tmp_path)).splitlines()[0])
    np.testing.assert_array_equal(record['preds'], np.asarray(record['probs']) > .5)
    sensor = Trainer(tmp_path, data=bars).train([record['round_id']])[0]
    rows = sensor.predict_all(bars)
    assert any(row.reason is None and row.prediction == int(row.probability > .5) for row in rows)
    assert build_search_strategy(config).domain.total_combinations == 16

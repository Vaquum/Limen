from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from ruamel.yaml import YAML
import torch

from limen import Trainer, UniversalExperimentLoop
from limen.experiment import _round_storage
from limen.sfd.foundational_sfd import lstm_binary as foundational
from limen.sfd.reference_architecture import LSTMBinary, lstm_binary
from limen.yaml import CompiledSFD, build_search_strategy, validate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def recorded():
    bars = pl.read_parquet(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet', n_rows=768)
    params = {key: values[0] for key, values in foundational.params().items()}
    params.update(epochs=2, batch_size=32, hidden_size=8, num_layers=2, dropout=0.1)
    manifest = foundational.manifest()
    return bars, params, manifest, manifest.prepare_data(bars, params)


@pytest.mark.parametrize('bidirectional', [False, True])
def test_recorded_lstm_trains_scales_and_reconstructs_probabilities(recorded, monkeypatch, bidirectional):
    _, params, manifest, original = recorded
    data = original | {'_record_model_outputs': True, '_alignment': deepcopy(original['_alignment'])}
    before = torch.get_rng_state().clone()
    steps = []
    step = torch.optim.Adam.step

    def observed(optimizer, *args, **kwargs):
        parameter = optimizer.param_groups[0]['params'][0]
        prior = parameter.detach().clone()
        value = step(optimizer, *args, **kwargs)
        steps.append(not torch.equal(prior, parameter))
        assert optimizer.defaults['lr'] == params['learning_rate']
        assert optimizer.defaults['betas'] == params['betas']
        return value

    monkeypatch.setattr(torch.optim.Adam, 'step', observed)
    result = manifest.run_model(data, params | {'bidirectional': bidirectional, 'bias': False})
    model = result['_model']
    assert isinstance(model, LSTMBinary) and isinstance(model.model.sequence, torch.nn.LSTM)
    assert model.model.sequence.bidirectional is bidirectional and not model.model.sequence.bias
    assert model.model.sequence.hidden_size == 8 and model.model.sequence.num_layers == 2
    assert model.model.sequence.dropout == 0.1
    assert len(steps) == params['epochs'] * int(np.ceil(len(data['y_train']) / params['batch_size'])) and any(steps)
    assert torch.equal(torch.get_rng_state(), before)
    columns = ['ret_1_lag_' + str(i) for i in range(params['lookback_end'], -1, -1)]
    train = data['x_train'].select(columns).to_numpy()
    assert model._mean == float(train.mean()) and model._scale == float(train.std())
    assert model._columns == columns
    prediction = model.predict({'x_test': data['x_test']})
    probabilities = prediction['_probs']
    values = (data['x_test'].select(columns).to_numpy() - train.mean()) / train.std()
    with torch.no_grad():
        _, (hidden, _) = model.model.sequence(torch.tensor(values[:, :, None], dtype=torch.float32))
    directions = 2 if bidirectional else 1
    final = hidden[-directions:].transpose(0, 1).flatten(1).numpy()
    logits = final @ model.model.head.weight.detach().numpy().T + model.model.head.bias.detach().numpy()
    np.testing.assert_allclose(probabilities, (1 / (1 + np.exp(-logits))).ravel(), atol=1e-7)
    truth = np.asarray(data['y_test'])
    assert result['accuracy'] == float(np.mean(prediction['_preds'] == truth))
    differences = probabilities[truth == 1, None] - probabilities[truth == 0]
    np.testing.assert_allclose(result['auc'], np.mean((differences > 0) + .5 * (differences == 0)), atol=1e-12)
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_array_equal(result['_preds'], np.asarray(data['_alignment']['model_outputs']['probs']) > .5)
    for index in range(5):
        np.testing.assert_allclose(model.predict({'x_test': data['x_test'].slice(index, 1)})['_probs'], probabilities[index:index + 1], atol=1e-7)
    np.testing.assert_allclose(model.predict({'x_test': data['x_test'].head(5)})['_probs'], probabilities[:5], atol=1e-7)
    np.testing.assert_array_equal(model.predict({'x_test': data['x_test']})['_preds'], prediction['_preds'])
    assert torch.equal(torch.get_rng_state(), before)


def test_lstm_compiled_recording_and_reconstruction(tmp_path, recorded):
    bars, _, _, _ = recorded
    config = YAML(typ='safe').load(ROOT / 'limen/yaml/templates/lstm_binary.yaml')
    config['sfd']['manifest']['data_source'] = {'method': 'limen.data.HistoricalData.get_any_file',
        'params': {'file_path_or_url': str(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet')}}
    config['sfd']['params']['epochs'] = [2]
    assert validate(config).valid
    loop = UniversalExperimentLoop(sfd=CompiledSFD(config), data=bars, experiment_dir=tmp_path,
                                   search_strategy=build_search_strategy(config), yaml_reference=config)
    loop.run('recorded_lstm', n_permutations=1, prep_each_round=True, progress_bar=False,
             record_model_outputs=True, recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True))
    assert loop.experiment_log.height == 1 and loop.experiment_log['strict_mode_error'].null_count() == 1
    record = json.loads(_round_storage.text(_round_storage.round_path(tmp_path)).splitlines()[0])
    np.testing.assert_array_equal(record['preds'], np.asarray(record['probs']) > .5)
    sensor = Trainer(tmp_path, data=bars).train([record['round_id']])[0]
    rows = sensor.predict_all(bars)
    assert any(row.reason is None and row.prediction == int(row.probability > .5) for row in rows)


def test_lstm_template_covers_native_and_training_controls():
    config = YAML(typ='safe').load(ROOT / 'limen/yaml/templates/lstm_binary.yaml')
    assert validate(config).valid
    controls = set(inspect.signature(lstm_binary).parameters) - {'data'}
    assert controls <= config['sfd']['params'].keys() and controls <= foundational.params().keys()
    assert build_search_strategy(config).domain.total_combinations == 16

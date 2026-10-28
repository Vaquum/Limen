from copy import deepcopy
from importlib import import_module
import inspect
import json
from math import erf, sqrt
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from ruamel.yaml import YAML

from limen import Trainer, UniversalExperimentLoop
from limen.experiment import _round_storage
from limen.sfd.foundational_sfd import cnn1d_binary as foundational
from limen.sfd.reference_architecture import CNN1DBinary, cnn1d_binary
from limen.yaml import CompiledSFD, build_search_strategy, validate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def torch():
    import torch
    return torch


@pytest.fixture(scope='module')
def recorded():
    bars = pl.read_parquet(ROOT / 'tests/fixtures/spot_1h_20240101_20241231.parquet', n_rows=768)
    params = {key: values[0] for key, values in foundational.params().items()}
    params.update(epochs=2, batch_size=32, out_channels=4, conv_layers=2, dropout=.1)
    manifest = foundational.manifest()
    return bars, params, manifest, manifest.prepare_data(bars, params)


def _convolution_oracle(network, values, pooling, torch):
    inputs = values[:, None, :].astype(np.float32)
    for layer in network.blocks:
        if isinstance(layer, torch.nn.Conv1d):
            weights = layer.weight.detach().numpy()
            kernel, stride, dilation = weights.shape[2], layer.stride[0], layer.dilation[0]
            starts = range(0, inputs.shape[2] - dilation * (kernel - 1), stride)
            outputs = [np.einsum('bck,ock->bo', inputs[:, :, start + np.arange(kernel) * dilation], weights) for start in starts]
            inputs = np.stack(outputs, axis=2)
            if layer.bias is not None:
                inputs += layer.bias.detach().numpy()[None, :, None]
        elif isinstance(layer, torch.nn.ReLU):
            inputs = np.maximum(inputs, 0)
        elif isinstance(layer, torch.nn.Tanh):
            inputs = np.tanh(inputs)
        else:
            assert isinstance(layer, torch.nn.GELU)
            inputs = .5 * inputs * (1 + np.vectorize(erf)(inputs / sqrt(2)))
    pooled = inputs.mean(axis=2) if pooling == 'mean' else inputs.max(axis=2)
    logits = pooled @ network.head.weight.detach().numpy().T + network.head.bias.detach().numpy()
    return (1 / (1 + np.exp(-logits))).ravel()


@pytest.mark.parametrize('activation,pooling,stride,dilation', [('relu', 'mean', 1, 1), ('tanh', 'max', 2, 1), ('gelu', 'mean', 1, 2)])
def test_recorded_cnn_time_axis_gradient_controls_and_probabilities(recorded, monkeypatch, torch, activation, pooling, stride, dilation):
    _, params, manifest, original = recorded
    data = original | {'_record_model_outputs': True, '_alignment': deepcopy(original['_alignment'])}
    before = torch.get_rng_state().clone()
    updates = []
    step = torch.optim.Adam.step

    def observed(optimizer, *args, **kwargs):
        parameter = optimizer.param_groups[0]['params'][0]
        prior = parameter.detach().clone()
        value = step(optimizer, *args, **kwargs)
        updates.append(not torch.equal(prior, parameter))
        assert optimizer.defaults['lr'] == params['learning_rate'] and optimizer.defaults['betas'] == params['betas']
        return value

    monkeypatch.setattr(torch.optim.Adam, 'step', observed)
    result = manifest.run_model(data, params | {'activation': activation, 'pooling': pooling, 'stride': stride, 'dilation': dilation, 'bias': False})
    model = result['_model']
    assert isinstance(model, CNN1DBinary)
    convolution = model.model.blocks[0]
    assert convolution.in_channels == 1 and convolution.out_channels == 4
    assert convolution.kernel_size == (3,) and convolution.stride == (stride,) and convolution.dilation == (dilation,)
    assert convolution.bias is None and model.model.dropout.p == .1 and not model.model.training
    assert len(updates) == params['epochs'] * int(np.ceil(len(data['y_train']) / params['batch_size'])) and any(updates)
    assert torch.equal(torch.get_rng_state(), before)
    columns = ['ret_1_lag_' + str(i) for i in range(params['lookback_end'], -1, -1)]
    train = data['x_train'].select(columns).to_numpy()
    assert model._columns == columns and model._mean == float(train.mean()) and model._scale == float(train.std())
    probability = model.predict(data)['_probs']
    values = (data['x_test'].select(columns).to_numpy() - train.mean()) / train.std()
    np.testing.assert_allclose(probability, _convolution_oracle(model.model, values, pooling, torch), atol=5e-7)
    prediction = probability > .5
    truth = np.asarray(data['y_test'])
    assert result['accuracy'] == float(np.mean(prediction == truth))
    differences = probability[truth == 1, None] - probability[truth == 0]
    np.testing.assert_allclose(result['auc'], np.mean((differences > 0) + .5 * (differences == 0)))
    np.testing.assert_array_equal(result['_preds'], np.asarray(data['_alignment']['model_outputs']['probs']) > .5)
    for index in range(5):
        np.testing.assert_allclose(model.predict({'x_test': data['x_test'].slice(index, 1)})['_probs'], probability[index:index + 1], atol=1e-7)
    np.testing.assert_allclose(model.predict({'x_test': data['x_test'].head(5)})['_probs'], probability[:5], atol=1e-7)
    assert torch.equal(torch.get_rng_state(), before)


def test_cnn_compiled_native_source_recording_and_reconstruction(tmp_path, recorded, monkeypatch):
    bars, _, _, _ = recorded
    config = YAML(typ='safe').load(ROOT / 'limen/yaml/templates/cnn1d_binary.yaml')
    config['sfd']['params']['epochs'] = [2]
    monkeypatch.setattr(import_module('limen.data.historical_data'), '_read_any_file', lambda *_args, **_kwargs: bars)
    assert validate(config).valid
    compiled = CompiledSFD(config)
    admitted = compiled.manifest().fetch_data()
    loop = UniversalExperimentLoop(sfd=compiled, data=admitted, experiment_dir=tmp_path,
                                   search_strategy=build_search_strategy(config), yaml_reference=config)
    loop.run('recorded_cnn', n_permutations=1, prep_each_round=True, progress_bar=False,
             record_model_outputs=True, recording=dict.fromkeys(('predictions', 'alignment', 'trade_ledger', 'fold_details'), True))
    assert loop.experiment_log.height == 1 and loop.experiment_log['strict_mode_error'].null_count() == 1
    record = json.loads(_round_storage.text(_round_storage.round_path(tmp_path)).splitlines()[0])
    np.testing.assert_array_equal(record['preds'], np.asarray(record['probs']) > .5)
    sensor = Trainer(tmp_path, data=bars).train([record['round_id']])[0]
    assert any(row.reason is None and row.prediction == int(row.probability > .5) for row in sensor.predict_all(bars))


def test_cnn_native_controls_and_invalid_receptive_field(recorded):
    _, _, _, data = recorded
    config = YAML(typ='safe').load(ROOT / 'limen/yaml/templates/cnn1d_binary.yaml')
    controls = set(inspect.signature(cnn1d_binary).parameters) - {'data'}
    assert validate(config).valid and controls <= config['sfd']['params'].keys() and controls <= foundational.params().keys()
    assert build_search_strategy(config).domain.total_combinations == 16
    for params in ({'conv_layers': 0}, {'activation': 'sigmoid'}, {'pooling': 'sum'}):
        with pytest.raises(ValueError, match='CNN1DBinary'):
            CNN1DBinary().train(data, **params)
    with pytest.raises(RuntimeError, match='Kernel size'):
        CNN1DBinary().train(data, kernel_size=data['x_train'].width * 2)

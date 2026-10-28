import inspect
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl

from limen.sfd.foundational_sfd import tabpfn_binary as foundational
from limen.sfd.reference_architecture.tabpfn_binary import TabPFNBinary, tabpfn_binary


def test_tabpfn_controls_reach_the_backend_without_coercion(monkeypatch):
    architecture = import_module('limen.sfd.reference_architecture.tabpfn_binary')

    captured = {}

    class Classifier:
        def __init__(self, **options):
            captured.update(options)

        def fit(self, X, y):
            assert X.shape[0] == y.shape[0] > 0

    monkeypatch.setattr(architecture, 'require_optional',
                        lambda *args: SimpleNamespace(TabPFNClassifier=Classifier))
    monkeypatch.setattr(TabPFNBinary, 'evaluate', lambda *args, **kwargs: {})
    manifest = foundational.manifest()
    bars = pl.read_parquet(Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet', n_rows=384)
    selected = {name: values[0] for name, values in foundational.params().items()}
    selected.update(categorical_features_indices=[0], softmax_temperature=1.1,
                    balance_probabilities=True, average_before_softmax=True,
                    model_path=Path('checkpoint.ckpt'), ignore_pretraining_limits=False,
                    inference_precision='auto', fit_mode='low_memory', memory_saving_mode=False,
                    random_state=np.random.default_rng(7), n_preprocessing_jobs=2,
                    inference_config={'OUTLIER_REMOVAL_STD': 2.5},
                    differentiable_input=True, eval_metric='roc_auc', tuning_config=None)
    prepared = manifest.prepare_data(bars, selected)
    resolved = manifest.resolve_model_kwargs(selected)
    result = tabpfn_binary(prepared, **resolved)
    assert isinstance(result['_model'].model, Classifier)
    expected = {name: value for name, value in resolved.items() if name != 'prediction_calibration_config'}
    expected['n_estimators'] = expected.pop('n_ensemble_configurations')
    assert captured.keys() == expected.keys()
    for name, value in expected.items():
        assert captured[name] is value


def test_foundational_tabpfn_exposes_every_supported_control():
    controls = {'n_ensemble_configurations', 'device', 'categorical_features_indices',
                'softmax_temperature', 'balance_probabilities', 'average_before_softmax',
                'model_path', 'ignore_pretraining_limits', 'inference_precision', 'fit_mode',
                'memory_saving_mode', 'random_state', 'n_preprocessing_jobs', 'inference_config',
                'differentiable_input', 'eval_metric', 'tuning_config'}
    signature = inspect.signature(tabpfn_binary).parameters
    assert set(signature) == controls | {'data', 'prediction_calibration_config'}
    assert controls <= foundational.params().keys()
    assert foundational.params()['device'] == ['cpu']
    assert signature['n_ensemble_configurations'].default == 4
    assert signature['softmax_temperature'].default == 0.9
    assert 'n_estimators' not in signature and 'n_jobs' not in signature

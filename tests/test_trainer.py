import importlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import types

import polars as pl
import pytest

import limen.experiment.trainer.trainer as trainer_module
from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.trainer import ReconstructionError
from limen.experiment.trainer import Trainer
from limen.sfd.foundational_sfd import logreg_binary as logreg_sfd
from limen.sfd.foundational_sfd import random_binary as random_sfd
from limen.sfd.reference_architecture.base import ReferenceModel
from limen.experiment.param_search import RandomStrategy
from tests.utils.historical_data import get_cached_spot_klines_2h


class _ReferenceModelA(ReferenceModel):
    deterministic = True

    def train(self, data, **kwargs):
        return self

    def predict(self, data, **kwargs):
        return {'_preds': []}

    def evaluate(self, data, **kwargs):
        return {}


class _ReferenceModelB(ReferenceModel):
    deterministic = True

    def train(self, data, **kwargs):
        return self

    def predict(self, data, **kwargs):
        return {'_preds': []}

    def evaluate(self, data, **kwargs):
        return {}


def test_trainer_end_to_end():

    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir) / 'experiment'
        params = logreg_sfd.params()
        domain = ParamDomain(params)
        strategy = RandomStrategy(domain, seed=42)

        uel = UniversalExperimentLoop(
            sfd=logreg_sfd,
            search_strategy=strategy,
            experiment_dir=experiment_dir,
            test_mode=True,
        )

        uel.run(
            experiment_name='test_trainer',
            n_permutations=2,
        )

        # metadata.json written with correct fields
        metadata_path = experiment_dir / 'metadata.json'
        assert metadata_path.exists()
        with metadata_path.open('r') as f:
            metadata = json.load(f)
        assert metadata['sfd_module'] == 'limen.sfd.foundational_sfd.logreg_binary'
        assert 'limen_version' in metadata
        assert 'created_at' in metadata

        # Trainer loads manifest, params, round data, and original log
        trainer = Trainer(experiment_dir, data=uel.data)
        assert trainer._manifest is not None
        assert len(trainer._param_keys) > 0
        assert len(trainer._round_data) == 2
        assert trainer._original_log is not None
        assert len(trainer._original_log) == 2

        # train returns Sensor instances with trained models
        permutation_ids = list(trainer._round_data.keys())
        sensors = trainer.train(permutation_ids)
        assert len(sensors) == 2

        for pid, sensor in zip(permutation_ids, sensors, strict=True):
            assert sensor.permutation_id == pid
            assert sensor.round_params is not None
            assert sensor.metadata is not None
            assert sensor.results is not None
            assert sensor.model is not None
            assert isinstance(sensor.model, ReferenceModel)

        # Sensor is callable with trained model
        data_dict = trainer._manifest.prepare_data(uel.data, sensors[0].round_params)
        result = sensors[0](data_dict)
        assert isinstance(result, dict)

        # invalid permutation ID raises ValueError
        try:
            trainer.train([99999])
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert '99999' in str(e)


def test_reconstruction_error_stochastic():

    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir) / 'experiment'
        params = random_sfd.params()
        domain = ParamDomain(params)
        strategy = RandomStrategy(domain, seed=42)

        uel = UniversalExperimentLoop(
            sfd=random_sfd,
            search_strategy=strategy,
            experiment_dir=experiment_dir,
            test_mode=True,
        )

        uel.run(
            experiment_name='test_stochastic',
            n_permutations=2,
        )

        trainer = Trainer(experiment_dir, data=uel.data)
        permutation_ids = list(trainer._round_data.keys())

        try:
            trainer.train(permutation_ids)
            assert False, 'Expected ReconstructionError'
        except ReconstructionError as e:
            assert 'metric mismatch' in str(e).lower()


def test_reconstruction_error_tampered_log():

    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir) / 'experiment'
        params = logreg_sfd.params()
        domain = ParamDomain(params)
        strategy = RandomStrategy(domain, seed=42)

        uel = UniversalExperimentLoop(
            sfd=logreg_sfd,
            search_strategy=strategy,
            experiment_dir=experiment_dir,
            test_mode=True,
        )

        uel.run(
            experiment_name='test_tampered',
            n_permutations=2,
        )

        trainer = Trainer(experiment_dir, data=uel.data)
        permutation_ids = list(trainer._round_data.keys())

        # Tamper with original log to force mismatch
        pid = permutation_ids[0]
        for key, value in trainer._original_log[pid].items():
            if isinstance(value, float) and key not in ('id', '_id'):
                trainer._original_log[pid][key] = value + 999.0
                break

        try:
            trainer.train([pid])
            assert False, 'Expected ReconstructionError'
        except ReconstructionError as e:
            assert str(pid) in str(e)


def test_deterministic_validation():

    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir) / 'experiment'
        params = logreg_sfd.params()
        domain = ParamDomain(params)
        strategy = RandomStrategy(domain, seed=42)

        uel = UniversalExperimentLoop(
            sfd=logreg_sfd,
            search_strategy=strategy,
            experiment_dir=experiment_dir,
            test_mode=True,
        )

        uel.run(
            experiment_name='test_logreg',
            n_permutations=2,
        )

        trainer = Trainer(experiment_dir, data=uel.data)
        permutation_ids = list(trainer._round_data.keys())
        sensors = trainer.train(permutation_ids)

        # Deterministic model produces zero mismatches
        for pid, sensor in zip(permutation_ids, sensors, strict=True):
            mismatches = trainer._validate_metrics(pid, sensor.results, True)
            assert mismatches == [], f"Permutation {pid} had mismatches: {mismatches}"


def test_pass2_uses_full_data():

    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir) / 'experiment'
        params = logreg_sfd.params()
        domain = ParamDomain(params)
        strategy = RandomStrategy(domain, seed=42)

        uel = UniversalExperimentLoop(
            sfd=logreg_sfd,
            search_strategy=strategy,
            experiment_dir=experiment_dir,
            test_mode=True,
        )

        uel.run(
            experiment_name='test_full_data',
            n_permutations=1,
        )

        trainer = Trainer(experiment_dir, data=uel.data)
        pid = next(iter(trainer._round_data.keys()))
        round_params = dict(trainer._round_data[pid]['round_params'])

        # Pass 1 data has original split — val and test are non-empty
        pass1_data = trainer._manifest.prepare_data(uel.data, round_params)
        assert len(pass1_data['x_val']) > 0
        assert len(pass1_data['x_test']) > 0
        pass1_train_rows = len(pass1_data['x_train'])

        # Pass 2 data uses split_config=(1,0,0) — all data in training
        manifest_full = trainer._manifest.with_params_override(split_config=(1, 0, 0))
        pass2_data = manifest_full.prepare_data(uel.data, round_params)
        pass2_train_rows = len(pass2_data['x_train'])
        assert len(pass2_data['x_val']) == 0
        assert len(pass2_data['x_test']) == 0
        assert pass2_train_rows > pass1_train_rows


def test_sensor_inference():

    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir) / 'experiment'
        params = logreg_sfd.params()
        domain = ParamDomain(params)
        strategy = RandomStrategy(domain, seed=42)

        uel = UniversalExperimentLoop(
            sfd=logreg_sfd,
            search_strategy=strategy,
            experiment_dir=experiment_dir,
            test_mode=True,
        )

        uel.run(
            experiment_name='test_inference',
            n_permutations=1,
        )

        trainer = Trainer(experiment_dir, data=uel.data)
        pid = next(iter(trainer._round_data.keys()))
        sensors = trainer.train([pid])
        sensor = sensors[0]

        # sensor.predict() works on feature-only data (no labels needed)
        data_dict = trainer._manifest.prepare_data(uel.data, sensor.round_params)
        live_data = {'x_test': data_dict['x_test']}
        result = sensor.predict(live_data)
        assert isinstance(result, dict)
        assert '_preds' in result
        assert '_probs' in result
        assert len(result['_preds']) == len(data_dict['x_test'])

        # __call__ delegates to predict()
        result_via_call = sensor(live_data)
        assert result_via_call.keys() == result.keys()


def test_trainer_with_feature_ablation():

    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir) / 'experiment'
        params = logreg_sfd.params()
        params['feature_drop_count'] = [1]
        params['feature_drop_seed'] = [42]
        domain = ParamDomain(params)
        strategy = RandomStrategy(domain, seed=42)

        uel = UniversalExperimentLoop(
            sfd=logreg_sfd,
            search_strategy=strategy,
            experiment_dir=experiment_dir,
            test_mode=True,
        )

        uel.run(
            experiment_name='test_ablation',
            n_permutations=1,
        )

        # _dropped_features stored in round_data.jsonl
        trainer = Trainer(experiment_dir, data=uel.data)
        pid = next(iter(trainer._round_data.keys()))
        rp = trainer._round_data[pid]['round_params']
        assert '_dropped_features' in rp
        assert len(rp['_dropped_features']) == 1

        # Trainer reproduces same drops
        train_rp = dict(rp)
        trainer._manifest.prepare_data(uel.data, train_rp)
        assert train_rp['_dropped_features'] == rp['_dropped_features']

        # Trainer Pass 1 validates with ablation active
        sensors = trainer.train([pid])
        assert len(sensors) == 1
        assert sensors[0].model is not None


def test_trainer_with_fractional_diff():

    large_data = get_cached_spot_klines_2h(20000)

    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir) / 'experiment'
        params = logreg_sfd.params()
        params['frac_diff_d'] = [0.4]
        domain = ParamDomain(params)
        strategy = RandomStrategy(domain, seed=42)

        uel = UniversalExperimentLoop(
            sfd=logreg_sfd,
            search_strategy=strategy,
            experiment_dir=experiment_dir,
            data=large_data,
        )

        uel.run(
            experiment_name='test_fracdiff',
            n_permutations=1,
        )

        trainer = Trainer(experiment_dir, data=large_data)
        pid = next(iter(trainer._round_data.keys()))

        # Trainer Pass 1 validates with fracdiff active
        sensors = trainer.train([pid])
        assert len(sensors) == 1
        assert sensors[0].model is not None


def test_load_round_data_skips_blank_and_malformed_lines() -> None:
    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir)
        (experiment_dir / 'round_data.jsonl').write_text(
            '\n'
            '{"round_id": 0, "round_params": {"alpha": 1}}\n'
            'not-json\n'
            '{"round_id": 1, "round_params": {"alpha": 2}}\n'
        )

        trainer = object.__new__(Trainer)
        trainer._experiment_dir = experiment_dir

        round_data = trainer._load_round_data()

    assert list(round_data) == [0, 1]
    assert round_data[1]['round_params'] == {'alpha': 2}


def test_load_round_data_requires_round_data_file() -> None:
    with TemporaryDirectory() as tmpdir:
        trainer = object.__new__(Trainer)
        trainer._experiment_dir = Path(tmpdir)

        with pytest.raises(FileNotFoundError, match=r'round_data\.jsonl'):
            trainer._load_round_data()


def test_load_original_log_returns_none_when_results_csv_is_missing() -> None:
    with TemporaryDirectory() as tmpdir:
        trainer = object.__new__(Trainer)
        trainer._experiment_dir = Path(tmpdir)

        assert trainer._load_original_log() is None


def test_resolve_model_class_requires_configured_architecture_function() -> None:
    trainer = object.__new__(Trainer)
    trainer._manifest = SimpleNamespace(architecture_function=None)

    with pytest.raises(ValueError, match='Architecture function not configured'):
        trainer._resolve_model_class()


def test_resolve_model_class_rejects_modules_without_reference_model_subclasses() -> None:
    def architecture_function():
        return None

    architecture_function.__module__ = 'dummy.no_model'
    trainer = object.__new__(Trainer)
    trainer._manifest = SimpleNamespace(architecture_function=architecture_function)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            trainer_module.importlib,
            'import_module',
            lambda module_name: types.SimpleNamespace(NotAReferenceModel=object),
        )

        with pytest.raises(ValueError, match='No ReferenceModel subclass found'):
            trainer._resolve_model_class()


def test_resolve_model_class_rejects_modules_with_multiple_reference_model_subclasses() -> None:
    def architecture_function():
        return None

    architecture_function.__module__ = 'dummy.multi_model'
    trainer = object.__new__(Trainer)
    trainer._manifest = SimpleNamespace(architecture_function=architecture_function)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            trainer_module.importlib,
            'import_module',
            lambda module_name: types.SimpleNamespace(
                ModelA=_ReferenceModelA,
                ModelB=_ReferenceModelB,
            ),
        )

        with pytest.raises(ValueError, match='Multiple ReferenceModel subclasses found'):
            trainer._resolve_model_class()


def test_validate_metrics_ignores_metadata_fields_and_accepts_small_stochastic_drift() -> None:
    trainer = object.__new__(Trainer)
    trainer._original_log = {
        7: {
            'id': 7,
            'param_alpha': 1.0,
            'accuracy': 0.8000001,
            'precision': 0.2000001,
            'label': 'baseline',
        }
    }
    trainer._param_keys = frozenset({'param_alpha'})

    mismatches = trainer._validate_metrics(
        7,
        {
            'id': 7,
            'param_alpha': 99.0,
            'accuracy': 0.8000002,
            'precision': 0.2000002,
            '_probs': [0.1, 0.9],
            'label': 'changed-string-is-ignored',
        },
        is_deterministic=False,
    )

    assert mismatches == []


def test_validate_metrics_reports_missing_permutations_and_large_deterministic_mismatches() -> None:
    trainer = object.__new__(Trainer)
    trainer._original_log = {
        3: {
            'accuracy': 0.80,
            'precision': 0.20,
        }
    }
    trainer._param_keys = frozenset()

    assert trainer._validate_metrics(99, {'accuracy': 0.5}, True) == [
        'permutation 99 not found in results.csv'
    ]

    mismatches = trainer._validate_metrics(
        3,
        {'accuracy': 0.81, 'precision': 0.20},
        is_deterministic=True,
    )

    assert mismatches == ['accuracy: original=0.8, new=0.81']


_SELF_CONTAINED_SFD_SOURCE = '''
from types import SimpleNamespace


def manifest():
    return SimpleNamespace(architecture_function=None)


def params():
    return {'alpha': 0.5}
'''


def test_trainer_loads_sfd_from_experiment_dir_without_sys_path_mutation() -> None:

    '''
    Pin the self-contained-experiment contract: Trainer(experiment_dir)
    must succeed when the SFD .py lives inside experiment_dir and is
    referenced by bare module name in metadata.json, even though that
    name is not on sys.path. The loaded module must be registered in
    sys.modules under the bare name so downstream
    importlib.import_module(name) lookups (e.g. _resolve_model_class)
    resolve to the same module instance.
    '''

    sys_path_before = list(sys.path)
    sfd_module_name = 'isolated_sfd_for_trainer_test'
    previous = sys.modules.pop(sfd_module_name, None)

    try:
        with TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir)

            (experiment_dir / f'{sfd_module_name}.py').write_text(
                _SELF_CONTAINED_SFD_SOURCE,
            )
            (experiment_dir / 'metadata.json').write_text(
                json.dumps({'sfd_module': sfd_module_name}),
            )
            (experiment_dir / 'round_data.jsonl').write_text('')

            trainer = Trainer(
                experiment_dir,
                data=pl.DataFrame({'x': [1, 2, 3]}),
            )

            assert trainer._param_keys == frozenset({'alpha'})
            assert trainer._manifest.architecture_function is None

            registered = sys.modules.get(sfd_module_name)
            assert registered is not None
            assert importlib.import_module(sfd_module_name) is registered
    finally:
        if previous is None:
            sys.modules.pop(sfd_module_name, None)
        else:
            sys.modules[sfd_module_name] = previous

    assert sys.path == sys_path_before


def test_trainer_loads_sfd_rolls_back_sys_modules_on_exec_failure() -> None:

    '''
    If the SFD module raises during exec_module, the partially
    initialised module must not remain in sys.modules and any
    previous entry under the same name must be restored.
    '''

    sfd_module_name = 'broken_sfd_for_trainer_test'
    previous = sys.modules.pop(sfd_module_name, None)

    try:
        with TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir)

            (experiment_dir / f'{sfd_module_name}.py').write_text(
                'raise RuntimeError("intentional sfd boom")\n',
            )
            (experiment_dir / 'metadata.json').write_text(
                json.dumps({'sfd_module': sfd_module_name}),
            )
            (experiment_dir / 'round_data.jsonl').write_text('')

            with pytest.raises(RuntimeError, match='intentional sfd boom'):
                Trainer(
                    experiment_dir,
                    data=pl.DataFrame({'x': [1, 2, 3]}),
                )

            assert sfd_module_name not in sys.modules
    finally:
        if previous is None:
            sys.modules.pop(sfd_module_name, None)
        else:
            sys.modules[sfd_module_name] = previous


def test_trainer_loads_sfd_restores_previous_sys_modules_entry_on_exec_failure() -> None:

    '''
    If a different module was already registered under
    `sfd_module_name`, an `exec_module` failure must restore that
    prior entry instead of leaving the slot empty.
    '''

    sfd_module_name = 'preexisting_sfd_for_trainer_test'
    sentinel = types.ModuleType(sfd_module_name)
    sentinel.__limen_test_sentinel__ = True
    saved = sys.modules.get(sfd_module_name)
    sys.modules[sfd_module_name] = sentinel

    try:
        with TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir)

            (experiment_dir / f'{sfd_module_name}.py').write_text(
                'raise RuntimeError("intentional sfd boom")\n',
            )
            (experiment_dir / 'metadata.json').write_text(
                json.dumps({'sfd_module': sfd_module_name}),
            )
            (experiment_dir / 'round_data.jsonl').write_text('')

            with pytest.raises(RuntimeError, match='intentional sfd boom'):
                Trainer(
                    experiment_dir,
                    data=pl.DataFrame({'x': [1, 2, 3]}),
                )

            assert sys.modules.get(sfd_module_name) is sentinel
    finally:
        if saved is None:
            sys.modules.pop(sfd_module_name, None)
        else:
            sys.modules[sfd_module_name] = saved


def test_trainer_falls_back_to_import_module_when_no_local_sfd_file() -> None:

    '''
    Built-in / packaged SFDs referenced by fully-qualified package
    name must continue to work via importlib.import_module when no
    `<sfd_module_name>.py` exists inside experiment_dir.
    '''

    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir)

        (experiment_dir / 'metadata.json').write_text(
            json.dumps({
                'sfd_module': 'limen.sfd.foundational_sfd.logreg_binary',
            }),
        )
        (experiment_dir / 'round_data.jsonl').write_text('')

        trainer = Trainer(
            experiment_dir,
            data=pl.DataFrame({'x': [1, 2, 3]}),
        )

        assert trainer._param_keys == frozenset(logreg_sfd.params().keys())


@pytest.mark.parametrize('bad_name', [
    '../etc/passwd',
    '..',
    '/etc/passwd',
    'foo/bar',
    'foo\\bar',
    'foo..bar',
    '.leading_dot',
    'trailing_dot.',
    '',
    '1starts_with_digit',
    'has space',
])
def test_trainer_rejects_path_traversal_and_invalid_module_names(bad_name: str) -> None:

    '''
    Pin the path-traversal hardening: `_load_sfd_module` must reject
    any `sfd_module` name that is not a dotted sequence of valid
    Python identifiers, before either the local-file or import_module
    branch runs.
    '''

    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir)

        (experiment_dir / 'metadata.json').write_text(
            json.dumps({'sfd_module': bad_name}),
        )
        (experiment_dir / 'round_data.jsonl').write_text('')

        with pytest.raises(ValueError, match='not a dotted sequence'):
            Trainer(
                experiment_dir,
                data=pl.DataFrame({'x': [1, 2, 3]}),
            )

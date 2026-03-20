import json
from pathlib import Path
from tempfile import TemporaryDirectory

from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.trainer import ReconstructionError
from limen.experiment.trainer import Trainer
from limen.sfd.foundational_sfd import logreg_binary as logreg_sfd
from limen.sfd.foundational_sfd import random_binary as random_sfd
from limen.sfd.reference_architecture.base import ReferenceModel
from tests.stubs.stubs import StubStrategy


def test_trainer_end_to_end():

    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir) / 'experiment'
        params = logreg_sfd.params()
        domain = ParamDomain(params)
        strategy = StubStrategy(domain)

        uel = UniversalExperimentLoop(
            sfd=logreg_sfd,
            search_strategy=strategy,
            experiment_dir=experiment_dir,
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

        for sensor in sensors:
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
        strategy = StubStrategy(domain)

        uel = UniversalExperimentLoop(
            sfd=random_sfd,
            search_strategy=strategy,
            experiment_dir=experiment_dir,
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
        strategy = StubStrategy(domain)

        uel = UniversalExperimentLoop(
            sfd=logreg_sfd,
            search_strategy=strategy,
            experiment_dir=experiment_dir,
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
        strategy = StubStrategy(domain)

        uel = UniversalExperimentLoop(
            sfd=logreg_sfd,
            search_strategy=strategy,
            experiment_dir=experiment_dir,
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

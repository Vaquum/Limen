import json
from pathlib import Path
from tempfile import TemporaryDirectory

from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.trainer import Trainer
from limen.sfd.foundational_sfd import random_binary as sfd_module
from tests.stubs.stubs import StubStrategy


def _run_experiment(tmpdir):

    experiment_dir = Path(tmpdir) / 'experiment'
    params = sfd_module.params()
    domain = ParamDomain(params)
    strategy = StubStrategy(domain)

    uel = UniversalExperimentLoop(
        sfd=sfd_module,
        search_strategy=strategy,
        experiment_dir=experiment_dir,
    )

    uel.run(
        experiment_name='test_trainer',
        n_permutations=3,
    )

    return uel, experiment_dir


def test_metadata_json_written():

    with TemporaryDirectory() as tmpdir:
        _, experiment_dir = _run_experiment(tmpdir)

        metadata_path = experiment_dir / 'metadata.json'
        assert metadata_path.exists()

        with metadata_path.open('r') as f:
            metadata = json.load(f)

        assert metadata['sfd_module'] == 'limen.sfd.foundational_sfd.random_binary'
        assert 'limen_version' in metadata
        assert 'created_at' in metadata


def test_trainer_loads_from_experiment_dir():

    with TemporaryDirectory() as tmpdir:
        _, experiment_dir = _run_experiment(tmpdir)

        trainer = Trainer(experiment_dir)

        assert trainer._manifest is not None
        assert trainer._params is not None
        assert len(trainer._round_data) == 3


def test_trainer_train_returns_sensors():

    with TemporaryDirectory() as tmpdir:
        uel, experiment_dir = _run_experiment(tmpdir)

        trainer = Trainer(experiment_dir, data=uel.data)

        permutation_ids = list(trainer._round_data.keys())
        sensors = trainer.train(permutation_ids[:2])

        assert len(sensors) == 2
        for sensor in sensors:
            assert sensor.round_params is not None
            assert sensor.metadata is not None
            assert sensor.results is not None
            assert sensor.model is None


def test_trainer_invalid_permutation_id():

    with TemporaryDirectory() as tmpdir:
        _, experiment_dir = _run_experiment(tmpdir)

        trainer = Trainer(experiment_dir)

        try:
            trainer.train([99999])
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert '99999' in str(e)


def test_sensor_no_model_raises():

    with TemporaryDirectory() as tmpdir:
        uel, experiment_dir = _run_experiment(tmpdir)

        trainer = Trainer(experiment_dir, data=uel.data)

        permutation_ids = list(trainer._round_data.keys())
        sensors = trainer.train(permutation_ids[:1])

        try:
            sensors[0]({'x_test': [], 'y_test': []})
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert 'no trained model' in str(e).lower()

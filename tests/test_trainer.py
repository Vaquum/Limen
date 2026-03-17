import json
from pathlib import Path
from tempfile import TemporaryDirectory

from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.trainer import Trainer
from limen.sfd.foundational_sfd import logreg_binary as logreg_sfd
from limen.sfd.foundational_sfd import random_binary as random_sfd
from tests.stubs.stubs import StubStrategy


def test_trainer_end_to_end():

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
            experiment_name='test_trainer',
            n_permutations=3,
        )

        # metadata.json written with correct fields
        metadata_path = experiment_dir / 'metadata.json'
        assert metadata_path.exists()
        with metadata_path.open('r') as f:
            metadata = json.load(f)
        assert metadata['sfd_module'] == 'limen.sfd.foundational_sfd.random_binary'
        assert 'limen_version' in metadata
        assert 'created_at' in metadata

        # Trainer loads manifest, params, round data, and original log
        trainer = Trainer(experiment_dir, data=uel.data)
        assert trainer._manifest is not None
        assert trainer._params is not None
        assert len(trainer._round_data) == 3
        assert trainer._original_log is not None
        assert len(trainer._original_log) == 3

        # train returns Sensor instances with results
        permutation_ids = list(trainer._round_data.keys())
        sensors = trainer.train(permutation_ids[:2])
        assert len(sensors) == 2
        for sensor in sensors:
            assert sensor.round_params is not None
            assert sensor.metadata is not None
            assert sensor.results is not None
            assert sensor.model is None

        # _validate_metrics detects mismatches on stochastic SFD
        mismatches = trainer._validate_metrics(
            permutation_ids[0], sensors[0].results,
        )
        assert isinstance(mismatches, list)

        # invalid permutation ID raises ValueError
        try:
            trainer.train([99999])
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert '99999' in str(e)

        # Pass 1 sensor with no model raises on call
        try:
            sensors[0]({'x_test': [], 'y_test': []})
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert 'no trained model' in str(e).lower()


def test_trainer_deterministic_validation():

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

        for pid, sensor in zip(permutation_ids, sensors, strict=True):
            mismatches = trainer._validate_metrics(pid, sensor.results)
            assert mismatches == [], f"Permutation {pid} had mismatches: {mismatches}"

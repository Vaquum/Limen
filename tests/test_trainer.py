import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import polars as pl
import pytest

import limen.experiment.trainer.trainer as trainer_module
from limen.experiment.trainer import Trainer


def test_trainer_requires_yaml_reference() -> None:

    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir)
        (experiment_dir / 'metadata.json').write_text(
            json.dumps({'sfd_module': 'limen.sfd.foundational_sfd.logreg_binary'}),
        )
        (experiment_dir / 'round_data.jsonl').write_text('')

        with pytest.raises(ValueError, match='yaml_reference'):
            Trainer(experiment_dir, data=pl.DataFrame({'x': [1, 2, 3]}))


def test_trainer_reconstructs_yaml_artifact_from_metadata_reference(monkeypatch) -> None:

    seen = {}

    class FakeCompiledSFD:

        def __init__(self, yaml_reference):
            seen['yaml_reference'] = yaml_reference

        def params(self):
            return {'alpha': [1]}

        def manifest(self):
            return SimpleNamespace(architecture_function=None)

    monkeypatch.setattr(trainer_module, 'CompiledSFD', FakeCompiledSFD)

    yaml_reference = {
        'metadata': {'name': 'yaml_exp'},
        'sfd': {'params': {'alpha': [1]}},
    }

    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir)
        (experiment_dir / 'metadata.json').write_text(
            json.dumps({'yaml_reference': yaml_reference}),
        )
        (experiment_dir / 'round_data.jsonl').write_text('')

        trainer = Trainer(
            experiment_dir,
            data=pl.DataFrame({'x': [1, 2, 3]}),
        )

    assert seen['yaml_reference'] == yaml_reference
    assert trainer._param_keys == frozenset({'alpha'})


@pytest.mark.parametrize(
    'yaml_reference',
    [
        [],
        {'metadata': {'name': 'yaml_exp'}, 'sfd': {}},
    ],
)
def test_trainer_rejects_malformed_yaml_reference(yaml_reference) -> None:

    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir)
        (experiment_dir / 'metadata.json').write_text(
            json.dumps({'yaml_reference': yaml_reference}),
        )
        (experiment_dir / 'round_data.jsonl').write_text('')

        with pytest.raises(ValueError, match='yaml_reference'):
            Trainer(
                experiment_dir,
                data=pl.DataFrame({'x': [1, 2, 3]}),
            )


def test_trainer_wraps_yaml_resolution_error(monkeypatch) -> None:

    class FakeCompiledSFD:

        def __init__(self, _yaml_reference):
            pass

        def params(self):
            return {'alpha': [1]}

        def manifest(self):
            raise trainer_module.ResolutionError(
                'bad.reference',
                ['limen'],
            )

    monkeypatch.setattr(trainer_module, 'CompiledSFD', FakeCompiledSFD)

    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir)
        (experiment_dir / 'metadata.json').write_text(
            json.dumps({
                'yaml_reference': {
                    'metadata': {'name': 'yaml_exp'},
                    'sfd': {'params': {'alpha': [1]}},
                },
            }),
        )
        (experiment_dir / 'round_data.jsonl').write_text('')

        with pytest.raises(ValueError, match='yaml_reference'):
            Trainer(
                experiment_dir,
                data=pl.DataFrame({'x': [1, 2, 3]}),
            )


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

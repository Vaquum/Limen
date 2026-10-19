from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from limen.experiment._parallel_encoding import decode, digest, encode
from limen.experiment._parallel_plan import ScalerFactory, freeze_manifest, validate_domain
from limen.experiment._parallel_resources import sklearn_stage, validate_resources, xgboost_stage
from limen.yaml import parse
from limen.yaml.compiler import build_manifest


def _configuration() -> dict[str, object]:
    config, errors = parse(Path('limen/yaml/templates/logreg_binary.yaml').read_text())
    assert not errors
    return config


def test_frozen_manifest_round_plan() -> None:
    config = _configuration()
    original = build_manifest(config)
    plan = freeze_manifest(original, config)
    assert plan.pipeline_digest == freeze_manifest(build_manifest(config), config).pipeline_digest
    assert len(pickle.dumps(plan)) <= 65536
    transported = pickle.loads(pickle.dumps(plan))
    assert transported.pipeline_digest == plan.pipeline_digest
    assert any(isinstance(stage.component, ScalerFactory) for stage in transported.stages)
    original.split_config = (2, 1, 1)
    assert plan.manifest.split_config != original.split_config
    with pytest.raises(ValueError, match='unchanged compiled YAML'):
        freeze_manifest(original, config)
    values = {'tuple': (1, 0.5, None), 'list': [1, 0.5, None], 'bool': True}
    assert decode(json.loads(json.dumps(encode(values)))) == values
    assert digest(values['tuple']) != digest(values['list'])
    assert digest(values) == digest(dict(reversed(list(values.items()))))


def test_effective_resources_and_output_evidence() -> None:
    config = _configuration()
    plan = freeze_manifest(build_manifest(config), config)
    params = {'n_jobs': 1, 'random_state': 42, 'solver': 'liblinear', 'scaler_type': 'robust',
              'roc_period': 1, 'frac_diff_d': 0.0, 'q': 0.5, 'shift': -1,
              'cal_method': 'sigmoid', 'threshold_min': 0.25, 'threshold_max': 0.65, 'threshold_step': 0.05}
    validate_resources(plan, params)
    validate_domain(plan, {key: [value] for key, value in params.items()}, None)
    for jobs in (None, True, 0, -1, 2):
        with pytest.raises(ValueError, match='effective n_jobs=1'):
            validate_resources(plan, params | {'n_jobs': jobs})
    for seed in (None, True, '42'):
        with pytest.raises(ValueError, match='fixed integer seed'):
            validate_resources(plan, params | {'random_state': seed})
    with pytest.raises(ValueError, match='fixed integer seed'):
        validate_domain(plan, {'n_jobs': [1], 'solver': ['lbfgs', 'saga'], 'random_state': [None],
                               **{key: [value] for key, value in params.items() if key not in ('n_jobs', 'solver', 'random_state')}}, None)
    sklearn_stage({'solver': 'lbfgs', 'random_state': None})
    for option in ({'device': 'cuda'}, {'tree_method': 'gpu_hist'}, {'sampling_method': 'gradient_based'}):
        with pytest.raises(ValueError, match='CPU backend required'):
            xgboost_stage({'n_jobs': 1, 'random_state': 42} | option)
    for value in (float('nan'), float('inf')):
        with pytest.raises(ValueError, match='nonfinite'):
            encode(value)
    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match='cycles'):
        encode(cycle)

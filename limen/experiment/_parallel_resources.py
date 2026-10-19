from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from limen._parallel_contract import ParallelRoundContract
from limen.experiment._parallel_encoding import reference
from limen.experiment._parallel_plan import FrozenManifestPlan, configurations


def deterministic_stage(params: Mapping[str, object]) -> None:
    for key in ('n_jobs', 'nthread', 'num_threads'):
        if key in params and (type(params[key]) is not int or params[key] != 1):
            raise ValueError('effective n_jobs=1 required')
    for key in ('device', 'device_type'):
        if key in params and params[key] not in (None, 'cpu'):
            raise ValueError('CPU backend required')


def seeded_stage(params: Mapping[str, object]) -> None:
    deterministic_stage(params)
    if type(params.get('random_state')) is not int:
        raise ValueError('fixed integer seed required for random_state')


def sklearn_stage(params: Mapping[str, object]) -> None:
    deterministic_stage(params)
    if params.get('solver') in ('liblinear', 'sag', 'saga'):
        seeded_stage(params)


def lightgbm_stage(params: Mapping[str, object]) -> None:
    seeded_stage(params)


def xgboost_stage(params: Mapping[str, object]) -> None:
    seeded_stage(params)
    if (params.get('tree_method') not in (None, 'auto', 'exact', 'approx', 'hist')
            or params.get('grow_policy') not in (None, 'depthwise', 'lossguide')
            or params.get('sampling_method') not in (None, 'uniform')
            or params.get('multi_strategy') not in (None, 'one_output_per_tree', 'multi_output_tree')
            or params.get('booster', 'gbtree') not in ('gbtree', 'dart', 'gblinear')):
        raise ValueError('CPU backend required')
    trees = params.get('num_parallel_tree')
    if trees is not None and (type(trees) is not int or cast(int, trees) < 1):
        raise ValueError('positive integer num_parallel_tree required')


def composed_stage(params: Mapping[str, object]) -> None:
    deterministic_stage(params)
    seed = params.get('seed')
    if type(seed) is not int:
        raise ValueError('fixed integer seed required for seed')
    for role in ('direction', 'sizing'):
        factory = params[f'{role}_factory']
        contract = getattr(factory, '__parallel_round_contract__', None)
        if not isinstance(contract, ParallelRoundContract):
            raise ValueError(f'{role}_factory: missing independence contract')
        _ = reference(factory)
        config = cast(Mapping[str, object], params.get(f'{role}_params') or {})
        contract.validate(dict(config) | {'seed': seed, 'random_state': seed})


def unfrozen_stage(params: Mapping[str, object]) -> None:
    raise ValueError('unfrozen external state')


def validate_resources(plan: FrozenManifestPlan, params: Mapping[str, object]) -> None:
    configs = configurations(plan.manifest, params)
    for stage in plan.stages:
        config = configs[stage.path]
        for key in stage.contract.parameter_keys:
            if key not in config:
                raise ValueError(f'Parallel stage {stage.path}: unresolved parameter {key}')
        try:
            stage.contract.validate(config)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Parallel stage {stage.path}: {exc}') from exc


def backend_imports(plan: FrozenManifestPlan) -> tuple[str, ...]:
    return tuple(sorted({name for stage in plan.stages for name in stage.contract.backend_imports}))

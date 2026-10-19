from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from string import Formatter

from limen.experiment.manifest_core import DataSourceConfig
from typing import cast

from limen.experiment._parallel_encoding import encode
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
    key = 'random_state' if 'random_state' in params else 'seed'
    if type(params.get(key)) is not int:
        raise ValueError(f'fixed integer seed required for {key}')


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
    if trees is not None and (type(trees) is not int or trees < 1):
        raise ValueError('positive integer num_parallel_tree required')


def composed_stage(params: Mapping[str, object]) -> None:
    deterministic_stage(params)
    seed = params.get('seed')
    if type(seed) is not int:
        raise ValueError('fixed integer seed required for seed')


def unfrozen_stage(_params: Mapping[str, object]) -> None:
    raise ValueError('unfrozen external state')


def validate_resources(plan: FrozenManifestPlan, params: Mapping[str, object]) -> None:
    configs = configurations(plan.manifest, params)
    for stage in plan.stages:
        if stage.path not in configs:
            continue  # The native Manifest skips this conditional stage for this candidate.
        if stage.path.startswith('architecture.') and configs['architecture'].get(stage.path.split('.', 1)[1]) is not stage.component:
            raise ValueError(f'Parallel stage {stage.path}: frozen independence contract differs')
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


def _static_source(value: object, domain: Mapping[str, list[object]]) -> None:
    if callable(value):
        raise ValueError('Parallel rounds require static source parameters')
    if isinstance(value, str):
        if value in domain or any(field is not None for _, field, _, _ in Formatter().parse(value)):
            raise ValueError('Parallel rounds reject parameter-dependent source declarations')
    elif isinstance(value, Mapping):
        for item in cast(Mapping[object, object], value).values():
            _static_source(item, domain)
    elif isinstance(value, (tuple, list)):
        for item in cast(tuple[object, ...] | list[object], value):
            _static_source(item, domain)


def validate_domain(plan: FrozenManifestPlan, domain: Mapping[str, list[object]], context: Mapping[str, object] | None) -> None:
    effective = dict(domain) | {key: [value] for key, value in (context or {}).items()}
    if any(not values for values in effective.values()):
        raise ValueError('Parallel rounds require nonempty parameter domains')
    _ = encode(effective)
    config = plan.manifest.backtest_config
    sources: list[object] = [plan.manifest.data_source_config, plan.manifest.test_data_source_config]
    if config is not None:
        sources.extend((config.execution_data_source, config.funding.data_source if config.funding is not None else None))
    for source in sources:
        if source is not None:
            if not isinstance(source, DataSourceConfig):
                raise ValueError('Parallel rounds require reconstructable declared sources')
            _static_source(source.params, effective)
    for candidate in product(*effective.values()):
        validate_resources(plan, dict(zip(effective, candidate, strict=True)))

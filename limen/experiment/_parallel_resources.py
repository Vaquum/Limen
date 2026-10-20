from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from inspect import Parameter, signature
from string import Formatter

from limen.experiment.manifest_core import DataSourceConfig, Manifest, MLManifest
from typing import cast

from limen.experiment._parallel_encoding import digest, encode
from limen.experiment._parallel_plan import FrozenManifestPlan, FrozenStage, ScalerFactory, configurations


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
    configs = _configurations(plan, params)
    for stage in plan.stages:
        _validate_stage(stage, configs)



def _same_component(left: object, right: object) -> bool:
    return left is right or digest(left) == digest(right)


def _configurations(plan: FrozenManifestPlan, params: Mapping[str, object]) -> dict[str, dict[str, object]]:
    configs = configurations(plan.manifest, params)
    for key, component in configs['architecture'].items():
        if key.endswith('_factory') and callable(component):
            path = f'architecture.{key}'
            if not any(stage.path == path and _same_component(stage.component, component) for stage in plan.stages):
                raise ValueError(f'Parallel stage {path}: missing frozen independence contract')
    return configs


def _validate_stage(stage: FrozenStage, configs: Mapping[str, dict[str, object]]) -> None:
    if stage.path not in configs:
        return  # The native Manifest skips this conditional stage for this candidate.
    if stage.path.startswith('architecture.') and not _same_component(configs['architecture'].get(stage.path.split('.', 1)[1]), stage.component):
        return  # A different frozen factory variant is selected for this candidate.
    config = configs[stage.path]
    for key in stage.contract.parameter_keys:
        if key not in config:
            raise ValueError(f'Parallel stage {stage.path}: unresolved parameter {key}')
    try:
        stage.contract.validate(config)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Parallel stage {stage.path}: {exc}') from exc


def _references(value: object, domain: Mapping[str, list[object]]) -> set[str]:
    if isinstance(value, str):
        result: set[str] = {value} if value in domain else set()
        if '{' in value and '}' in value:
            for _, field, spec, _ in Formatter().parse(value):
                if field is not None:
                    result.add(field.split('.', 1)[0].split('[', 1)[0])
                result.update(_references(spec, domain))
        return result.intersection(domain)
    if isinstance(value, Mapping):
        return set[str]().union(*(_references(item, domain) for item in cast(Mapping[object, object], value).values()))
    if isinstance(value, (tuple, list)):
        return set[str]().union(*(_references(item, domain) for item in cast(tuple[object, ...] | list[object], value)))
    return set()


def _bindings(manifest: Manifest, path: str, domain: Mapping[str, list[object]]) -> tuple[Mapping[str, object], tuple[str, ...]]:
    if path == 'architecture':
        function = manifest.architecture_function
        assert function is not None
        parameters = signature(function).parameters
        names_to_bind = domain if any(value.kind == Parameter.VAR_KEYWORD for value in parameters.values()) else parameters
        return {key: key for key in names_to_bind if key != 'data'}, ()
    if path.startswith('architecture.'):
        factory = path.split('.', 1)[1]
        key = factory.removesuffix('_factory') + '_params'
        function = manifest.architecture_function
        assert function is not None
        parameter = signature(function).parameters.get(key)
        default: object = None if parameter is None else parameter.default
        variants: list[Mapping[str, object]] = []
        for option in domain.get(key, [default]):
            if isinstance(option, Mapping):
                variants.append(cast(Mapping[str, object], option))
        names = {name for value in variants for name in value}
        arguments: dict[str, object] = {name: [value.get(name) for value in variants] for name in names}
        seeded = 'seed' in signature(function).parameters or any(value.kind == Parameter.VAR_KEYWORD for value in signature(function).parameters.values())
        return arguments | ({'seed': 'seed'} if seeded else {}), (key, factory)
    if path.startswith('feature_transforms.'):
        entry = manifest.feature_transforms[int(path.split('.')[1])]
        return dict(entry.params), tuple(key for key in (entry.include_if, 'feature_groups' if entry.group is not None else None) if key is not None)
    if path.startswith('target') and manifest.target_class_config is not None:
        target = manifest.target_class_config
        return dict(target.fit_params) | {'transform_params': target.transform_params} | (dict(target.transform_params) if path == 'target.transform' else {}), ()
    if path in ('pre_split_data_selector', 'bar_formation'):
        step = getattr(manifest, path)
        return dict(step[1]), ('bar_type',) if path == 'bar_formation' else ()
    return _ml_bindings(manifest, path, domain) if isinstance(manifest, MLManifest) else ({}, ())


def _ml_bindings(manifest: MLManifest, path: str, domain: Mapping[str, list[object]]) -> tuple[Mapping[str, object], tuple[str, ...]]:
    if path.startswith('scaler.') and manifest.scaler is not None:
        return _scaler_bindings(manifest, path)
    if path in ('calibration', 'threshold') and manifest.prediction_calibration_config is not None:
        return dict(getattr(manifest.prediction_calibration_config, f'{path}_params')), ('use_calibration', 'use_threshold')
    if path == 'ablation' and manifest.ablation_config is not None:
        return {'random_state': manifest.ablation_config.seed_key}, ()
    if path in ('data_dict_extension', 'feature_shims'):
        return {'round_params': {key: key for key in domain}}, ()
    return {}, ()


def _scaler_bindings(manifest: MLManifest, path: str) -> tuple[Mapping[str, object], tuple[str, ...]]:
    assert manifest.scaler is not None
    if path == 'scaler.transform':
        return dict(manifest.scaler[2]), ()
    _, function, arguments = manifest.scaler[0][int(path.split('.')[2])]
    if isinstance(function, ScalerFactory):
        return function.static_params | dict(arguments), (function.dynamic_selector,) if function.dynamic_selector is not None else ()
    return dict(arguments), ()


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
    baseline = {key: values[0] for key, values in effective.items()}
    baseline_configs = _configurations(plan, baseline)
    for stage in plan.stages:
        keys = stage.contract.parameter_keys if stage.contract.validation_keys is None else stage.contract.validation_keys
        arguments, selectors = _bindings(plan.manifest, stage.path, effective)
        inputs = set(keys) | set(stage.contract.parameter_keys)
        dependencies = set(selectors)
        for key in inputs:
            dependencies.update(_references(arguments.get(key), effective))
        projected = {key: values for key, values in effective.items() if key in dependencies}
        for index, values in enumerate(product(*projected.values())):
            candidate = baseline | dict(zip(projected, values, strict=True))
            _validate_stage(stage, baseline_configs if index == 0 else _configurations(plan, candidate))

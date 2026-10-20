from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from itertools import product
from inspect import Parameter, signature, isfunction, Signature
from dataclasses import dataclass
from functools import partial
from importlib import import_module

import limen.experiment.manifest_core as _manifest
from limen.experiment.manifest_core import DataSourceConfig, Manifest, MLManifest
from typing import Protocol, cast

from limen.experiment._parallel_encoding import encode
from limen.experiment._parallel_state import component_state, object_state, restore_object
from limen.scalers.registry import SCALER_REGISTRY
from limen.experiment._parallel_plan import FrozenManifestPlan, FrozenStage, ScalerFactory, component_digest, static_source, parameter_references, factory_reference, factory_allocator

_resolve_params = cast(Callable[[Mapping[str, object], Mapping[str, object]], dict[str, object]], vars(import_module('limen.experiment.manifest_core'))['_resolve_params'])


class _TargetClass(Protocol):
    transform: Callable[..., object]


@dataclass
class FrozenClassFactory:
    """Avoid Python 3.10 dataclass signature inference on the instance-only property."""

    factory_class: type
    initializer: Callable[..., object] | None
    allocator: Callable[..., object] | None

    @property
    def __signature__(self) -> Signature:
        result = signature(self.initializer or self.allocator or self.factory_class)
        return result.replace(parameters=list(result.parameters.values())[1:]) if self.initializer is not None or self.allocator is not None else result

    @property
    def __parallel_round_contract__(self) -> object:
        return getattr(self.factory_class, '__parallel_round_contract__', None)

    def __call__(self, *args: object, **kwargs: object) -> object:
        allocate = self.allocator or self.factory_class.__new__
        instance = allocate(self.factory_class, *(() if allocate is object.__new__ else args), **({} if allocate is object.__new__ else kwargs))
        if isinstance(instance, self.factory_class):
            initialize = self.initializer or self.factory_class.__init__
            if initialize(instance, *args, **kwargs) is not None:
                raise TypeError('Parallel factory initializer must return None')
        return instance


def _factory_component(value: object) -> object:
    if isinstance(value, type):
        if type(value).__call__ is not type.__call__:
            raise TypeError('Parallel transport rejects unfrozen factory metaclass state')
        return FrozenClassFactory(value, value.__init__ if isfunction(value.__init__) else None, value.__new__ if isfunction(value.__new__) else None)
    if isinstance(value, partial):
        attributes, slots, bound, native = object_state(cast(partial[object], value))
        assert bound is not None
        result = restore_object(type(cast(object, value)), (attributes, slots, (_factory_component(bound[0]), *bound[1:]), native))
        vars(result)['__parallel_round_contract__'] = getattr(cast(object, value), '__parallel_round_contract__', None)
        return result
    return value


def factory_bindings(manifest: Manifest, options: Mapping[str, list[object]] | None) -> dict[str, list[object]]:
    function = manifest.architecture_function
    if function is None:
        return {}
    defaults = {name: [item.default] for name, item in signature(function).parameters.items() if name.endswith('_factory')}
    result: dict[str, list[object]] = {}
    for name, values in (defaults | dict(options or {})).items():
        for value in values:
            if name.endswith('_factory') and callable(value) and value is not Parameter.empty:
                source, imported = factory_reference(cast(object, value))
                result.setdefault(name, []).append((source, component_digest(cast(object, value)), imported, _factory_component(cast(object, value))))
    return result


def bind_factories(inventory: Mapping[str, list[object]], kwargs: Mapping[str, object]) -> dict[str, object]:
    result = dict(kwargs)
    for name, variants in inventory.items():
        value = kwargs.get(name)
        if callable(value):
            source, imported = factory_reference(cast(object, value))
            fingerprint = component_digest(cast(object, value))
            choices = cast(list[tuple[list[object], str, bool, object]], variants)
            exact = [component for _, key, _, component in choices if key == fingerprint or component_digest(component) == fingerprint]
            matches = [component for identity, _, live, component in choices if identity == source and live and imported]
            candidates = {component_digest(component): component for component in exact or matches}
            if len(candidates) != 1:
                raise ValueError(f'Parallel stage architecture.{name}: missing frozen independence contract or ambiguous factory selection')
            result[name] = next(iter(candidates.values()))
    return result


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
    return left is right or component_digest(left) == component_digest(right)


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
    if stage.path.startswith('architecture.') and not _same_component(configs['architecture'].get(stage.path.split('.')[1]), stage.component):
        return  # A different frozen factory variant is selected for this candidate.
    config = configs[stage.path]
    for key in stage.contract.parameter_keys:
        if key not in config:
            raise ValueError(f'Parallel stage {stage.path}: unresolved parameter {key}')
    try:
        stage.contract.validate(config)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Parallel stage {stage.path}: {exc}') from exc


def _bindings(manifest: Manifest, path: str, domain: Mapping[str, list[object]]) -> tuple[Mapping[str, object], tuple[str, ...]]:
    if path == 'architecture':
        function = manifest.architecture_function
        assert function is not None
        parameters = signature(function).parameters
        names_to_bind = domain if any(value.kind == Parameter.VAR_KEYWORD for value in parameters.values()) else parameters
        return {key: key for key in names_to_bind if key != 'data'}, ()
    if path.startswith('architecture.'):
        factory = path.split('.')[1]
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
            static_source(source.params, effective)
    baseline = {key: values[0] for key, values in effective.items()}
    baseline_configs = _configurations(plan, baseline)
    for stage in plan.stages:
        keys = stage.contract.parameter_keys if stage.contract.validation_keys is None else stage.contract.validation_keys
        arguments, selectors = _bindings(plan.manifest, stage.path, effective)
        inputs = set(keys) | set(stage.contract.parameter_keys)
        dependencies = set(selectors)
        for key in inputs:
            dependencies.update(parameter_references(arguments.get(key), effective))
        projected = {key: values for key, values in effective.items() if key in dependencies}
        for index, values in enumerate(product(*projected.values())):
            candidate = baseline | dict(zip(projected, values, strict=True))
            _validate_stage(stage, baseline_configs if index == 0 else _configurations(plan, candidate))


def _defaults(component: object, params: Mapping[str, object], *, skip: int = 1, path: str = 'architecture', runtime: tuple[str, ...] = ()) -> dict[str, object]:
    signature = inspect.signature(cast(Callable[..., object], component))
    owner = component.__self__ if inspect.ismethod(component) else component
    state = component_state(owner) if not (inspect.isfunction(owner) or isinstance(owner, type)) else {}
    result = {key: value for key, value in state.items() if key != '__parallel_round_contract__'} | dict(params)
    for i, (key, value) in enumerate(signature.parameters.items()):
        if i < skip or value.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL) or key in runtime:
            continue
        if key not in params:
            if value.default is inspect.Parameter.empty:
                raise ValueError(f'Parallel stage {path}: unresolved parameter {key}')
            result[key] = value.default
    return result


def _configured(path: str, component: object, arguments: Mapping[str, object], params: Mapping[str, object], *, skip: int = 1, runtime: tuple[str, ...] = ()) -> dict[str, object]:
    try:
        return _defaults(component, _resolve_params(arguments, params), skip=skip, path=path, runtime=runtime)
    except KeyError as exc:
        raise ValueError(f'Parallel stage {path}: unresolved parameter {exc.args[0]}') from exc


def configurations(manifest: Manifest, params: Mapping[str, object]) -> dict[str, dict[str, object]]:
    _ = _defaults(manifest.architecture_function, params)
    resolved = cast(dict[str, object], manifest.resolve_model_kwargs(dict(params)))
    from limen.experiment._prepare_trade_context import resolve_component_kwargs
    resolve_component_kwargs(manifest.architecture_function, {}, resolved, params)
    result = {'architecture': resolved}
    for key, factory in resolved.items():
        if key.endswith('_factory') and callable(factory):
            arguments = dict(cast(Mapping[str, object], resolved.get(key.removesuffix('_factory') + '_params') or {}))
            if 'seed' in resolved:
                arguments['seed'] = resolved['seed']
            result[f'architecture.{key}'] = _defaults(factory, arguments, skip=0)
            if (allocator := factory_allocator(factory)) is not None:
                result[f'architecture.{key}.allocate'] = _defaults(allocator, arguments, skip=0)
    for path in ('pre_split_data_selector', 'bar_formation'):
        step = getattr(manifest, path)
        if step is not None and (path != 'bar_formation' or params.get('bar_type', 'base') != 'base'):
            result[path] = _configured(path, step[0], step[1], params)
    included = cast(Callable[[object, Mapping[str, object]], bool], vars(_manifest)['_should_include_transform'])
    result.update({f'feature_transforms.{i}': _configured(f'feature_transforms.{i}', entry.func, entry.params, params)
                   for i, entry in enumerate(manifest.feature_transforms) if included(entry, params)})
    target = manifest.target_class_config
    if target is not None:
        runtime = ('trade_context',) if getattr(target.target_class, 'requires_trade_context', False) else ()
        if any(key in arguments for arguments in (target.fit_params, target.transform_params) for key in runtime):
            raise ValueError('Parallel stage target: trade outcome context is reserved to the framework')
        initializer = getattr(target, '_parallel_init', None)
        result['target'] = _defaults(target.target_class if initializer is None else initializer, _resolve_params(target.fit_params, params), skip=2 if initializer is None else 3, path='target', runtime=runtime)
        transform_config = _defaults(getattr(target, '_parallel_transform', None) or cast(_TargetClass, target.target_class).transform, _resolve_params(target.transform_params, params), skip=2, path='target.transform', runtime=runtime)
        result['target']['transform_params'] = transform_config
        result.update({'target.fit': result['target'], 'target.transform': result['target'] | transform_config})
        if (allocator := getattr(target, '_parallel_new', None)) is not None:
            result['target.allocate'] = _defaults(allocator, _resolve_params(target.fit_params, params), skip=3, runtime=runtime)
    if isinstance(manifest, MLManifest):
        if manifest.scaler is not None:
            computations, transform, arguments = manifest.scaler
            result['scaler.transform'] = _configured('scaler.transform', transform, arguments, params)
            for i, (_, function, arguments) in enumerate(computations):
                config = _resolve_params(arguments, dict(params))
                result[f'scaler.fit.{i}'] = _defaults(function, config)
                if isinstance(function, ScalerFactory):
                    selected = function.selected(config)
                    classes = (function.transform_class,) if function.dynamic_selector is None else tuple(SCALER_REGISTRY.values())
                    for j, cls in enumerate(classes):
                        result[f'scaler.class.{i}.{j}'] = _defaults(cls, function.static_params | config) if cls is selected else {}
        if manifest.data_dict_extension is not None:
            result['data_dict_extension'] = _defaults(manifest.data_dict_extension, {'round_params': dict(params)}, skip=0, runtime=('data_dict', 'split_data', 'fitted_params'))
        if manifest.ablation_config is not None:
            seed = params.get(manifest.ablation_config.seed_key)
            result['ablation'] = {'random_state': 0 if seed is None else seed}
        resolve_shims = cast(Callable[[object, Mapping[str, object]], object], vars(import_module('limen.experiment._column_shims'))['_resolved'])
        result.update(pca={'svd_solver': 'full'}, feature_shims={'specs': resolve_shims(manifest.feature_shims, params), 'round_params': dict(params)})
        calibration = manifest.prediction_calibration_config
        if calibration is not None and (params.get('use_calibration', True) or params.get('use_threshold', True)):
            config = calibration.resolve(dict(params))
            for name in ('calibration', 'threshold'):
                function = getattr(config, f'{name}_func')
                if function is not None and params.get(f'use_{name}', True):
                    result[name] = _defaults(function, getattr(config, f'{name}_params'), skip=3 if name == 'calibration' else 2, path=name)
    return result

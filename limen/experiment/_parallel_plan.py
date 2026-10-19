from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from itertools import product
from string import Formatter
from typing import cast

import polars as pl

from limen._parallel_contract import ParallelRoundContract, attach
from limen.experiment._parallel_encoding import digest, encode, module_identity, reference
from limen.experiment.manifest_core import DataSourceConfig, Manifest, MLManifest, _resolve_params
from limen.scalers.registry import SCALER_REGISTRY


@dataclass(frozen=True)
class ScalerFactory:
    transform_class: object
    static_params: dict[str, object]
    dynamic_selector: str | None

    def selected(self, params: Mapping[str, object]) -> object:
        return self.transform_class if self.dynamic_selector is None else SCALER_REGISTRY[cast(str, params['scaler_type'])]

    def __call__(self, data: pl.DataFrame, **params: object) -> object:
        function = cast(Callable[..., object], self.selected(params))
        return function(data, **self.static_params, **{key: value for key, value in params.items() if key != 'scaler_type'})


attach(ScalerFactory)


@dataclass(frozen=True)
class FrozenStage:
    path: str
    component: object
    contract: ParallelRoundContract


@dataclass(frozen=True)
class FrozenManifestPlan:
    manifest: Manifest
    stages: tuple[FrozenStage, ...]
    pipeline_digest: str
    resource_digest: str


def _normalize(value: object) -> object:
    if inspect.isfunction(value) and value.__module__ == 'limen.experiment.manifest_core':
        if value.__qualname__ == 'make_fitted_scaler.<locals>._factory':
            cls, params = cast(tuple[object, dict[str, object]], value.__defaults__)
            return ScalerFactory(cls, copy.deepcopy(params), None)
        if value.__qualname__ == 'MLManifest.set_scaler_from_params.<locals>._scaler_factory':
            closure = {name: cell.cell_contents for name, cell in zip(value.__code__.co_freevars, value.__closure__ or (), strict=True)}
            return ScalerFactory(None, cast(dict[str, object], copy.deepcopy(closure['_static'])), cast(str, closure['param_name']))
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in cast(dict[str, object], value).items()}
    if isinstance(value, (list, tuple)):
        items = [_normalize(item) for item in cast(list[object] | tuple[object, ...], value)]
        return tuple(items) if isinstance(value, tuple) else items
    if is_dataclass(value) and not isinstance(value, type):
        clone = copy.deepcopy(value)
        for field in fields(value):
            object.__setattr__(clone, field.name, _normalize(getattr(value, field.name)))
        return clone
    return copy.deepcopy(value)


def _contract(component: object, path: str) -> ParallelRoundContract:
    contract = getattr(component, '__parallel_round_contract__', None)
    if not isinstance(contract, ParallelRoundContract):
        raise ValueError(f'Parallel stage {path}: missing independence contract')
    _ = reference(contract.validate)
    return contract


def components(manifest: Manifest) -> list[tuple[str, object]]:
    result: list[tuple[str, object]] = [('architecture', manifest.architecture_function)]
    for path in ('pre_split_data_selector', 'bar_formation'):
        step = getattr(manifest, path)
        if step is not None:
            result.append((path, step[0]))
    result.extend((f'feature_transforms.{i}', entry.func) for i, entry in enumerate(manifest.feature_transforms))
    if manifest.target_class_config is not None:
        result.append(('target', manifest.target_class_config.target_class))
    if isinstance(manifest, MLManifest):
        if manifest.scaler is not None:
            computations, transform, _ = manifest.scaler
            result.extend((f'scaler.fit.{i}', compute) for i, (_, compute, _) in enumerate(computations))
            result.append(('scaler.transform', transform))
            for i, (_, compute, _) in enumerate(computations):
                if isinstance(compute, ScalerFactory):
                    classes = (compute.transform_class,) if compute.dynamic_selector is None else tuple(SCALER_REGISTRY.values())
                    result.extend((f'scaler.class.{i}.{j}', cls) for j, cls in enumerate(classes))
        if manifest.data_dict_extension is not None:
            result.append(('data_dict_extension', manifest.data_dict_extension))
        from limen.experiment.manifest_core import _apply_feature_ablation, _pca_transformer
        from limen.experiment._column_shims import apply_column_shims
        if manifest.ablation_config is not None:
            result.append(('ablation', _apply_feature_ablation))
        if manifest.pca_compression_config is not None:
            result.append(('pca', _pca_transformer))
        if manifest.feature_shims:
            result.append(('feature_shims', apply_column_shims))
        calibration = manifest.prediction_calibration_config
        if calibration is not None:
            for name in ('calibration', 'threshold'):
                function = getattr(calibration, f'{name}_func')
                if function is not None:
                    result.append((name, function))
    return result


def freeze_manifest(manifest: Manifest, yaml_reference: Mapping[str, object] | None) -> FrozenManifestPlan:
    from limen.yaml.compiler import build_manifest

    frozen = cast(Manifest, _normalize(manifest))
    if yaml_reference is not None and digest(frozen) != digest(_normalize(build_manifest(dict(yaml_reference)))):
        raise ValueError('Parallel rounds require the unchanged compiled YAML manifest pipeline')
    stages = tuple(FrozenStage(path, component, _contract(component, path)) for path, component in components(frozen))
    # Force transport validation even when a stage validator consumes no parameters.
    _ = encode(frozen)
    imports = sorted({name for stage in stages for name in stage.contract.backend_imports})
    resources = ([stage.contract for stage in stages], [module_identity(name) for name in imports],
                 module_identity('limen.experiment._parallel_resources'), {'native_threads': 1, 'startup_threads': 1})
    return FrozenManifestPlan(frozen, stages, digest(frozen), digest(resources))


def _defaults(component: object, params: Mapping[str, object], *, skip: int = 1) -> dict[str, object]:
    signature = inspect.signature(cast(Callable[..., object], component))
    result = dict(params)
    for i, (key, value) in enumerate(signature.parameters.items()):
        if i < skip or value.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
            continue
        if key not in result and value.default is not inspect.Parameter.empty:
            result[key] = value.default
    return result


def configurations(manifest: Manifest, params: Mapping[str, object]) -> dict[str, dict[str, object]]:
    resolved = cast(dict[str, object], manifest.resolve_model_kwargs(dict(params)))
    from limen.experiment._prepare_trade_context import resolve_component_kwargs
    resolve_component_kwargs(manifest.architecture_function, {}, resolved, params)
    result = {'architecture': resolved}
    for path in ('pre_split_data_selector', 'bar_formation'):
        step = getattr(manifest, path)
        if step is not None:
            result[path] = _defaults(step[0], _resolve_params(step[1], dict(params)))
    result.update({f'feature_transforms.{i}': _defaults(entry.func, _resolve_params(entry.params, dict(params)))
                   for i, entry in enumerate(manifest.feature_transforms)})
    target = manifest.target_class_config
    if target is not None:
        result['target'] = _defaults(target.target_class, _resolve_params(target.fit_params, dict(params)), skip=2)
        result['target']['transform_params'] = _resolve_params(target.transform_params, dict(params))
    if isinstance(manifest, MLManifest):
        if manifest.scaler is not None:
            computations, _, arguments = manifest.scaler
            result['scaler.transform'] = _resolve_params(arguments, dict(params))
            for i, (_, function, arguments) in enumerate(computations):
                config = _resolve_params(arguments, dict(params))
                result[f'scaler.fit.{i}'] = _defaults(function, config)
                if isinstance(function, ScalerFactory):
                    selected = function.selected(config)
                    classes = (function.transform_class,) if function.dynamic_selector is None else tuple(SCALER_REGISTRY.values())
                    for j, cls in enumerate(classes):
                        result[f'scaler.class.{i}.{j}'] = _defaults(cls, function.static_params | config) if cls is selected else {}
        if manifest.data_dict_extension is not None:
            result['data_dict_extension'] = dict(params)
        if manifest.ablation_config is not None:
            result['ablation'] = {'random_state': params.get(manifest.ablation_config.seed_key, 0)}
        result.update(pca={'svd_solver': 'full'}, feature_shims=dict(params))
        calibration = manifest.prediction_calibration_config
        if calibration is not None:
            config = calibration.resolve(dict(params))
            for name in ('calibration', 'threshold'):
                function = getattr(config, f'{name}_func')
                if function is not None:
                    result[name] = _defaults(function, getattr(config, f'{name}_params'), skip=2)
    return result


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
    from limen.experiment._parallel_resources import validate_resources

    effective = dict(domain) | {key: [value] for key, value in (context or {}).items()}
    if any(not values for values in effective.values()):
        raise ValueError('Parallel rounds require nonempty parameter domains')
    _ = encode(effective)
    config = plan.manifest.backtest_config
    sources = [plan.manifest.data_source_config, plan.manifest.test_data_source_config]
    if config is not None:
        sources.extend((config.execution_data_source, config.funding.data_source if config.funding is not None else None))
    for source in sources:
        if source is not None:
            if not isinstance(source, DataSourceConfig):
                raise ValueError('Parallel rounds require reconstructable declared sources')
            _static_source(source.params, effective)
    for candidate in product(*effective.values()):
        validate_resources(plan, dict(zip(effective, candidate, strict=True)))

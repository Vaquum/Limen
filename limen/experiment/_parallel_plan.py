from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import cast

import polars as pl

from limen._parallel_contract import ParallelRoundContract, attach
from limen.experiment._parallel_encoding import digest, encode, module_identity, reference, source_digest
from limen.experiment.manifest_core import Manifest, MLManifest
import limen.experiment.manifest_core as _manifest

from limen.scalers.registry import SCALER_REGISTRY

_resolve_params = cast(Callable[[Mapping[str, object], Mapping[str, object]], dict[str, object]], vars(_manifest)['_resolve_params'])


@dataclass(frozen=True)
class ScalerFactory:
    transform_class: object
    static_params: dict[str, object]
    dynamic_selector: str | None

    def selected(self, params: Mapping[str, object]) -> object:
        return self.transform_class if self.dynamic_selector is None else SCALER_REGISTRY[cast(str, params['scaler_type'])]

    def __call__(self, data: pl.DataFrame, **params: object) -> object:
        function = cast(Callable[..., object], self.selected(params))
        extras = {key: value for key, value in params.items() if self.dynamic_selector is None or key != 'scaler_type'}
        return function(data, **self.static_params, **extras)


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


_PARALLEL_EVIDENCE_SINCE = (5, 45, 3)


def legacy_metadata(metadata: Mapping[str, object]) -> bool:
    version = metadata.get('limen_version')
    if not isinstance(version, str) or len(parts := version.split('.')) != len(_PARALLEL_EVIDENCE_SINCE) or not all(part.isdigit() for part in parts):
        raise ValueError('Cannot resume: missing or inconsistent parallel evidence version')
    return tuple(map(int, parts)) < _PARALLEL_EVIDENCE_SINCE and 'parallel_plan' not in metadata


def plan_identity(plan: FrozenManifestPlan, frames: tuple[pl.DataFrame | None, ...]) -> dict[str, object]:
    return {'version': 1, 'pipeline_digest': plan.pipeline_digest, 'resource_digest': plan.resource_digest,
            'source_digest': source_digest(frames), 'parameter_encoding': 1}


def _normalize(value: object, active: frozenset[int] = frozenset()) -> object:
    if id(value) in active:
        raise ValueError('Parallel encoding rejects cycles')
    active |= {id(value)}
    if inspect.isfunction(value) and value.__module__ == 'limen.experiment.manifest_core':
        if value.__qualname__ == 'make_fitted_scaler.<locals>._factory':
            cls, params = cast(tuple[object, dict[str, object]], value.__defaults__)
            return ScalerFactory(cls, copy.deepcopy(params), None)
        if value.__qualname__ == 'MLManifest.set_scaler_from_params.<locals>._scaler_factory':
            closure = {name: cell.cell_contents for name, cell in zip(value.__code__.co_freevars, value.__closure__ or (), strict=True)}
            return ScalerFactory(None, cast(dict[str, object], copy.deepcopy(closure['_static'])), cast(str, closure['param_name']))
    if isinstance(value, dict):
        return {key: _normalize(item, active) for key, item in cast(dict[str, object], value).items()}
    if isinstance(value, (list, tuple)):
        items = [_normalize(item, active) for item in cast(list[object] | tuple[object, ...], value)]
        return tuple(items) if isinstance(value, tuple) else items
    if is_dataclass(value) and not isinstance(value, type):
        clone = copy.deepcopy(value)
        for field in fields(value):
            object.__setattr__(clone, field.name, _normalize(getattr(value, field.name), active))
        return clone
    return copy.deepcopy(value)


def _contract(component: object, path: str) -> ParallelRoundContract:
    contract = getattr(component, '__parallel_round_contract__', None)
    if not isinstance(contract, ParallelRoundContract):
        raise ValueError(f'Parallel stage {path}: missing independence contract')
    if not inspect.isfunction(contract.validate):
        raise ValueError(f'Parallel stage {path}: validator must be an importable function')
    _ = reference(contract.validate)
    return contract


def components(manifest: Manifest) -> list[tuple[str, object]]:
    result: list[tuple[str, object]] = [('architecture', manifest.architecture_function)]
    if manifest.architecture_function is not None:
        for name, parameter in inspect.signature(manifest.architecture_function).parameters.items():
            if name.endswith('_factory') and callable(parameter.default):
                result.append((f'architecture.{name}', parameter.default))
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
        from limen.experiment._column_shims import apply_column_shims
        if manifest.ablation_config is not None:
            result.append(('ablation', vars(_manifest)['_apply_feature_ablation']))
        if manifest.pca_compression_config is not None:
            result.append(('pca', vars(_manifest)['_pca_transformer']))
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
    for stage in stages:
        if ((stage.path == 'architecture' and stage.contract.prediction_kind not in ('binary', 'continuous', 'target_exposure'))
                or (stage.path != 'architecture' and stage.contract.prediction_kind is not None)):
            raise ValueError(f'Parallel stage {stage.path}: invalid prediction-kind declaration')
    # Force transport validation even when a stage validator consumes no parameters.
    _ = encode(frozen)
    pipeline, resources = plan_digests(frozen, stages)
    return FrozenManifestPlan(frozen, stages, pipeline, resources)


def plan_digests(manifest: Manifest, stages: tuple[FrozenStage, ...]) -> tuple[str, str]:
    imports = [module_identity(name) for name in sorted({name for stage in stages for name in stage.contract.backend_imports})]
    return (digest((manifest, stages, imports)),
            digest((stages, imports, module_identity('limen.experiment._parallel_resources'), {'native_threads': 1, 'startup_threads': 1})))


def _defaults(component: object, params: Mapping[str, object], *, skip: int = 1, path: str = 'architecture') -> dict[str, object]:
    signature = inspect.signature(cast(Callable[..., object], component))
    owner = component.__self__ if inspect.ismethod(component) else component
    state = vars(owner) if not (inspect.isfunction(owner) or isinstance(owner, type)) and hasattr(owner, '__dict__') else {}
    result = cast(dict[str, object], dict(state)) | dict(params)
    for i, (key, value) in enumerate(signature.parameters.items()):
        if i < skip or value.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
            continue
        if key not in result:
            if value.default is inspect.Parameter.empty:
                raise ValueError(f'Parallel stage {path}: unresolved parameter {key}')
            result[key] = value.default
    return result


def _configured(path: str, component: object, arguments: Mapping[str, object], params: Mapping[str, object], *, skip: int = 1) -> dict[str, object]:
    try:
        return _defaults(component, _resolve_params(arguments, params), skip=skip, path=path)
    except KeyError as exc:
        raise ValueError(f'Parallel stage {path}: unresolved parameter {exc.args[0]}') from exc


def configurations(manifest: Manifest, params: Mapping[str, object]) -> dict[str, dict[str, object]]:
    _ = _defaults(manifest.architecture_function, params)
    resolved = cast(dict[str, object], manifest.resolve_model_kwargs(dict(params)))
    from limen.experiment._prepare_trade_context import resolve_component_kwargs
    resolve_component_kwargs(manifest.architecture_function, {}, resolved, params)
    result = {'architecture': resolved}
    for role in ('direction', 'sizing'):
        key = f'{role}_factory'
        if key in resolved:
            result[f'architecture.{key}'] = _defaults(resolved[key], dict(cast(Mapping[str, object], resolved.get(f'{role}_params') or {})) | {'seed': resolved.get('seed')}, skip=0)
    for path in ('pre_split_data_selector', 'bar_formation'):
        step = getattr(manifest, path)
        if step is not None and (path != 'bar_formation' or params.get('bar_type', 'base') != 'base'):
            result[path] = _configured(path, step[0], step[1], params)
    included = cast(Callable[[object, Mapping[str, object]], bool], vars(_manifest)['_should_include_transform'])
    result.update({f'feature_transforms.{i}': _configured(f'feature_transforms.{i}', entry.func, entry.params, params)
                   for i, entry in enumerate(manifest.feature_transforms) if included(entry, params)})
    target = manifest.target_class_config
    if target is not None:
        result['target'] = _configured('target', target.target_class, target.fit_params, params, skip=2)
        result['target']['transform_params'] = _configured('target.transform', target.target_class.transform, target.transform_params, params, skip=2)
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
            seed = params.get(manifest.ablation_config.seed_key)
            result['ablation'] = {'random_state': 0 if seed is None else seed}
        result.update(pca={'svd_solver': 'full'}, feature_shims=dict(params))
        calibration = manifest.prediction_calibration_config
        if calibration is not None and (params.get('use_calibration', True) or params.get('use_threshold', True)):
            config = calibration.resolve(dict(params))
            for name in ('calibration', 'threshold'):
                function = getattr(config, f'{name}_func')
                if function is not None and params.get(f'use_{name}', True):
                    result[name] = _defaults(function, getattr(config, f'{name}_params'), skip=3 if name == 'calibration' else 2, path=name)
    return result



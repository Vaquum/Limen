from __future__ import annotations

import copy
import inspect
import json
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from functools import partial
from importlib import import_module
from typing import Protocol, cast

import polars as pl
from typing_extensions import override

from limen._parallel_contract import ParallelRoundContract, attach
from limen.experiment._parallel_encoding import decode, digest, encode, reference, source_digest
from limen.experiment._parallel_state import component_state
from limen.experiment._parallel_identity import distribution_providers, identity_snapshot, module_identity, package_distributions
from limen.experiment.manifest_core import Manifest, MLManifest
import limen.experiment.manifest_core as _manifest

from limen.scalers.registry import SCALER_REGISTRY

_resolve_params = cast(Callable[[Mapping[str, object], Mapping[str, object]], dict[str, object]], vars(_manifest)['_resolve_params'])


class _TargetClass(Protocol):
    transform: Callable[..., object]


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
    distribution_providers: dict[str, list[str]]

    @override
    def __reduce__(self) -> tuple[Callable[..., object], tuple[bytes, dict[str, list[str]]]]:
        with distribution_providers(self.distribution_providers), identity_snapshot():
            payload = zlib.compress(json.dumps(encode(self), ensure_ascii=False, separators=(',', ':')).encode())
        return restore_plan, (payload, self.distribution_providers)


def restore_plan(payload: bytes, providers: Mapping[str, list[str]]) -> FrozenManifestPlan:
    with distribution_providers(providers), identity_snapshot():
        plan = decode(json.loads(zlib.decompress(payload)))
    if not isinstance(plan, FrozenManifestPlan):
        raise ValueError('Invalid frozen parallel plan transport')
    return plan


def clone_manifest(manifest: Manifest) -> Manifest:
    return cast(Manifest, _normalize(manifest))


_PARALLEL_EVIDENCE_SINCE = (5, 45, 7)


def legacy_metadata(metadata: Mapping[str, object]) -> bool:
    version = metadata.get('limen_version')
    if not isinstance(version, str) or len(parts := version.split('.')) != len(_PARALLEL_EVIDENCE_SINCE) or not all(part.isdigit() for part in parts):
        raise ValueError('Cannot resume: missing or inconsistent parallel evidence version')
    older = tuple(map(int, parts)) < _PARALLEL_EVIDENCE_SINCE
    if older and 'parallel_plan' in metadata:
        raise ValueError('Cannot resume: missing or inconsistent parallel evidence version')
    return older


def plan_identity(plan: FrozenManifestPlan, frames: tuple[pl.DataFrame | None, ...]) -> dict[str, object]:
    return {'version': 1, 'pipeline_digest': plan.pipeline_digest, 'resource_digest': plan.resource_digest,
            'source_digest': source_digest(frames), 'parameter_encoding': 1}


_NO_ANCESTORS: frozenset[int] = frozenset()


def _normalize(value: object, active: frozenset[int] = _NO_ANCESTORS) -> object:
    if id(value) in active:
        raise ValueError('Parallel encoding rejects cycles')
    active |= {id(value)}
    is_component = callable(value)
    if inspect.isfunction(value) and value.__module__ == 'limen.experiment.manifest_core':
        if value.__qualname__ == 'make_fitted_scaler.<locals>._factory':
            cls, params = cast(tuple[object, dict[str, object]], value.__defaults__)
            return ScalerFactory(cls, copy.deepcopy(params), None)
        if value.__qualname__ == 'MLManifest.set_scaler_from_params.<locals>._scaler_factory':
            closure = {name: cell.cell_contents for name, cell in zip(value.__code__.co_freevars, value.__closure__ or (), strict=True)}
            return ScalerFactory(None, cast(dict[str, object], copy.deepcopy(closure['_static'])), cast(str, closure['param_name']))
    if isinstance(value, dict) and not is_component:
        return {key: _normalize(item, active) for key, item in cast(dict[str, object], value).items()}
    if isinstance(value, (list, tuple)) and not is_component:
        items = [_normalize(item, active) for item in cast(list[object] | tuple[object, ...], value)]
        return tuple(items) if isinstance(value, tuple) else items
    if inspect.isfunction(value) or inspect.ismethod(value) or issubclass(type(value), partial):
        clone = decode(encode(value))
        if hasattr(value, '__dict__') and '__parallel_round_contract__' in vars(value):
            vars(clone)['__parallel_round_contract__'] = vars(value)['__parallel_round_contract__']
    elif is_dataclass(value) and not isinstance(value, type):
        clone = copy.deepcopy(value)
        for field in fields(value):
            object.__setattr__(clone, field.name, _normalize(getattr(value, field.name), active))
    else:
        clone = copy.deepcopy(value)
    return clone


def _contract(component: object, path: str) -> ParallelRoundContract:
    contract = getattr(component, '__parallel_round_contract__', None)
    if not isinstance(contract, ParallelRoundContract):
        raise ValueError(f'Parallel stage {path}: missing independence contract')
    if not inspect.isfunction(contract.validate):
        raise ValueError(f'Parallel stage {path}: validator must be an importable function')
    _ = reference(contract.validate)
    return contract


def components(manifest: Manifest, factory_options: Mapping[str, list[object]] | None = None) -> list[tuple[str, object]]:
    result: list[tuple[str, object]] = [('architecture', manifest.architecture_function)]
    if manifest.architecture_function is not None:
        parameters = inspect.signature(manifest.architecture_function).parameters
        options = dict(factory_options or {})
        names = dict.fromkeys([*parameters, *(options if any(value.kind == inspect.Parameter.VAR_KEYWORD for value in parameters.values()) else ())])
        for name in names:
            parameter = parameters.get(name)
            default = inspect.Parameter.empty if parameter is None else parameter.default
            if name.endswith('_factory'):
                result.extend((f'architecture.{name}', component) for component in options.get(name, [default])
                              if component is not inspect.Parameter.empty and callable(component))
    for path in ('pre_split_data_selector', 'bar_formation'):
        step = getattr(manifest, path)
        if step is not None:
            result.append((path, step[0]))
    result.extend((f'feature_transforms.{i}', entry.func) for i, entry in enumerate(manifest.feature_transforms))
    if manifest.target_class_config is not None:
        result.append(('target', manifest.target_class_config.target_class))
        result.extend((f'target.{name}', getattr(manifest.target_class_config.target_class, method))
                      for name, method in (('fit', '__init__'), ('transform', 'transform')))
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


def freeze_manifest(manifest: Manifest, yaml_reference: Mapping[str, object] | None,
                    factory_options: Mapping[str, list[object]] | None = None) -> FrozenManifestPlan:
    from limen.yaml.compiler import build_manifest

    frozen = clone_manifest(manifest)
    if yaml_reference is not None and digest(frozen) != digest(_normalize(build_manifest(dict(yaml_reference)))):
        raise ValueError('Parallel rounds require the unchanged compiled YAML manifest pipeline')
    stages = tuple(FrozenStage(path, component, _contract(frozen.target_class_config.target_class if path.startswith('target.') and frozen.target_class_config is not None else component, path))
                   for path, component in components(frozen, factory_options))
    for stage in stages:
        if ((stage.path == 'architecture' and stage.contract.prediction_kind not in ('binary', 'continuous', 'target_exposure'))
                or (stage.path != 'architecture' and stage.contract.prediction_kind is not None)):
            raise ValueError(f'Parallel stage {stage.path}: invalid prediction-kind declaration')
    pipeline, resources = plan_digests(frozen, stages)
    plan = FrozenManifestPlan(frozen, stages, pipeline, resources, copy.deepcopy(dict(package_distributions())))
    # Include factory variants that exist only in the stage inventory.
    with identity_snapshot():
        _ = decode(encode(plan))
    return plan


def plan_digests(manifest: Manifest, stages: tuple[FrozenStage, ...]) -> tuple[str, str]:
    with identity_snapshot():
        imports = [module_identity(name) for name in sorted({name for stage in stages for name in stage.contract.backend_imports})]
        selectors = SCALER_REGISTRY if any(isinstance(stage.component, ScalerFactory) and stage.component.dynamic_selector is not None for stage in stages) else None
        return (digest((manifest, stages, imports, selectors)),
                digest((stages, imports, selectors, module_identity('limen.experiment._parallel_resources'), {'native_threads': 1, 'startup_threads': 1})))


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
        result['target'] = _configured('target', target.target_class, target.fit_params, params, skip=2, runtime=runtime)
        transform_config = _configured('target.transform', cast(_TargetClass, target.target_class).transform, target.transform_params, params, skip=2, runtime=runtime)
        result['target']['transform_params'] = transform_config
        result.update({'target.fit': result['target'], 'target.transform': result['target'] | transform_config})
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



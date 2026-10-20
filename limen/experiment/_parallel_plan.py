from __future__ import annotations

import copy
import inspect
import json
import zlib
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from functools import partial
from importlib import import_module
from string import Formatter
from types import CellType, FunctionType, MethodType
from typing import Protocol, cast

import polars as pl
from typing_extensions import override

from limen._parallel_contract import ParallelRoundContract, attach
from limen.experiment._parallel_encoding import import_reference, decode, digest, encode, reference, source_digest
from limen.experiment._parallel_graph import copy_context
from limen.experiment._parallel_state import ObjectState, call_implementation, object_state, restore_object, mutable_function
from limen.experiment._parallel_identity import distribution_providers, identity_snapshot, module_identity, package_distributions
from limen.experiment.manifest_core import Manifest, MLManifest
import limen.experiment.manifest_core as _manifest

from limen.scalers.registry import SCALER_REGISTRY

class TargetClass(Protocol):
    transform: Callable[..., object]


class _CapturedCall(partial[object]):
    def __getattr__(self, name: str) -> object:
        return getattr(cast(MethodType, self.func).__self__, name)

    @override
    def __setattr__(self, name: str, value: object) -> None:
        if name == '__parallel_round_contract__':
            vars(self)[name] = value
        else:
            setattr(cast(MethodType, self.func).__self__, name, value)


@dataclass(frozen=True)
class ScalerFactory:
    transform_class: object
    static_params: dict[str, object]
    dynamic_selector: str | None

    def selected(self, params: Mapping[str, object]) -> object:
        return self.transform_class if self.dynamic_selector is None else cast(dict[str, object], self.transform_class)[cast(str, params['scaler_type'])]

    def __call__(self, data: pl.DataFrame, **params: object) -> object:
        extras = {key: value for key, value in params.items() if self.dynamic_selector is None or key != 'scaler_type'}
        return cast(Callable[..., object], self.selected(params))(data, **self.static_params, **extras)


attach(ScalerFactory)


@dataclass
class _FrozenTargetConfig(_manifest.TargetClassConfig):
    _parallel_init: Callable[..., object] | None = None
    _parallel_transform: Callable[..., object] | None = None
    _parallel_new: Callable[..., object] | None = None

    def _parallel_make(self, **params: object) -> object:
        cls = self.target_class
        allocator = self._parallel_new or cast(Callable[..., object], cls.__new__)
        instance = allocator(cls, **({} if allocator is object.__new__ else params))
        if isinstance(instance, cls):
            assert self._parallel_init is not None
            _ = self._parallel_init(instance, **params)
        return instance


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
            return restore_plan, (zlib.compress(json.dumps(encode(self), ensure_ascii=False, separators=(',', ':')).encode()), self.distribution_providers)


def restore_plan(payload: bytes, providers: Mapping[str, list[str]]) -> FrozenManifestPlan:
    with distribution_providers(providers), identity_snapshot():
        plan = decode(json.loads(zlib.decompress(payload)), isolate_functions=True)
    if not isinstance(plan, FrozenManifestPlan):
        raise ValueError('Invalid frozen parallel plan transport')
    return plan


@contextmanager
def shared_copy(memo: dict[int, tuple[object, object]]) -> Iterator[None]:
    token = copy_context.set((memo, {}))
    try:
        yield
    finally:
        copy_context.reset(token)


def clone_manifest(manifest: Manifest, *, snapshot: bool = False) -> Manifest:
    with identity_snapshot():
        return cast(Manifest, _normalize(manifest, snapshot=snapshot))


def component_digest(component: object) -> str:
    return digest(_normalize(component, snapshot=True))


def factory_reference(value: object) -> tuple[list[object], bool]:
    if isinstance(value, partial):
        bound = cast(partial[object], value)
        state = object_state(bound)
        source, imported = factory_reference(bound.func)
        return [reference(type(bound)), source, encode((state[0], state[1], bound.args, bound.keywords, state[3]))], imported
    if inspect.ismethod(value):
        identity = reference(value.__func__)
        return [identity, factory_reference(value.__self__)[0]], import_reference(identity) is value.__func__
    if not (inspect.isfunction(value) or isinstance(value, type)):
        return [reference(type(value)), encode(object_state(value))], callable(value)
    identity = reference(value)
    return identity, import_reference(identity) is value


def factory_allocator(value: object) -> Callable[..., object] | None:
    if isinstance(value, partial):
        bound = cast(partial[object], value)
        allocator = factory_allocator(bound.func)
        return None if allocator is None else partial(allocator, *bound.args, **bound.keywords)
    if isinstance(value, cast(type, vars(import_module('limen.experiment._parallel_resources'))['FrozenClassFactory'])):
        allocator = cast(Callable[..., object] | None, vars(value)['allocator'])
        return None if allocator is None else partial(allocator, vars(value)['factory_class'])
    return None


_PARALLEL_EVIDENCE_SINCE = (5, 45, 16)


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


def _scaler_factory(cls: object, params: dict[str, object], selector: str | None) -> ScalerFactory:
    return cast(Callable[[object, dict[str, object], str | None], ScalerFactory], vars(import_module('limen.experiment._parallel_resources'))['scaler_factory'])(cls, params, selector)


def _normalize(value: object, active: frozenset[int] = _NO_ANCESTORS, *, snapshot: bool = False, memo: dict[int, tuple[object, object]] | None = None,
               state_memo: dict[int, tuple[object, object]] | None = None) -> object:
    if id(value) in active:
        raise ValueError('Parallel encoding rejects cycles')
    memo = {} if memo is None else memo
    state_memo = {} if state_memo is None else state_memo
    key = id(value)
    if key in memo:
        return memo[key][1]
    active |= {id(value)}
    is_component = callable(value)
    if type(value) is _manifest.TargetClassConfig:
        target = value
        value = _FrozenTargetConfig(target.target_class, target.fit_params, target.transform_params,
                                    cast(Callable[..., object], target.target_class.__init__), cast(TargetClass, target.target_class).transform,
                                    target.target_class.__new__ if inspect.isfunction(target.target_class.__new__) else None)
    if inspect.isfunction(value) and value.__module__ == 'limen.experiment.manifest_core':
        if value.__qualname__ == 'make_fitted_scaler.<locals>._factory':
            cls, params = cast(tuple[object, dict[str, object]], value.__defaults__)
            value = _scaler_factory(cls, params, None)
        elif value.__qualname__ == 'MLManifest.set_scaler_from_params.<locals>._scaler_factory':
            closure = {name: cell.cell_contents for name, cell in zip(value.__code__.co_freevars, value.__closure__ or (), strict=True)}
            value = _scaler_factory(None, cast(dict[str, object], closure['_static']), cast(str, closure['param_name']))
    clone: object
    if isinstance(value, CellType):
        clone = state_memo[key][1] if key in state_memo else CellType(None)
        cast(CellType, clone).cell_contents = _normalize(value.cell_contents, active, snapshot=snapshot, memo=memo, state_memo=state_memo)
    elif isinstance(value, dict) and not (is_component or isinstance(value, defaultdict)):
        mapped = {name: _normalize(item, active, snapshot=snapshot, memo=memo, state_memo=state_memo) for name, item in cast(dict[object, object], value).items()}
        clone = state_memo[key][1] if key in state_memo else dict[object, object]()
        cast(dict[object, object], clone).clear()
        cast(dict[object, object], clone).update(mapped)
    elif isinstance(value, (list, tuple)) and not is_component:
        items = [_normalize(item, active, snapshot=snapshot, memo=memo, state_memo=state_memo) for item in cast(list[object] | tuple[object, ...], value)]
        clone = tuple(items) if isinstance(value, tuple) else state_memo[key][1] if key in state_memo else list[object]()
        if isinstance(clone, list):
            cast(list[object], clone)[:] = items
    else:
        clone = _normalize_component(cast(object, value), active, snapshot=snapshot, memo=memo, state_memo=state_memo)
    memo[key] = (value, clone)
    if not callable(cast(object, value)) or key not in state_memo:
        state_memo[key] = (cast(object, value), clone)
    return cast(object, clone)


def _normalize_component(value: object, active: frozenset[int], *, snapshot: bool, memo: dict[int, tuple[object, object]], state_memo: dict[int, tuple[object, object]]) -> object:
    if (callable(value) and not isinstance(value, type)) or isinstance(value, defaultdict):
        imported = import_reference(reference(value.__func__ if isinstance(value, MethodType) else value)) if not snapshot and isinstance(value, (FunctionType, MethodType)) else None
        isolate = snapshot or (imported is not None and encode(getattr(imported, '__parallel_round_contract__', None)) != encode(getattr(cast(object, value), '__parallel_round_contract__', None)))
        with shared_copy(state_memo):
            clone = decode(encode(cast(object, value)), isolate_functions=isolate)
        if isinstance(value, FunctionType) and clone is not value and (snapshot or mutable_function(value)):
            cast(FunctionType, clone).__defaults__ = cast(tuple[object, ...] | None, _normalize(value.__defaults__, active, snapshot=snapshot, memo=memo, state_memo=state_memo))
            cast(FunctionType, clone).__kwdefaults__ = cast(dict[str, object] | None, _normalize(value.__kwdefaults__, active, snapshot=snapshot, memo=memo, state_memo=state_memo))
            _ = _normalize(value.__closure__, active, snapshot=snapshot, memo=memo, state_memo=state_memo)
        elif not inspect.isfunction(source := cast(object, value)):
            owner = source.__self__ if inspect.ismethod(source) else source
            restored = clone.__self__ if inspect.ismethod(clone) else clone
            if not isinstance(owner, type):
                _ = restore_object(type(owner), cast(ObjectState, _normalize(object_state(owner), active, snapshot=snapshot, memo=memo, state_memo=state_memo)), restored)
            if inspect.ismethod(source):
                _ = _normalize(source.__func__, active, snapshot=snapshot, memo=memo, state_memo=state_memo)
        if (hasattr(cast(object, value), '__dict__') and '__parallel_round_contract__' in vars(value)
                and (snapshot or not isinstance(value, (FunctionType, MethodType)) or encode(getattr(clone, '__parallel_round_contract__', None)) != encode(vars(value)['__parallel_round_contract__']))):
            vars(clone)['__parallel_round_contract__'] = _normalize(vars(value)['__parallel_round_contract__'], active, snapshot=snapshot, memo=memo, state_memo=state_memo)
    elif is_dataclass(value) and not isinstance(value, type):
        clone = state_memo[id(value)][1] if id(value) in state_memo else copy.deepcopy(value)
        for field in fields(value):
            object.__setattr__(clone, field.name, _normalize(getattr(value, field.name), active, snapshot=snapshot, memo=memo, state_memo=state_memo))
    else:
        clone = copy.deepcopy(cast(object, value))
    implementation = call_implementation(cast(object, value))
    implementation = implementation.__func__ if inspect.ismethod(implementation) else implementation
    if snapshot and inspect.isfunction(implementation) and (implementation.__defaults__ or implementation.__kwdefaults__ or implementation.__closure__):
        with shared_copy(state_memo):
            clone = _CapturedCall(cast(Callable[..., object], decode(encode(cast(Callable[..., object], clone).__call__), isolate_functions=True)))
        vars(clone)['__parallel_round_contract__'] = _normalize(getattr(cast(object, value), '__parallel_round_contract__', None), active, snapshot=snapshot, memo=memo, state_memo=state_memo)
    return clone


def _contract(component: object, path: str) -> ParallelRoundContract:
    contract = getattr(component, '__parallel_round_contract__', None)
    if not isinstance(contract, ParallelRoundContract):
        raise ValueError(f'Parallel stage {path}: missing independence contract')
    if not inspect.isfunction(contract.validate):
        raise ValueError(f'Parallel stage {path}: validator must be an importable function')
    return cast(ParallelRoundContract, _normalize(contract, snapshot=True))


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
        result.extend((f'target.{name}', getattr(manifest.target_class_config, captured))
                      for name, captured in (('fit', '_parallel_init'), ('transform', '_parallel_transform'), ('allocate', '_parallel_new'))
                      if getattr(manifest.target_class_config, captured) is not None)
    if isinstance(manifest, MLManifest):
        if manifest.scaler is not None:
            computations, transform, _ = manifest.scaler
            result.extend((f'scaler.fit.{i}', compute) for i, (_, compute, _) in enumerate(computations))
            result.append(('scaler.transform', transform))
            for i, (_, compute, _) in enumerate(computations):
                if isinstance(compute, ScalerFactory):
                    classes = (compute.transform_class,) if compute.dynamic_selector is None else tuple(cast(dict[str, object], compute.transform_class).values())
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
    resources = vars(import_module('limen.experiment._parallel_resources'))
    bind_factories = cast(Callable[[Mapping[str, list[object]], Mapping[str, object]], dict[str, object]], resources['bind_factories'])
    factory_bindings = cast(Callable[[Manifest, Mapping[str, list[object]] | None], dict[str, list[object]]], resources['factory_bindings'])

    source = copy.copy(manifest)
    vars(source)['_parallel_factory_options'] = factory_bindings(manifest, factory_options)
    with identity_snapshot():
        frozen, factory_options = cast(tuple[Manifest, Mapping[str, list[object]] | None], _normalize((source, factory_options), snapshot=True))
    comparison = copy.copy(frozen)
    vars(comparison)['_parallel_factory_options'] = {}
    if yaml_reference is not None and digest(comparison) != digest(_normalize(build_manifest(dict(yaml_reference)))):
        raise ValueError('Parallel rounds require the unchanged compiled YAML manifest pipeline')
    stages = tuple(FrozenStage(path, component, _contract(frozen.target_class_config.target_class if path.startswith('target.') and frozen.target_class_config is not None else component, path))
                   for path, original in components(frozen, factory_options)
                   for component in [bind_factories(cast(dict[str, list[object]], vars(frozen)['_parallel_factory_options']), {path.removeprefix('architecture.'): original})[path.removeprefix('architecture.')]])
    stages += tuple(FrozenStage(stage.path + '.allocate', stage.component, stage.contract) for stage in stages
                    if stage.path.startswith(('architecture.', 'scaler.class.')) and factory_allocator(stage.component) is not None)
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


def static_source(value: object, domain: Mapping[str, list[object]]) -> None:
    if callable(value):
        raise ValueError('Parallel rounds require static source parameters')
    if isinstance(value, str):
        if value in domain or any(field is not None for _, field, _, _ in Formatter().parse(value)):
            raise ValueError('Parallel rounds reject parameter-dependent source declarations')
    elif isinstance(value, Mapping):
        for item in cast(Mapping[object, object], value).values():
            static_source(item, domain)
    elif isinstance(value, (tuple, list)):
        for item in cast(tuple[object, ...] | list[object], value):
            static_source(item, domain)


def parameter_references(value: object, domain: Mapping[str, list[object]]) -> set[str]:
    if isinstance(value, str):
        result: set[str] = {value} if value in domain else set()
        if '{' in value and '}' in value:
            for _, field, spec, _ in Formatter().parse(value):
                if field is not None:
                    result.add(field.split('.', 1)[0].split('[', 1)[0])
                result.update(parameter_references(spec, domain))
        return result.intersection(domain)
    if isinstance(value, Mapping):
        return set[str]().union(*(parameter_references(item, domain) for item in cast(Mapping[object, object], value).values()))
    if isinstance(value, (tuple, list)):
        return set[str]().union(*(parameter_references(item, domain) for item in cast(tuple[object, ...] | list[object], value)))
    return set()



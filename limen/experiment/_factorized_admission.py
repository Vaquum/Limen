from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
from itertools import product
from types import CodeType, FunctionType
from typing import cast

from limen.experiment._factorized_contract import ReuseContract
from limen.experiment._factorized_key import parameter_key
from limen.experiment._factorized_native import contract_for, native_rule
from limen.experiment._factorized_projection import projections
from limen.experiment.manifest_core import MLManifest, Manifest, RuleBasedManifest, make_fitted_scaler
from limen.scalers import SCALER_REGISTRY, RobustScaler
from limen.targets.next_return import NextReturnTarget
from limen.targets.quantile_binary import QuantileBinaryTarget
from limen.features.lagged_features import lag_range
from limen.indicators.window_return import window_return
import limen.experiment.manifest_core as core

_resolve = cast(Callable[[Mapping[str, object], Mapping[str, object]], dict[str, object]], vars(core)['_resolve_params'])
_include = cast(Callable[[object, dict[str, object]], bool], vars(core)['_should_include_transform'])


def _effective(component: object, arguments: Mapping[str, object], *, skip: int = 1) -> dict[str, object]:
    return {name: parameter.default for name, parameter in list(inspect.signature(cast(Callable[..., object], component)).parameters.items())[skip:]
            if parameter.default is not inspect.Parameter.empty} | dict(arguments)


def _stage(path: str, component: object, arguments: Mapping[str, object], axes: frozenset[str]) -> ReuseContract:
    declaration = contract_for(component)
    if declaration is None:
        raise ValueError(f'factorize requires stage {path}: missing deterministic reuse declaration')
    effective = _effective(component, arguments)
    consumed = set(effective) if declaration.consumes is None else {name.split('.')[0] for name in declaration.consumes}
    if axes.intersection(consumed):
        raise ValueError(f'factorize requires stage {path}: consumes an economic parameter')
    if set(declaration.parameter_keys) - set(effective):
        raise ValueError(f'factorize requires stage {path}: missing required declared parameters')
    try:
        declaration.validate(effective)
    except ValueError as exc:
        raise ValueError(f'factorize requires stage {path}: {exc}') from exc
    return declaration


def _scaler(manifest: MLManifest, params: dict[str, object], axes: frozenset[str]) -> None:
    if manifest.scaler is None:
        return
    computations, transform, settings = manifest.scaler
    if len(computations) != 1 or transform is not vars(core)['_apply_fitted_transform']:
        raise ValueError('factorize requires stage scaler: unknown factory adaptation')
    name, factory, dynamic = computations[0]
    reference = make_fitted_scaler(name, object)[0][0][1]
    if not isinstance(factory, FunctionType) or not isinstance(reference, FunctionType) or settings != {'fitted_transform': name}:
        raise ValueError('factorize requires stage scaler: unknown factory adaptation')
    arguments = _resolve(dynamic, params)
    if factory.__code__ is reference.__code__:
        signature = inspect.signature(factory).parameters
        component, fixed = signature['_cls'].default, signature['_p'].default
    else:
        codes = [value for value in MLManifest.set_scaler_from_params.__code__.co_consts if isinstance(value, CodeType)]
        if factory.__code__ not in codes:
            raise ValueError('factorize requires stage scaler: undeclared custom factory')
        state = inspect.getclosurevars(factory).nonlocals
        component = SCALER_REGISTRY[str(arguments.pop('scaler_type'))]
        fixed = state['_static']
    _ = _stage('scaler', component, {**cast(dict[str, object], fixed), **arguments}, axes)


def validate_candidate(manifest: Manifest, params: dict[str, object], axes: frozenset[str]) -> ReuseContract:
    owner = MLManifest if isinstance(manifest, MLManifest) else RuleBasedManifest
    if any(getattr(type(manifest), name, None) is not getattr(owner, name, None) for name in ('prepare_data', '_prepare_data_with_fitted', 'run_model')):
        raise ValueError('factorize requires the original manifest preparation and model methods')
    if manifest.split_walk_forward is not None or manifest.allocation_config is not None or getattr(manifest, 'objective', None) is not None:
        raise ValueError('factorize requires independent snapshot execution without folds, allocation or objective')
    if manifest.resolve_trade_policy(params) is not None:
        raise ValueError('factorize requires snapshot execution, not event execution')
    function = manifest.architecture_function
    model_kwargs = manifest.resolve_model_kwargs(params)
    declaration = _stage('architecture', function, model_kwargs, axes)
    if isinstance(manifest, RuleBasedManifest) and not native_rule(function) and declaration.replay is None:
        raise ValueError('factorize stage architecture: custom rule economics require explicit replay')
    if declaration.prediction_kind is None and declaration.replay is None:
        raise ValueError('factorize requires stage architecture: declare prediction kind or economic replay')
    for index, entry in enumerate(manifest.feature_transforms):
        if _include(entry, params):
            _ = _stage(f'transforms[{index}]', entry.func, _resolve(entry.params, params), axes)
    for name in ('pre_split_data_selector', 'bar_formation'):
        step = getattr(manifest, name)
        if step is not None and (name != 'bar_formation' or params.get('bar_type', 'base') != 'base'):
            component, arguments = step
            _ = _stage(name, component, _resolve(arguments, params), axes)
    target = manifest.target_class_config
    if target is not None:
        _ = _stage('target', target.target_class, {'target_name': manifest.target_column, **_resolve(target.fit_params, params), **_resolve(target.transform_params, params)}, axes)
    if isinstance(manifest, MLManifest):
        _scaler(manifest, params, axes)
        config = manifest.prediction_calibration_config
        if config is None:
            supplied = model_kwargs.get('prediction_calibration_config')
            if supplied is not None and not isinstance(supplied, core.CalibrationConfig):
                raise ValueError('factorize requires a declared calibration configuration')
            config = supplied
        if config is not None:
            config = config.resolve(params)
            signature = inspect.signature(cast(Callable[..., object], function))
            if ((config.calibration_func is not None and params.get('use_calibration', True)) or (config.threshold_func is not None and params.get('use_threshold', True))) and not any(name == 'prediction_calibration_config' or item.kind == inspect.Parameter.VAR_KEYWORD for name, item in signature.parameters.items()):
                raise ValueError('factorize requires architecture acceptance of active calibration configuration')
            for name, component, arguments, active in (
                ('calibration', config.calibration_func, config.calibration_params, params.get('use_calibration', True)),
                ('threshold', config.threshold_func, config.threshold_params, params.get('use_threshold', True)),
            ):
                if component is not None and active:
                    _ = _stage(name, component, arguments, axes)
                    if callable(arguments.get('metric')):
                        _ = _stage(name + '.metric', arguments['metric'], {}, axes)
        if manifest.data_dict_extension is not None:
            _ = _stage('data_dict_extension', manifest.data_dict_extension, params, axes)
    elif not isinstance(manifest, RuleBasedManifest) or manifest.strategy is None or target is not None:
        raise ValueError('factorize requires a configured ML or rule-based manifest')
    return declaration


def validate_domain(manifest: Manifest, domain: Mapping[str, list[object]], axes: frozenset[str]) -> ReuseContract:
    for key, values in domain.items():
        for value in values:
            _ = parameter_key({key: value})
    base = {name: values[0] for name, values in domain.items()}
    declaration: ReuseContract | None = None
    seen: set[str] = set()
    for group in projections(manifest, domain):
        names = sorted(group - axes)
        for values in product(*(domain[name] for name in names)):
            params = base | dict(zip(names, values, strict=True))
            identity = parameter_key(params)
            if identity not in seen:
                declaration = validate_candidate(manifest, params, axes)
                seen.add(identity)
    assert declaration is not None
    return declaration


def _supports_scaler(manifest: MLManifest) -> bool:
    if manifest.scaler is None:
        return True
    computations, transform, params = manifest.scaler
    if len(computations) != 1:
        return False
    name, factory, dynamic = computations[0]
    reference = make_fitted_scaler(name, RobustScaler)
    reference_factory = reference[0][0][1]
    return (isinstance(factory, FunctionType) and isinstance(reference_factory, FunctionType)
            and factory.__code__ is reference_factory.__code__ and transform is reference[1]
            and inspect.signature(factory).parameters['_cls'].default is RobustScaler
            and set(dynamic).issubset({'quantile_range'})
            and all(isinstance(value, str) for value in dynamic.values())
            and params == {'fitted_transform': name})


def supports_preparation(manifest: MLManifest) -> bool:
    target = manifest.target_class_config
    return (target is not None and target.target_class in (NextReturnTarget, QuantileBinaryTarget)
            and _supports_scaler(manifest) and manifest.ablation_config is None
            and manifest.pca_compression_config is None and manifest.prediction_calibration_config is None
            and not manifest.feature_shims
            and manifest.data_dict_extension is None and manifest.objective is None
            and all(entry.func in (window_return, lag_range) for entry in manifest.feature_transforms))



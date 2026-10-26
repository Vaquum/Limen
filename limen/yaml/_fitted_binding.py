from collections.abc import Callable, Mapping, Sequence
import inspect
import logging
from typing import cast

from limen.scalers import SCALER_REGISTRY
from limen.yaml._callable_binding import binding_errors
from limen.yaml.errors import ResolutionError, YAMLError
from limen.yaml.resolver import resolve

__all__: list[str] = []
logger = logging.getLogger(__name__)
_ROOT = 'sfd.manifest.'


def _mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def _resolve(name: object, path: str, errors: list[YAMLError], *, class_only: bool = False) -> Callable[..., object] | None:
    if not isinstance(name, str):
        errors.append(YAMLError(message='Component must name a resolvable callable', path=path))
        return None
    try:
        component = resolve(name)
    except ResolutionError:
        errors.append(YAMLError(message=f"Cannot resolve '{name}'", path=path))
        return None
    if not callable(component) or (class_only and not inspect.isclass(component)):
        errors.append(YAMLError(message=f"'{name}' must resolve to a {'class' if class_only else 'callable'}", path=path))
        return None
    return component


def _check(component: Callable[..., object], args: Sequence[object], params: object, path: str,
           errors: list[YAMLError], *, supplied: Mapping[str, object] | None = None,
           reserved: tuple[str, ...] = (), context: str = '') -> None:
    if params is not None and not isinstance(params, Mapping):
        errors.append(YAMLError(message='Call parameters must be a mapping', path=path))
        return
    values = dict(cast(Mapping[object, object], params)) if params is not None else {}
    framework = dict(supplied or {})
    for key in (*framework, *reserved):
        if key in values:
            errors.append(YAMLError(message=f"'{key}' collides with a framework operand{context}", path=f'{path}.{key}'))
            del values[key]
    issues = binding_errors(component, args, {**values, **framework})
    if issues is not None:
        errors.extend(YAMLError(message=message + context, path=path + (f'.{key}' if key else '')) for key, message in issues.items())


def _target(block: Mapping[str, object], errors: list[YAMLError]) -> None:
    path = _ROOT + 'target'
    component = _resolve(block.get('class'), path + '.class', errors, class_only=True)
    if component is None:
        return
    cls = cast(type[object], component)
    context = {'trade_context': object()} if getattr(cls, 'requires_trade_context', False) else {}
    _check(component, (), block.get('fit_params'), path + '.fit_params', errors,
           supplied={'train_data': object(), 'target_name': object(), **context})
    descriptor = inspect.getattr_static(cls, 'transform', None)
    if isinstance(descriptor, staticmethod):
        transform, args = descriptor.__func__, (object(),)
    elif isinstance(descriptor, classmethod):
        transform, args = cast(Callable[..., object], getattr(cls, 'transform', None)), (object(),)
    elif inspect.isfunction(descriptor):
        transform, args = descriptor, (object(), object())
    else:
        logger.debug('Target transform binding is indeterminate for %s', cls.__qualname__)
        return
    _check(transform, args, block.get('transform_params'), path + '.transform_params', errors, supplied=context)


def _scaler(block: Mapping[str, object], axes: Mapping[str, object], errors: list[YAMLError]) -> None:
    path = _ROOT + 'scaler'
    candidates: dict[str, Callable[..., object]] = {}
    dynamic = 'from_params' in block
    if dynamic:
        name = block['from_params']
        choices = axes.get(name) if isinstance(name, str) else None
        if not isinstance(choices, list) or not choices:
            errors.append(YAMLError(message='from_params must name a declared nonempty scaler axis', path=path + '.from_params'))
            return
        for index, value in enumerate(cast(list[object], choices)):
            if not isinstance(value, str) or value not in SCALER_REGISTRY:
                errors.append(YAMLError(message=f'Unknown scaler candidate {value!r}', path=f'sfd.params.{name}[{index}]'))
            else:
                candidates[value] = SCALER_REGISTRY[value]
    else:
        component = _resolve(block.get('class'), path + '.class', errors, class_only=True)
        if component is not None:
            candidates[str(block['class'])] = component
    reserved = ('data', 'scaler_type') if dynamic else ('data', '_cls', '_p')
    for name, component in candidates.items():
        _check(component, (object(),), block.get('params'), path + '.params', errors, reserved=reserved,
               context=f' for scaler candidate {name!r} ({component.__name__})')


def _calibration(block: Mapping[str, object], manifest: Mapping[str, object], errors: list[YAMLError]) -> None:
    path = _ROOT + 'calibration'
    architecture = _resolve(manifest.get('reference_architecture'), _ROOT + 'reference_architecture', errors)
    if architecture is not None:
        try:
            parameters = inspect.signature(architecture).parameters
        except (TypeError, ValueError) as error:
            logger.debug('Calibration architecture acceptance is indeterminate: %s', error)
        else:
            accepted = any(p.kind == inspect.Parameter.VAR_KEYWORD or
                           (name == 'prediction_calibration_config' and p.kind in (
                               inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY))
                           for name, p in parameters.items())
            if not accepted:
                errors.append(YAMLError(message='Architecture does not accept prediction_calibration_config', path=path))
    objective = manifest.get('objective') is not None
    for kind, arity in (('probability_calibration', 3), ('threshold_function', 2)):
        declaration = _mapping(block.get(kind))
        if not declaration:
            continue
        component = _resolve(declaration.get('func'), path + f'.{kind}.func', errors)
        if component is None:
            continue
        threshold = kind == 'threshold_function'
        supplied = {'metric': object(), '_objective_maximize': object()} if threshold and objective else {}
        reserved = ('self', 'func', '_objective_maximize') if threshold else ('self', 'func')
        if objective:
            reserved += ('y_val', 'val_proba') if threshold else ('clf', 'x_val', 'y_val')
        _check(component, (object(),) * arity, declaration.get('params'), path + f'.{kind}.params', errors,
               supplied=supplied, reserved=reserved)


def fitted_errors(document: Mapping[str, object]) -> list[YAMLError]:
    sfd = _mapping(document.get('sfd'))
    manifest = _mapping(sfd.get('manifest'))
    errors: list[YAMLError] = []
    if manifest.get('type') == 'ml':
        if target := _mapping(manifest.get('target')):
            _target(target, errors)
        if scaler := _mapping(manifest.get('scaler')):
            _scaler(scaler, _mapping(sfd.get('params')), errors)
        if calibration := _mapping(manifest.get('calibration')):
            _calibration(calibration, manifest, errors)
    return errors

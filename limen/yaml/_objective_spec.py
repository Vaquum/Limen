from collections.abc import Mapping
from typing import cast

from limen.experiment._objective import ObjectiveConfig
from limen.experiment.manifest_core import MLManifest
from limen.yaml.errors import YAMLError


def _mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def _read_objective(yaml_dict: Mapping[str, object], errors: list[YAMLError]) -> ObjectiveConfig | None:
    sfd = _mapping(yaml_dict.get('sfd'))
    manifest = _mapping(sfd.get('manifest'))
    for path, container in (('objective', yaml_dict), ('uel.objective', _mapping(yaml_dict.get('uel')))):
        if 'objective' in container:
            errors.append(YAMLError(message='Declare the objective only at sfd.manifest.objective', path=path))
    if 'objective' not in manifest:
        return None
    path = 'sfd.manifest.objective'
    if manifest.get('type') != 'ml':
        errors.append(YAMLError(message='Objectives require an ML manifest', path=path))
    value = manifest['objective']
    if not isinstance(value, Mapping):
        errors.append(YAMLError(message='Objective must be a mapping with metric and direction', path=path))
        return None
    declaration = _mapping(cast(object, value))
    if set(declaration) != {'metric', 'direction'}:
        errors.append(YAMLError(message='Objective requires exactly metric and direction fields', path=path))
        return None
    metric, direction = declaration['metric'], declaration['direction']
    if not isinstance(metric, str) or not isinstance(direction, str):
        errors.append(YAMLError(message='Objective metric and direction must be literal strings', path=path))
        return None
    try:
        return ObjectiveConfig(metric=metric, direction=direction)
    except ValueError as exc:
        errors.append(YAMLError(message=str(exc), path=path))
        return None


def read_objective(yaml_dict: Mapping[str, object]) -> ObjectiveConfig | None:
    errors: list[YAMLError] = []
    objective = _read_objective(yaml_dict, errors)
    if errors:
        raise ValueError('; '.join(f'{error.path}: {error.message}' for error in errors))
    return objective


def apply_objective(manifest: MLManifest, yaml_dict: Mapping[str, object]) -> MLManifest:
    objective = read_objective(yaml_dict)
    if objective is not None:
        _ = manifest.set_objective(metric=objective.metric, direction=objective.direction)
    return manifest


def _reducer_params(objective: ObjectiveConfig | None, reducer_type: str, params: Mapping[str, object]) -> dict[str, object]:
    bound = dict(params)
    if objective is None or reducer_type == 'budget' and params.get('trim_strategy', 'random') != 'worst_first':
        return bound
    if reducer_type not in {'correlation', 'focus', 'sanity', 'saturation', 'budget'}:
        return bound
    expected: dict[str, object] = {'metric': objective.column}
    if reducer_type in {'correlation', 'focus', 'budget'}:
        expected['maximize'] = objective.maximize
    for key, value in expected.items():
        if key in bound and (not isinstance(bound[key], type(value)) or bound[key] != value):
            raise ValueError(f'{reducer_type}.{key} conflicts with the declared objective: expected {value!r}')
        bound[key] = value
    return bound


def objective_reducer_params(yaml_dict: Mapping[str, object], reducer_type: str, params: Mapping[str, object]) -> dict[str, object]:
    return _reducer_params(read_objective(yaml_dict), reducer_type, params)


def check_objective_spec(yaml_dict: Mapping[str, object], errors: list[YAMLError]) -> None:
    objective = _read_objective(yaml_dict, errors)
    if objective is None:
        return
    specs = _mapping(yaml_dict.get('uel')).get('pruning_strategies', [])
    if not isinstance(specs, list):
        return
    for index, raw_spec in enumerate(cast(list[object], specs)):
        spec = _mapping(raw_spec)
        reducer_type = spec.get('type')
        params = spec.get('params', {})
        if isinstance(reducer_type, str) and isinstance(params, Mapping):
            try:
                _ = _reducer_params(objective, reducer_type, _mapping(cast(object, params)))
            except ValueError as exc:
                errors.append(YAMLError(message=str(exc), path=f'uel.pruning_strategies[{index}].params'))

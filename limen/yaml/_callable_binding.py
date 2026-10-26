from collections.abc import Callable, Mapping, Sequence
import inspect
import logging
from typing import TypeVar, cast

from limen.yaml.errors import YAMLError
from limen.yaml.resolver import is_resolvable, resolve

__all__: list[str] = []
logger = logging.getLogger(__name__)
_Key = TypeVar('_Key')


def binding_errors(component: Callable[..., object], args: Sequence[object], kwargs: Mapping[_Key, object]) -> dict[str, str] | None:
    errors: dict[str, str] = {}
    values: dict[str, object] = {}
    for key, value in kwargs.items():
        if isinstance(key, str):
            values[key] = value
        else:
            errors[str(key)] = 'Keyword names must be strings'
    if errors:
        return errors
    try:
        signature = inspect.signature(component)
    except (TypeError, ValueError) as error:
        logger.debug('Callable binding is indeterminate: %s', error)
        return None
    try:
        _ = signature.bind_partial(*args)
    except TypeError as error:
        return {'': str(error)}
    for name, value in values.items():
        try:
            _ = signature.bind_partial(*args, **{name: value})
        except TypeError as error:
            errors[name] = str(error)
    if errors:
        return errors
    bound = signature.bind_partial(*args, **values)
    missing = [name for name, parameter in signature.parameters.items()
               if parameter.default is inspect.Parameter.empty
               and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
               and name not in bound.arguments]
    if missing:
        errors[''] = 'Missing required arguments: ' + ', '.join(missing)
    return errors


def check_transform_binding(block: Mapping[str, object], path: str, errors: list[YAMLError]) -> None:
    kind = path.removeprefix('sfd.manifest.').split('[', 1)[0]
    if kind not in ('indicators', 'features', 'bar_formation', 'pre_split_data_selector'):
        return
    name, params = block.get('func'), block.get('params')
    if not isinstance(name, str) or not is_resolvable(name) or (params is not None and not isinstance(params, Mapping)):
        return
    component = resolve(name)
    if not callable(component):
        return
    values = dict(cast(Mapping[str, object], params)) if params is not None else {}
    reserved = {'self', 'func'}
    if kind in ('indicators', 'features'):
        _ = values.pop('group', None)
        reserved.add('include_if')
    if kind != 'pre_split_data_selector':
        reserved.add('function')
    for key in sorted(reserved & values.keys()):
        errors.append(YAMLError(message=f"'{key}' collides with the transform wrapper", path=f'{path}.params.{key}'))
        del values[key]
    issues = binding_errors(component, (object(),), values)
    if issues is not None:
        errors.extend(YAMLError(message=message, path=f'{path}.params' + (f'.{key}' if key else '')) for key, message in issues.items())


def transform_errors(document: Mapping[str, object]) -> list[YAMLError]:
    sfd = document.get('sfd')
    manifest = cast(Mapping[str, object], sfd).get('manifest') if isinstance(sfd, Mapping) else None
    errors: list[YAMLError] = []
    if isinstance(manifest, Mapping):
        for kind in ('indicators', 'features', 'bar_formation', 'pre_split_data_selector'):
            entries = cast(Mapping[str, object], manifest).get(kind)
            blocks = enumerate(cast(list[object], entries)) if isinstance(entries, list) else ((None, entries),)
            for index, block in blocks:
                if isinstance(block, Mapping):
                    check_transform_binding(cast(Mapping[str, object], block), 'sfd.manifest.' + kind +
                                            (f'[{index}]' if index is not None else ''), errors)
    return errors

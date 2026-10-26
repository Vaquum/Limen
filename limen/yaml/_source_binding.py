from collections.abc import Callable, Mapping
import inspect
import logging
from typing import TypeVar, cast

from limen.experiment._source_contract import source_owner, source_parameters, trade_dispatch
from limen.yaml._callable_binding import binding_errors
from limen.yaml.errors import ResolutionError, YAMLError
from limen.yaml.resolver import resolve

__all__: list[str] = []
logger = logging.getLogger(__name__)
_PATH = 'sfd.manifest.data_source'
_Key = TypeVar('_Key')


def source_call_errors(method: Callable[..., object], params: Mapping[_Key, object], dates: Mapping[str, object] | None,
                       *, path: str = _PATH, native_dispatch: bool = True) -> list[YAMLError]:
    if inspect.isclass(method):
        return [YAMLError(message='Source classes are unsupported; declare a supported source method', path=path + '.method')]
    if invalid := [key for key in params if not isinstance(key, str)]:
        return [YAMLError(message='Keyword names must be strings', path=f'{path}.params.{key}') for key in invalid]
    declared = cast(Mapping[str, object], params)
    try:
        values = source_parameters(method, declared, dates) if native_dispatch else dict(declared)
        owner = source_owner(method, values) if native_dispatch or trade_dispatch(method) else None
    except (TypeError, ValueError) as error:
        logger.debug('Source dispatch binding is indeterminate: %s', error)
        return []
    errors: list[YAMLError] = []
    if owner is not None:
        issues = binding_errors(owner, (), {})
        if issues:
            errors.append(YAMLError(message=f"Source owner {owner.__qualname__} cannot be constructed: " + '; '.join(issues.values()), path=path + '.method'))
    issues = binding_errors(method, (object(),) if owner is not None else (), values)
    if issues:
        errors.extend(YAMLError(message=message, path=path + '.params' + (f'.{key}' if key else '')) for key, message in issues.items())
    return errors


def _block_errors(source: object, path: str, dates: object, *, native_dispatch: bool) -> list[YAMLError]:
    if not isinstance(source, Mapping):
        return [] if source is None else [YAMLError(message='Source must be a mapping', path=path)]
    block = cast(Mapping[str, object], source)
    name, params = block.get('method'), block.get('params')
    if not isinstance(name, str):
        return [YAMLError(message='Source method must be a string', path=path + '.method')]
    if params is not None and not isinstance(params, Mapping):
        return [YAMLError(message='Source params must be a mapping', path=path + '.params')]
    try:
        method = resolve(name)
    except ResolutionError:
        return [YAMLError(message=f"Cannot resolve '{name}'", path=path + '.method')]
    if not callable(method):
        return [YAMLError(message=f"'{name}' resolves to {type(method).__name__}, not a callable", path=path + '.method')]
    return source_call_errors(method, cast(Mapping[str, object], params) if params is not None else {},
                              cast(Mapping[str, object], dates) if isinstance(dates, Mapping) else None,
                              path=path, native_dispatch=native_dispatch)


def source_errors(document: Mapping[str, object]) -> list[YAMLError]:
    sfd = document.get('sfd')
    manifest = cast(Mapping[str, object], sfd).get('manifest') if isinstance(sfd, Mapping) else None
    if not isinstance(manifest, Mapping):
        return []
    declaration = cast(Mapping[str, object], manifest)
    errors = _block_errors(declaration.get('data_source'), _PATH, declaration.get('split_dates'), native_dispatch=True)
    backtest = declaration.get('backtest')
    if isinstance(backtest, Mapping):
        settings = cast(Mapping[str, object], backtest)
        errors.extend(_block_errors(settings.get('execution_data_source'), 'sfd.manifest.backtest.execution_data_source', None, native_dispatch=False))
        funding = settings.get('funding')
        if isinstance(funding, Mapping):
            errors.extend(_block_errors(cast(Mapping[str, object], funding).get('data_source'),
                                       'sfd.manifest.backtest.funding.data_source', None, native_dispatch=False))
    return errors

from limen.yaml._callable_binding import check_transform_binding
from limen.yaml._source_binding import source_errors
from limen.yaml._backtest_spec import check_backtest_spec as _check_backtest_spec
from ruamel.yaml.scalarbool import ScalarBoolean
from limen.experiment._required_columns import column_issue
from limen._parameter_identity import choice_issue
from limen.yaml._parameter_refs import lookup_references
from limen.yaml._parameter_usage import declared_keys, used_references
from limen.experiment._walk_forward_split import read_walk_forward_config
import inspect
import math
import re
from datetime import date
from itertools import pairwise
from typing import Any, Protocol, cast

from limen.experiment.reducer.registry import REDUCER_REGISTRY
from limen.yaml.config import is_list, is_mapping
from limen.yaml.errors import ResolutionError, YAMLError
from limen.yaml.resolver import is_resolvable, resolve
from limen.yaml.schema import PRUNING_STRATEGY_OPTIONAL
from limen.yaml.schema import PRUNING_STRATEGY_REQUIRED
from limen.yaml.schema import CONDITION_REQUIRED, CONDITION_OPTIONAL

_SPLIT_DATE_KEYS = ('train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end')
_SPLIT_DATE_GUARD_KEYS = ('val_predict_guard', 'test_predict_guard')

_PARAM_REF_RE = re.compile(r'\{(\w+)\}')


def _check_callable_path(value: str, path: str, errors: list[YAMLError]) -> None:

    '''Resolve value and emit an error if it cannot be resolved or does not resolve to a callable.'''

    try:
        obj = resolve(value)
    except ResolutionError:
        errors.append(YAMLError(
            message=f"Cannot resolve '{value}'",
            path=path,
            suggestion='Path must be within an allowed limen.* namespace',
        ))
        return
    if not callable(obj):
        errors.append(YAMLError(
            message=f"'{value}' resolves to {type(obj).__name__}, not a callable or class",
            path=path,
            suggestion='Specify a callable function or class',
        ))


def _get_manifest(yaml_dict: dict[str, Any]) -> dict[str, Any]:
    sfd: dict[str, Any] = yaml_dict.get('sfd') or {}
    return sfd.get('manifest') or {}


_FUNC_BLOCK_KNOWN_KEYS = frozenset({'func', 'params'})


def _check_func_block(block: dict[str, Any],
                      base_path: str,
                      errors: list[YAMLError],
                      warnings: list[YAMLError],
                      known_keys: frozenset[str] = _FUNC_BLOCK_KNOWN_KEYS) -> None:

    '''Validate func (required, callable) and optional params mapping on a {func, params?} block.'''

    NoUnknownKeys(base_path, set(known_keys)).check_block(block, errors, warnings)

    func = block.get('func')
    if func is None:
        errors.append(YAMLError(message="Missing required field 'func'", path=f'{base_path}.func'))
        return
    if isinstance(func, str):
        _check_callable_path(func, f'{base_path}.func', errors)
    else:
        errors.append(YAMLError(
            message=f"'func' must be a string (got {type(func).__name__})",
            path=f'{base_path}.func',
        ))
    params = block.get('params')
    if params is None:
        return
    if not is_mapping(params):
        errors.append(YAMLError(
            message=f"'params' must be a mapping (got {type(params).__name__})",
            path=f'{base_path}.params',
        ))
        return
    for k, v in params.items():
        if isinstance(v, str) and v.startswith('limen.'):
            _check_callable_path(v, f'{base_path}.params.{k}', errors)


_CONDITION_LEAF_FIELDS: dict[str, tuple[str, ...]] = {
    'threshold':  ('column', 'operator', 'value'),
    'relative':   ('column', 'operator', 'other_column'),
    'crossover':  ('column', 'other_column'),
    'slope':      ('column',),
    'sql_expr':   ('expr',),
}


class Rule(Protocol):

    '''Protocol that all validation rules must satisfy.'''

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              warnings: list[YAMLError]) -> None: ...


def get_at(d: dict[str, Any], path: str) -> tuple[bool, Any]:

    '''
    Traverse a dot-notation path into a nested dict.

    Args:
        d (dict): The dict to traverse
        path (str): Dot-separated key path e.g. 'sfd.manifest.type'

    Returns:
        tuple[bool, Any]: (found, value) — found is False if any key is missing

    '''

    current: Any = d
    for part in path.split('.'):
        if not is_mapping(current) or part not in current:
            return False, None
        current = current[part]
    return True, current


_NAME_SLUG_RE = re.compile(r'^[A-Za-z0-9_-]+$')


class NameSlug:

    '''metadata.name must be a safe slug (alphanumeric, underscores, hyphens only).'''

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        found, value = get_at(yaml_dict, 'metadata.name')
        if not found or not isinstance(value, str):
            return
        if not _NAME_SLUG_RE.match(value):
            errors.append(YAMLError(
                message=f"'metadata.name' must contain only letters, digits, underscores, or hyphens (got '{value}')",
                path='metadata.name',
                suggestion='Use a safe name e.g. my_experiment_v2',
            ))


class Required:

    '''Field must exist and optionally match expected_type.'''

    def __init__(self,
                 path: str,
                 expected_type: type | None = None,
                 suggestion: str | None = None) -> None:

        super().__init__()

        self._path = path
        self._type = expected_type
        self._suggestion = suggestion

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        found, value = get_at(yaml_dict, self._path)
        label = self._path.split('.')[-1]

        if not found or value is None:
            errors.append(YAMLError(
                message=f"Missing required field '{label}'",
                path=self._path,
                suggestion=self._suggestion,
            ))
            return

        if self._type and not isinstance(value, self._type):
            errors.append(YAMLError(
                message=f"'{label}' must be a {self._type.__name__}",
                path=self._path,
            ))


class OneOf:

    '''Field value must be one of the allowed choices.'''

    def __init__(self,
                 path: str,
                 choices: set[str],
                 suggestion: str | None = None) -> None:

        super().__init__()

        self._path = path
        self._choices = choices
        self._suggestion = suggestion or f"Use one of: {', '.join(sorted(str(c) for c in choices))}"

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        found, value = get_at(yaml_dict, self._path)
        if not found or value is None:
            return
        if value not in self._choices:
            errors.append(YAMLError(
                message=f"Invalid value '{value}'",
                path=self._path,
                suggestion=self._suggestion,
            ))


class Resolvable:

    '''Field value must be an importable limen.* path that resolves to a callable or class.'''

    def __init__(self, path: str) -> None:
        super().__init__()

        self._path = path

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        found, value = get_at(yaml_dict, self._path)
        if not found or not isinstance(value, str):
            return
        _check_callable_path(value, self._path, errors)


class NoUnknownKeys:

    '''All keys at path must be in the known set; extras become warnings or errors depending on severity.'''

    def __init__(self, path: str, known: set[str], severity: str = 'error', *, rejected: dict[str, str] | None = None) -> None:

        super().__init__()

        self._path = path
        self._known = known
        self._severity = severity
        self._rejected = rejected or {}

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              warnings: list[YAMLError]) -> None:

        found, value = (True, yaml_dict) if not self._path else get_at(yaml_dict, self._path)
        if not found or not is_mapping(value):
            return
        self.check_block(value, errors, warnings)

    def check_block(self, value: dict[str, object], errors: list[YAMLError], warnings: list[YAMLError]) -> None:
        diagnosed = {error.path for error in errors}
        for key in value:
            path = f'{self._path}.{key}' if self._path else str(key)
            if path in diagnosed:
                continue
            if key in self._rejected:
                errors.append(YAMLError(message=self._rejected[key], path=path))
            elif key not in self._known:
                target = warnings if self._severity == 'warning' else errors
                target.append(YAMLError(message=f"Unknown field '{key}'", path=path,
                    suggestion='Use uel.output_path' if path == 'uel.experiment_dir' else None))


class When:

    '''Apply child rules only when condition_path equals condition_value.'''

    def __init__(self,
                 condition_path: str,
                 condition_value: Any,
                 rules: list[Rule]) -> None:

        super().__init__()

        self._condition_path = condition_path
        self._condition_value = condition_value
        self._rules = rules

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              warnings: list[YAMLError]) -> None:

        for rule in self.selected_rules(yaml_dict):
            rule.check(yaml_dict, errors, warnings)

    def selected_rules(self, yaml_dict: dict[str, object]) -> list[Rule]:
        found, value = get_at(yaml_dict, self._condition_path)
        return self._rules if found and value == self._condition_value else []


class SchemaVersion:

    '''Require the exact supported manifest schema version.'''

    def __init__(self, current_version: str) -> None:
        super().__init__()

        self._version = current_version

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        version = yaml_dict.get('schema_version')
        if not isinstance(version, str):
            errors.append(YAMLError(
                message="'schema_version' must be a string",
                path='schema_version',
                suggestion=f'Set schema_version: "{self._version}"',
            ))
            return
        if version != self._version:
            errors.append(YAMLError(
                message=f"Unsupported schema_version '{version}'; this Limen supports '{self._version}'",
                path='schema_version',
                suggestion=f'Update schema_version to "{self._version}"',
            ))


class DataSource:

    '''Validate the source; date limits belong to split_dates or walk-forward source params.'''

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        manifest = _get_manifest(yaml_dict)

        if 'test_data_source' in manifest:
            errors.append(YAMLError(
                message="'test_data_source' is not supported in YAML experiments — use the Python manifest API for programmatic test-mode data sources",
                path='sfd.manifest.test_data_source',
                suggestion='Remove test_data_source; date limits are injected from split_dates automatically',
            ))

        src = manifest.get('data_source')
        if src is None:
            errors.append(YAMLError(
                message="Missing required field 'data_source'",
                path='sfd.manifest.data_source',
            ))
            return

        if not is_mapping(src):
            errors.append(YAMLError(
                message="'data_source' must be a mapping",
                path='sfd.manifest.data_source',
            ))
            return

        if 'method' not in src:
            errors.append(YAMLError(
                message="Missing required field 'method'",
                path='sfd.manifest.data_source.method',
            ))
            return

        method = src['method']
        if not isinstance(method, str):
            errors.append(YAMLError(
                message=f"'method' must be a string (got {type(method).__name__})",
                path='sfd.manifest.data_source.method',
            ))

        params = src.get('params')
        if params is not None and not isinstance(params, dict):
            errors.append(YAMLError(
                message="'params' in 'data_source' must be a mapping",
                path='sfd.manifest.data_source.params',
            ))
            return

        if isinstance(params, dict) and 'split_walk_forward' not in manifest:
            for reserved in ('start_date_limit', 'end_date_limit'):
                if reserved in params:
                    errors.append(YAMLError(
                        message=f"'{reserved}' must not be set in 'data_source.params' — it is derived automatically from 'split_dates'",
                        path=f'sfd.manifest.data_source.params.{reserved}',
                        suggestion=f"Remove '{reserved}' from data_source.params and set the date window via split_dates instead",
                    ))

        errors.extend(source_errors(yaml_dict))


class FuncList:

    '''Validate a list of {{func, params}} dicts at a given manifest path.'''

    def __init__(self, path: str) -> None:
        super().__init__()

        self.path = path
        self.known = _FUNC_BLOCK_KNOWN_KEYS | {'include_if'}

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              warnings: list[YAMLError]) -> None:

        found, items = get_at(yaml_dict, self.path)
        if not found or items is None:
            return
        if not is_list(items):
            errors.append(YAMLError(message=f"'{self.path}' must be a list", path=self.path))
            return

        for i, item in enumerate(items):
            if not is_mapping(item):
                errors.append(YAMLError(
                    message='Each entry must be a mapping with a func key',
                    path=f'{self.path}[{i}]',
                ))
                continue
            _check_func_block(item, f'{self.path}[{i}]', errors, warnings, known_keys=self.known)
            check_transform_binding(item, f'{self.path}[{i}]', errors)
            if 'include_if' in item:
                val = item['include_if']
                if not isinstance(val, str) or not val:
                    errors.append(YAMLError(
                        message="'include_if' must be a non-empty string (a round_params key)",
                        path=f'{self.path}[{i}].include_if',
                    ))


class SingleFuncBlock:

    '''When a {func, params?} block is present, validate func is present and resolvable.'''

    def __init__(self, path: str) -> None:
        super().__init__()

        self.path = path
        self.known = _FUNC_BLOCK_KNOWN_KEYS

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              warnings: list[YAMLError]) -> None:

        found, block = get_at(yaml_dict, self.path)
        if not found or block is None:
            return
        if not is_mapping(block):
            errors.append(YAMLError(message=f"'{self.path}' must be a mapping", path=self.path))
            return
        _check_func_block(block, self.path, errors, warnings, known_keys=self.known)
        check_transform_binding(block, self.path, errors)


class RequiredColumnsSpec:

    '''Admit explicit nonempty lists of distinct nonblank column names.'''

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        manifest = _get_manifest(yaml_dict)
        if 'required_columns' not in manifest:
            return
        if issue := column_issue(manifest['required_columns']):
            index, message, _ = issue
            errors.append(YAMLError(
                message=message,
                path='sfd.manifest.required_columns' + (f'[{index}]' if index is not None else ''),
                suggestion='Use distinct nonblank column names, or omit required_columns to use the default',
            ))


class DecoderLookbackSpec:

    '''When decoder_lookback is present, it must be exactly 1 (the only supported value).'''

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        manifest = _get_manifest(yaml_dict)
        value = manifest.get('decoder_lookback')
        if value is None:
            return
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(YAMLError(
                message=f"'decoder_lookback' must be an integer (got {type(value).__name__})",
                path='sfd.manifest.decoder_lookback',
                suggestion='Set decoder_lookback: 1 (the only supported value)',
            ))
            return
        if value < 1:
            errors.append(YAMLError(
                message=f"'decoder_lookback' must be >= 1 (got {value})",
                path='sfd.manifest.decoder_lookback',
                suggestion='Set decoder_lookback: 1 (the only supported value)',
            ))
            return
        if value > 1:
            errors.append(YAMLError(
                message=f"'decoder_lookback' > 1 is not yet supported (got {value}); only decoder_lookback: 1 is supported",
                path='sfd.manifest.decoder_lookback',
                suggestion='Set decoder_lookback: 1 or omit the field (defaults to 1)',
            ))


class ConditionsList:

    '''Validate the rule_based strategy conditions list structure.'''

    def __init__(self) -> None:
        super().__init__()
        self.known = CONDITION_REQUIRED | CONDITION_OPTIONAL | {'type', 'operands', 'operator', 'direction', 'lookback'} | {key for fields in _CONDITION_LEAF_FIELDS.values() for key in fields}
        self.path = 'sfd.manifest.strategy.conditions'

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        strategy = _get_manifest(yaml_dict).get('strategy')
        if not is_mapping(strategy):
            return

        conditions = strategy.get('conditions')
        if not is_list(conditions):
            return

        for i, cond in enumerate(conditions):
            if not is_mapping(cond):
                errors.append(YAMLError(
                    message='Each condition must be a mapping',
                    path=f'sfd.manifest.strategy.conditions[{i}]',
                ))
                continue
            NoUnknownKeys(f'{self.path}[{i}]', self.known).check_block(cond, errors, _warnings)
            for key in ('id', 'name'):
                if key not in cond:
                    errors.append(YAMLError(
                        message=f"Missing required field '{key}'",
                        path=f'sfd.manifest.strategy.conditions[{i}].{key}',
                    ))
            if 'type' in cond:
                ptype = cond['type']
                if ptype not in _CONDITION_LEAF_FIELDS:
                    errors.append(YAMLError(
                        message=f"Unknown condition type '{ptype}'",
                        path=f'sfd.manifest.strategy.conditions[{i}].type',
                        suggestion=f"Use one of: {', '.join(sorted(_CONDITION_LEAF_FIELDS))}",
                    ))
                else:
                    for key in _CONDITION_LEAF_FIELDS[ptype]:
                        if key not in cond:
                            errors.append(YAMLError(
                                message=f"Missing required field '{key}' for '{ptype}' condition",
                                path=f'sfd.manifest.strategy.conditions[{i}].{key}',
                            ))
            elif 'operands' not in cond:
                errors.append(YAMLError(
                    message="Condition must have either 'type' (leaf) or 'operands' (composite)",
                    path=f'sfd.manifest.strategy.conditions[{i}]',
                ))


class PruningStrategiesSpec:

    '''Validate the uel.pruning_strategies list: reducer types and item structure.'''

    def __init__(self) -> None:
        super().__init__()
        self.path = 'uel.pruning_strategies'
        self.known = PRUNING_STRATEGY_REQUIRED | PRUNING_STRATEGY_OPTIONAL

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        uel = yaml_dict.get('uel')
        if not is_mapping(uel) or 'pruning_strategies' not in uel:
            return

        specs = uel['pruning_strategies']
        if not is_list(specs):
            errors.append(YAMLError(
                message="'pruning_strategies' must be a list",
                path='uel.pruning_strategies',
            ))
            return

        valid_types = set(REDUCER_REGISTRY)
        for i, spec in enumerate(specs):
            base = f'uel.pruning_strategies[{i}]'
            if not is_mapping(spec):
                errors.append(YAMLError(
                    message='Each pruning strategy must be a mapping',
                    path=base,
                ))
                continue
            if 'type' not in spec:
                errors.append(YAMLError(
                    message="Missing required field 'type'",
                    path=f'{base}.type',
                ))
            elif spec['type'] not in valid_types:
                errors.append(YAMLError(
                    message=f"Unknown pruning strategy type '{spec['type']}'",
                    path=f'{base}.type',
                    suggestion=f"Use one of: {', '.join(sorted(valid_types))}",
                ))
            for key in sorted(set(spec) - self.known):
                errors.append(YAMLError(message=f"Unknown field '{key}'", path=f'{base}.{key}'))
            if 'params' in spec and not is_mapping(spec['params']):
                errors.append(YAMLError(
                    message="'params' must be a mapping",
                    path=f'{base}.params',
                ))


class AcceptanceSpec:
    def check(self, yaml_dict: dict[str, object], errors: list[YAMLError], _warnings: list[YAMLError]) -> None:
        from limen.experiment._acceptance_settings import read_acceptance

        for path in ('acceptance', 'uel.acceptance'):
            found, _ = get_at(yaml_dict, path)
            if found:
                errors.append(YAMLError(message='Declare acceptance under sfd.manifest', path=path))
        found, value = get_at(yaml_dict, 'sfd.manifest.acceptance')
        if found:
            configured, _ = get_at(yaml_dict, 'sfd.manifest.split_walk_forward')
            if not configured:
                errors.append(YAMLError(message='acceptance requires split_walk_forward', path='sfd.manifest.acceptance'))
            try:
                _ = read_acceptance(value)
            except ValueError as exc:
                errors.append(YAMLError(message=str(exc), path='sfd.manifest.acceptance'))


class SplitSpec:

    '''Require exactly one absolute-date or walk-forward split declaration.'''

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        manifest = _get_manifest(yaml_dict)

        if 'split_config' in manifest:
            errors.append(YAMLError(
                message="'split_config' is not supported — use 'split_dates' with explicit date windows",
                path='sfd.manifest.split_config',
                suggestion='Replace with split_dates: {train_start: YYYY-MM-DD, train_end: YYYY-MM-DD, val_start: YYYY-MM-DD, val_end: YYYY-MM-DD, test_start: YYYY-MM-DD, test_end: YYYY-MM-DD}',
            ))
            return

        if 'split_walk_forward' in manifest:
            if 'split_dates' in manifest:
                errors.append(YAMLError(
                    message="'split_walk_forward' conflicts with 'split_dates'",
                    path='sfd.manifest.split_walk_forward',
                ))
            try:
                _ = read_walk_forward_config(manifest['split_walk_forward'])
            except ValueError as exc:
                errors.append(YAMLError(message=str(exc), path='sfd.manifest.split_walk_forward'))
            return

        if 'split_dates' not in manifest:
            errors.append(YAMLError(
                message="'split_dates' or 'split_walk_forward' is required",
                path='sfd.manifest',
                suggestion='Add split_dates: {train_start: YYYY-MM-DD, train_end: YYYY-MM-DD, val_start: YYYY-MM-DD, val_end: YYYY-MM-DD, test_start: YYYY-MM-DD, test_end: YYYY-MM-DD}',
            ))
            return

        sd = manifest['split_dates']
        if not is_mapping(sd):
            errors.append(YAMLError(
                message="'split_dates' must be a mapping",
                path='sfd.manifest.split_dates',
            ))
            return
        parsed: dict[str, date] = {}
        for key in _SPLIT_DATE_KEYS:
            if key not in sd:
                errors.append(YAMLError(
                    message=f"Missing required field '{key}'",
                    path=f'sfd.manifest.split_dates.{key}',
                ))
            elif not isinstance(sd[key], str):
                errors.append(YAMLError(
                    message=f"'{key}' must be a date string (e.g. '2022-01-01')",
                    path=f'sfd.manifest.split_dates.{key}',
                ))
            else:
                try:
                    parsed[key] = date.fromisoformat(sd[key])
                except ValueError:
                    errors.append(YAMLError(
                        message=f"'{key}' is not a valid ISO-8601 date (got '{sd[key]}')",
                        path=f'sfd.manifest.split_dates.{key}',
                        suggestion="Use format 'YYYY-MM-DD', e.g. '2022-01-01'",
                    ))
        if len(parsed) == len(_SPLIT_DATE_KEYS):
            for (a_key, a_val), (b_key, b_val) in pairwise(parsed.items()):
                if a_val > b_val:
                    errors.append(YAMLError(
                        message=f"split_dates must be non-decreasing: '{a_key}' ({a_val}) > '{b_key}' ({b_val})",
                        path='sfd.manifest.split_dates',
                        suggestion='Ensure train_start <= train_end <= val_start <= val_end <= test_start <= test_end',
                        ))

        for key in _SPLIT_DATE_GUARD_KEYS:
            if key in sd and not isinstance(sd[key], bool):
                errors.append(YAMLError(
                    message=f"'{key}' must be true or false (got {sd[key]!r})",
                    path=f'sfd.manifest.split_dates.{key}',
                    suggestion=f'Use a boolean, e.g. {key}: false',
                ))


class ScalerSpec:

    '''When scaler is present as a dict, exactly one of from_params or class must be set.'''

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        manifest = _get_manifest(yaml_dict)
        scaler = manifest.get('scaler')
        if scaler is None:
            return
        if not is_mapping(scaler):
            errors.append(YAMLError(
                message=f"'scaler' must be a mapping (got {type(scaler).__name__})",
                path='sfd.manifest.scaler',
            ))
            return
        has_from_params = 'from_params' in scaler
        has_class = 'class' in scaler
        if not has_from_params and not has_class:
            errors.append(YAMLError(
                message="'scaler' must have either 'from_params' or 'class'",
                path='sfd.manifest.scaler',
                suggestion="Use 'from_params: param_name' or 'class: limen.scalers.StandardScaler'",
            ))
        elif has_from_params and has_class:
            errors.append(YAMLError(
                message="'scaler' cannot have both 'from_params' and 'class' — use one or the other",
                path='sfd.manifest.scaler',
            ))
        else:
            key = 'from_params' if has_from_params else 'class'
            val = scaler[key]
            if not isinstance(val, str) or not val:
                errors.append(YAMLError(
                    message=f"'scaler.{key}' must be a non-empty string",
                    path=f'sfd.manifest.scaler.{key}',
                ))


class SfdParams:

    '''Every value in sfd.params must be a non-empty list.'''

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        sfd: dict[str, Any] = yaml_dict.get('sfd') or {}
        params = sfd.get('params')
        if not isinstance(params, dict):
            errors.append(YAMLError(
                message="'sfd.params' must be a mapping",
                path='sfd.params',
                suggestion='Each key maps to a list of values, e.g. lookback: [12, 24, 48]',
            ))
            return

        if not params:
            errors.append(YAMLError(
                message="'sfd.params' must not be empty",
                path='sfd.params',
                suggestion='Add at least one parameter, e.g. lookback: [12, 24, 48]',
            ))
            return

        for key, values in cast(dict[Any, Any], params).items():
            if not isinstance(key, str):
                errors.append(YAMLError(
                    message=f"Parameter name '{key}' must be a string",
                    path='sfd.params',
                    suggestion='Use a plain string key, e.g. lookback: [12, 24, 48]',
                ))
                continue
            if key.startswith('_'):
                errors.append(YAMLError(
                    message=f"Parameter name '{key}' must not start with '_'",
                    path=f'sfd.params.{key}',
                    suggestion="The '_' prefix is reserved for framework metadata. Rename the parameter.",
                ))
            if not is_list(values) or len(values) == 0:
                errors.append(YAMLError(message=f"Parameter '{key}' must be a non-empty list of values", path=f'sfd.params.{key}',
                    suggestion=f'Add at least one value, e.g. {key}: [1]'))
            elif issue := choice_issue(key, values):
                errors.append(YAMLError(message=issue[1], path=f'sfd.params.{key}[{issue[0]}]'))


class CalibrationPresence:

    '''If calibration is present, at least one of probability_calibration or threshold_function must be set.'''

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        manifest = _get_manifest(yaml_dict)
        cal = manifest.get('calibration')
        if cal is None:
            return
        if not is_mapping(cal):
            errors.append(YAMLError(
                message=f"'calibration' must be a mapping (got {type(cal).__name__})",
                path='sfd.manifest.calibration',
            ))
            return
        if cal.get('probability_calibration') is None and cal.get('threshold_function') is None:
            errors.append(YAMLError(
                message="'calibration' block is empty — add at least 'probability_calibration' or 'threshold_function'",
                path='sfd.manifest.calibration',
                suggestion='Add probability_calibration: {func: ...} or threshold_function: {func: ...}',
            ))


class CalibrationCrossRef:

    '''
    For each value in calibration params blocks:
    {param_name} references must exist in sfd.params; limen.* paths must be resolvable.
    '''

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        sfd: dict[str, Any] = yaml_dict.get('sfd') or {}
        manifest: dict[str, Any] = sfd.get('manifest') or {}
        cal = manifest.get('calibration')
        if not is_mapping(cal):
            return

        sfd_params: set[str] = set(sfd.get('params') or {})

        for section_name in ('probability_calibration', 'threshold_function'):
            section = cal.get(section_name)
            if section is None:
                continue
            if not is_mapping(section):
                errors.append(YAMLError(
                    message=f"'calibration.{section_name}' must be a mapping (got {type(section).__name__})",
                    path=f'sfd.manifest.calibration.{section_name}',
                ))
                continue

            func = section.get('func')
            if func is None:
                errors.append(YAMLError(
                    message=f"Missing required field 'func' in calibration '{section_name}'",
                    path=f'sfd.manifest.calibration.{section_name}.func',
                ))
            elif isinstance(func, str):
                _check_callable_path(func, f'sfd.manifest.calibration.{section_name}.func', errors)
            else:
                errors.append(YAMLError(
                    message=f"'func' must be a string (got {type(func).__name__})",
                    path=f'sfd.manifest.calibration.{section_name}.func',
                ))

            params = section.get('params')
            if params is None:
                continue
            if not is_mapping(params):
                errors.append(YAMLError(
                    message=f"'calibration.{section_name}.params' must be a mapping (got {type(params).__name__})",
                    path=f'sfd.manifest.calibration.{section_name}.params',
                ))
                continue

            for key, value in params.items():
                param_path = f'sfd.manifest.calibration.{section_name}.params.{key}'
                if not isinstance(value, str):
                    continue
                m = _PARAM_REF_RE.fullmatch(value.strip())
                if m:
                    ref_key = m.group(1)
                    if ref_key not in sfd_params:
                        errors.append(YAMLError(
                            message=f"Calibration param '{key}' references '{{{ref_key}}}' which is not in sfd.params",
                            path=param_path,
                            suggestion=f"Add '{ref_key}' to sfd.params",
                        ))
                elif value.startswith('limen.'):
                    _check_callable_path(value, param_path, errors)


class ParamCoverage:

    '''Require declared lookups and reject proven unused search axes.
    Opaque consumers suppress only reverse admission.
    '''

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError], *, forward: bool = True) -> None:

        sfd: dict[str, Any] = yaml_dict.get('sfd') or {}
        if not is_mapping(sfd):
            return
        sfd_param_values: dict[str, Any] = sfd.get('params') or {}
        if not is_mapping(sfd_param_values):
            return
        sfd_params = set(sfd_param_values)
        manifest: dict[str, Any] = sfd.get('manifest') or {}
        if not is_mapping(manifest):
            return

        arch_path = manifest.get('reference_architecture')
        arch_params: set[str] = set()
        open_signature = True
        if isinstance(arch_path, str) and is_resolvable(arch_path):
            component = resolve(arch_path)
            dependencies = declared_keys(component)
            arch_params.update(dependencies or ())
            try:
                sig = inspect.signature(component)
                open_signature = dependencies is None and any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                arch_params.update(name for name, p in sig.parameters.items() if name != 'data' and
                    p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY))
            except (TypeError, ValueError):
                open_signature = dependencies is None

        meta_params: set[str] = set()
        if manifest.get('type') == 'ml' and manifest.get('calibration'):
            meta_params.update({'use_calibration', 'use_threshold'})
        if manifest.get('type') == 'ml' and manifest.get('bar_formation'):
            meta_params.add('bar_type')
        if manifest.get('type') == 'ml' and isinstance(manifest.get('feature_ablation'), dict):
            fa = manifest['feature_ablation']
            meta_params.add(fa.get('drop_count_key') or 'feature_drop_count')
            meta_params.add(fa.get('seed_key') or 'feature_drop_seed')
        if manifest.get('type') == 'ml' and isinstance(manifest.get('scaler'), dict):
            fp = manifest['scaler'].get('from_params')
            if isinstance(fp, str):
                meta_params.add(fp)
        if manifest.get('type') == 'ml' and isinstance(manifest.get('pca_compression'), dict):
            pca = manifest['pca_compression']
            meta_params.add(pca.get('enabled_param') or 'auto_pca')
            meta_params.add(pca.get('n_components_param') or 'pca_k')

        include_if_keys: set[str] = set()
        for field in ('indicators', 'features'):
            items: object = manifest.get(field) or []
            if not is_list(items):
                open_signature = True
                continue
            for item in items:
                open_signature = open_signature or not is_mapping(item)
                if is_mapping(item) and isinstance(item.get('include_if'), str):
                    include_if_keys.add(item['include_if'])
                if is_mapping(item) and is_mapping(item.get('params')) and item['params'].get('group') is not None:
                    meta_params.add('feature_groups')
        meta_params.update(include_if_keys)

        if forward:
            for key in sorted(include_if_keys):
                if key not in sfd_params:
                    errors.append(YAMLError(
                        message=f"'include_if: {key}' references '{key}' but '{key}' is not defined in sfd.params",
                        path='sfd.manifest',
                        suggestion=f"Add '{key}' to sfd.params",
                    ))
                else:
                    values: list[Any] = sfd_param_values.get(key) or []
                    if not all(isinstance(v, bool) for v in values):
                        errors.append(YAMLError(
                            message=f"'include_if: {key}' — sfd.params['{key}'] must contain only boolean values (true/false)",
                            path=f'sfd.params.{key}',
                            suggestion=f"Change '{key}' values to [true] or [true, false]",
                        ))

            for ref, path in sorted(lookup_references(manifest, sfd_params).items()):
                if ref not in sfd_params:
                    errors.append(YAMLError(
                        message=f"Manifest references '{{{ref}}}' but '{ref}' is not defined in sfd.params",
                        path=path,
                        suggestion=f"Add '{ref}' to sfd.params" + (f"; this also overrides architecture '{ref}', or use a distinct parameter name" if ref in arch_params else ''),
                    ))

        if open_signature:
            return
        valid = used_references(manifest, sfd_params) | arch_params | meta_params

        for key in sorted(sfd_params, key=str):
            if key not in valid and is_list(sfd_param_values[key]):
                errors.append(YAMLError(
                    message=f"Unused parameter '{key}': {len(sfd_param_values[key])} declared choice(s) have no consuming binding",
                    path=f'sfd.params.{key}',
                    suggestion='Remove it or bind it to an executed component; use uel.search_strategy.seed for traversal randomness',
                ))

    def unused_errors(self, document: dict[str, object]) -> list[YAMLError]:
        errors: list[YAMLError] = []
        self.check(document, errors, [], forward=False)
        return errors


class BlockSpec:

    '''Emit an error for each listed path that is present but not a mapping.'''

    def __init__(self, *paths: str) -> None:
        super().__init__()

        self._paths = paths

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        for path in self._paths:
            found, value = get_at(yaml_dict, path)
            if found and value is not None and not isinstance(value, dict):
                errors.append(YAMLError(
                    message=f"'{path}' must be a mapping (got {type(value).__name__})",
                    path=path,
                ))


class ParamKeyFields:

    '''Validate that param-key fields within a block are non-empty strings when present.'''

    def __init__(self, path: str, *fields: str) -> None:
        super().__init__()

        self._path = path
        self._fields = fields

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        found, block = get_at(yaml_dict, self._path)
        if not found or not is_mapping(block):
            return
        for field_name in self._fields:
            if field_name not in block:
                continue
            value = block[field_name]
            if not isinstance(value, str) or not value:
                errors.append(YAMLError(
                    message=f"'{field_name}' must be a non-empty string",
                    path=f'{self._path}.{field_name}',
                    suggestion=f"Use a param name string, e.g. {field_name}: my_param_name",
                ))


class BacktestCostSpec:

    '''Each backtest knob is a number (fee_bps/slip_bps non-negative; notional_rate in (0, 1]) or a {param} reference into sfd.params.'''

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        _check_backtest_spec(yaml_dict, errors)


_UEL_INTERVALS = ('feedback_interval', 'checkpoint_interval')
_UEL_TYPE_CHECKS: list[tuple[str, type, str | None]] = [
    ('workers', int, None),
    ('record_execution', bool, None),
    ('record_model_outputs', bool, None),
    ('factorize', bool, None),
    ('search_strategy', dict, 'Use search_strategy:\n  type: random'),
    ('output_path', str, None),
    ('prep_each_round', bool, None),
    ('feedback_interval', int, None),
    ('checkpoint_interval', int, None),
]


class UelSpec:

    '''Check optional uel types and positive execution intervals.'''

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        uel: dict[str, Any] = yaml_dict.get('uel') or {}
        if 'n_permutations' in uel:
            value = uel['n_permutations']
            if not isinstance(value, int) or isinstance(value, (bool, ScalarBoolean)):
                errors.append(YAMLError(
                    message=f"'uel.n_permutations' must be an int (got {type(value).__name__})",
                    path='uel.n_permutations',
                ))
            elif value <= 0:
                errors.append(YAMLError(
                    message=f"'uel.n_permutations' must be a positive integer (got {value})",
                    path='uel.n_permutations',
                ))
            else:
                sfd: dict[str, Any] = yaml_dict.get('sfd') or {}
                params = sfd.get('params')
                if is_mapping(params) and params and all(isinstance(v, list) for v in params.values()):
                    total = math.prod(len(v) for v in params.values())
                    if total > 0 and value > total:
                        errors.append(YAMLError(
                            message=(
                                f"'uel.n_permutations' ({value}) cannot exceed the available parameter space ({total})"
                            ),
                            path='uel.n_permutations',
                        ))
        for field, expected, suggestion in _UEL_TYPE_CHECKS:
            if field not in uel:
                continue
            value = uel[field]
            if not isinstance(value, expected) or (expected is int and isinstance(value, (bool, ScalarBoolean))):
                errors.append(YAMLError(
                    message=f"'uel.{field}' must be a {expected.__name__} (got {type(value).__name__})",
                    path=f'uel.{field}',
                    suggestion=suggestion,
                ))
            elif (field == 'workers' or field in _UEL_INTERVALS) and value <= 0:
                errors.append(YAMLError(
                    message=f"'uel.{field}' must be a positive integer",
                    path=f'uel.{field}',
                ))

        strategy = uel.get('search_strategy')
        if is_mapping(strategy):
            seed = strategy.get('seed')
            if strategy.get('type') == 'sobol':
                if set(strategy) - {'type', 'seed'}:
                    errors.append(YAMLError(message="Sobol search supports only 'type' and 'seed'", path='uel.search_strategy'))
                if type(seed) is not int or seed < 0:
                    errors.append(YAMLError(message='Sobol search requires a nonnegative integer seed', path='uel.search_strategy.seed'))
            shuffle = strategy.get('shuffle', False)
            if 'shuffle' in strategy and strategy.get('type', 'random') != 'grid':
                errors.append(YAMLError(message="'uel.search_strategy.shuffle' is only supported for grid search", path='uel.search_strategy.shuffle'))
            if type(shuffle) is not bool:
                errors.append(YAMLError(message="'uel.search_strategy.shuffle' must be a bool", path='uel.search_strategy.shuffle'))
            if (seed is not None or (shuffle is True and strategy.get('type') == 'grid')) and type(seed) is not int:
                errors.append(YAMLError(
                    message=f"'uel.search_strategy.seed' must be an int (got {type(seed).__name__})",
                    path='uel.search_strategy.seed',
                ))


class RuleEngine:

    '''Run a list of rules against a yaml_dict, collecting errors and warnings.'''

    def __init__(self, rules: list[Rule]) -> None:
        super().__init__()

        self._rules = rules

    def run(self,
            yaml_dict: dict[str, Any],
            errors: list[YAMLError],
            warnings: list[YAMLError]) -> None:

        for rule in self._rules:
            rule.check(yaml_dict, errors, warnings)

    def unknown_errors(self, document: dict[str, object]) -> list[YAMLError]:
        '''Apply only the existing structural key contracts to a saved declaration.'''
        errors: list[YAMLError] = []
        warnings: list[YAMLError] = []

        def visit(rules: list[Rule]) -> None:
            for rule in rules:
                if isinstance(rule, When):
                    visit(rule.selected_rules(document))
                elif isinstance(rule, NoUnknownKeys):
                    rule.check(document, errors, warnings)
                elif isinstance(rule, (FuncList, ConditionsList, SingleFuncBlock, PruningStrategiesSpec)):
                    found, value = get_at(document, rule.path)
                    if not found:
                        continue
                    blocks: list[tuple[str, object]] = [(rule.path, value)]
                    if isinstance(rule, (FuncList, ConditionsList, PruningStrategiesSpec)):
                        blocks = [(f'{rule.path}[{index}]', block) for index, block in enumerate(cast(list[object], value))] if isinstance(value, list) else []
                    for path, block in blocks:
                        if is_mapping(block):
                            NoUnknownKeys(path, set(rule.known)).check_block(block, errors, warnings)

        visit(self._rules)
        return errors

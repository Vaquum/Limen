import inspect
import re
from datetime import date
from itertools import pairwise
from typing import Any
from typing import Protocol

from limen.yaml.errors import ResolutionError
from limen.yaml.errors import YAMLError
from limen.yaml.resolver import is_resolvable
from limen.yaml.resolver import resolve

_SPLIT_DATE_KEYS = ('train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end')

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
    return (yaml_dict.get('sfd') or {}).get('manifest') or {}


_FUNC_BLOCK_KNOWN_KEYS = frozenset({'func', 'params'})


def _check_func_block(block: dict[str, Any],
                      base_path: str,
                      errors: list[YAMLError],
                      warnings: list[YAMLError],
                      extra_known_keys: frozenset[str] = frozenset()) -> None:

    '''Validate func (required, callable) and optional params mapping on a {func, params?} block.'''

    known = _FUNC_BLOCK_KNOWN_KEYS | extra_known_keys
    for key in block:
        if key not in known:
            warnings.append(YAMLError(
                message=f"Unknown field '{key}'",
                path=f'{base_path}.{key}',
            ))

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
    if not isinstance(params, dict):
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
        if not isinstance(current, dict) or part not in current:
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
                 choices: set,
                 suggestion: str | None = None) -> None:

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

    def __init__(self, path: str, known: set, severity: str = 'warning') -> None:

        self._path = path
        self._known = known
        self._severity = severity

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              warnings: list[YAMLError]) -> None:

        found, value = get_at(yaml_dict, self._path)
        if not found or not isinstance(value, dict):
            return
        for key in value:
            if key not in self._known:
                target = warnings if self._severity == 'warning' else errors
                target.append(YAMLError(
                    message=f"Unknown field '{key}'",
                    path=f"{self._path}.{key}",
                ))


class When:

    '''Apply child rules only when condition_path equals condition_value.'''

    def __init__(self,
                 condition_path: str,
                 condition_value: Any,
                 rules: list[Rule]) -> None:

        self._condition_path = condition_path
        self._condition_value = condition_value
        self._rules = rules

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              warnings: list[YAMLError]) -> None:

        found, value = get_at(yaml_dict, self._condition_path)
        if found and value == self._condition_value:
            for rule in self._rules:
                rule.check(yaml_dict, errors, warnings)


class WarnIfPresent:

    '''Emit a warning for each key in keys that exists at path.'''

    def __init__(self, path: str, keys: set, message_template: str) -> None:

        self._path = path
        self._keys = keys
        self._message_template = message_template

    def check(self,
              yaml_dict: dict[str, Any],
              _errors: list[YAMLError],
              warnings: list[YAMLError]) -> None:

        found, value = get_at(yaml_dict, self._path)
        if not found or not isinstance(value, dict):
            return
        for key in self._keys:
            if key in value:
                warnings.append(YAMLError(
                    message=self._message_template.format(key=key),
                    path=f"{self._path}.{key}",
                ))


class SchemaVersion:

    '''schema_version must be a string; warn if it differs from current_version.'''

    def __init__(self, current_version: str) -> None:
        self._version = current_version

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              warnings: list[YAMLError]) -> None:

        version = yaml_dict.get('schema_version')
        if not isinstance(version, str):
            errors.append(YAMLError(
                message="'schema_version' must be a string",
                path='schema_version',
                suggestion=f'Set schema_version: "{self._version}"',
            ))
            return
        if version != self._version:
            warnings.append(YAMLError(
                message=f"schema_version '{version}' != current version '{self._version}'",
                path='schema_version',
                suggestion=f'Update schema_version to "{self._version}"',
            ))


class DataSource:

    '''Validate data_source (required) and test_data_source (optional) blocks.'''

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        manifest = _get_manifest(yaml_dict)

        for section, required in (('data_source', True), ('test_data_source', False)):
            src = manifest.get(section)
            if src is None:
                if required:
                    errors.append(YAMLError(
                        message=f"Missing required field '{section}'",
                        path=f'sfd.manifest.{section}',
                    ))
                continue

            if not isinstance(src, dict):
                errors.append(YAMLError(
                    message=f"'{section}' must be a mapping",
                    path=f'sfd.manifest.{section}',
                ))
                continue

            if 'method' not in src:
                errors.append(YAMLError(
                    message="Missing required field 'method'",
                    path=f'sfd.manifest.{section}.method',
                ))
                continue

            method = src['method']
            if isinstance(method, str):
                _check_callable_path(method, f'sfd.manifest.{section}.method', errors)
            else:
                errors.append(YAMLError(
                    message=f"'method' must be a string (got {type(method).__name__})",
                    path=f'sfd.manifest.{section}.method',
                ))

            params = src.get('params')
            if params is not None and not isinstance(params, dict):
                errors.append(YAMLError(
                    message=f"'params' in '{section}' must be a mapping",
                    path=f'sfd.manifest.{section}.params',
                ))


class FuncList:

    '''Validate a list of {{func, params}} dicts at a given manifest path.'''

    def __init__(self, path: str) -> None:
        self._path = path

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              warnings: list[YAMLError]) -> None:

        found, items = get_at(yaml_dict, self._path)
        if not found or items is None:
            return
        if not isinstance(items, list):
            errors.append(YAMLError(message=f"'{self._path}' must be a list", path=self._path))
            return

        for i, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(YAMLError(
                    message='Each entry must be a mapping with a func key',
                    path=f'{self._path}[{i}]',
                ))
                continue
            _check_func_block(item, f'{self._path}[{i}]', errors, warnings,
                               extra_known_keys=frozenset({'include_if'}))
            if 'include_if' in item:
                val = item['include_if']
                if not isinstance(val, str) or not val:
                    errors.append(YAMLError(
                        message="'include_if' must be a non-empty string (a round_params key)",
                        path=f'{self._path}[{i}].include_if',
                    ))


class SingleFuncBlock:

    '''When a {func, params?} block is present, validate func is present and resolvable.'''

    def __init__(self, path: str) -> None:
        self._path = path

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              warnings: list[YAMLError]) -> None:

        found, block = get_at(yaml_dict, self._path)
        if not found or block is None:
            return
        if not isinstance(block, dict):
            errors.append(YAMLError(message=f"'{self._path}' must be a mapping", path=self._path))
            return
        _check_func_block(block, self._path, errors, warnings)


class RequiredColumnsSpec:

    '''When required_columns is present, it must be a non-empty list of strings.'''

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        manifest = _get_manifest(yaml_dict)
        cols = manifest.get('required_columns')
        if cols is None:
            return
        if not isinstance(cols, list):
            errors.append(YAMLError(
                message=f"'required_columns' must be a list (got {type(cols).__name__})",
                path='sfd.manifest.required_columns',
                suggestion="Use a list of column name strings, e.g. required_columns: [open, high, low, close]",
            ))
            return
        for i, item in enumerate(cols):
            if not isinstance(item, str):
                errors.append(YAMLError(
                    message=f"'required_columns[{i}]' must be a string (got {type(item).__name__})",
                    path=f'sfd.manifest.required_columns[{i}]',
                ))


class ConditionsList:

    '''Validate the rule_based strategy conditions list structure.'''

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        strategy = ((yaml_dict.get('sfd') or {}).get('manifest') or {}).get('strategy')
        if not isinstance(strategy, dict):
            return

        conditions = strategy.get('conditions')
        if not isinstance(conditions, list):
            return

        for i, cond in enumerate(conditions):
            if not isinstance(cond, dict):
                errors.append(YAMLError(
                    message='Each condition must be a mapping',
                    path=f'sfd.manifest.strategy.conditions[{i}]',
                ))
                continue
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


class SplitSpec:

    '''Exactly one of split_config (ratio) or split_dates (absolute) must be present.'''

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        manifest = _get_manifest(yaml_dict)
        has_config = 'split_config' in manifest
        has_dates = 'split_dates' in manifest

        if has_config and has_dates:
            errors.append(YAMLError(
                message="Cannot specify both 'split_config' and 'split_dates' — use one or the other",
                path='sfd.manifest',
                suggestion="Remove 'split_config' to use absolute-date splits, or remove 'split_dates' to use ratio splits",
            ))
            return

        if not has_config and not has_dates:
            errors.append(YAMLError(
                message="One of 'split_config' or 'split_dates' is required",
                path='sfd.manifest',
                suggestion='Add split_config: {train: 8, val: 1, test: 2} or split_dates: {train_start: ..., ...}',
            ))
            return

        if has_config:
            sc = manifest['split_config']
            if not isinstance(sc, dict):
                errors.append(YAMLError(
                    message="'split_config' must be a mapping",
                    path='sfd.manifest.split_config',
                ))
                return
            for key in ('train', 'val', 'test'):
                if key not in sc:
                    errors.append(YAMLError(
                        message=f"Missing required field '{key}'",
                        path=f'sfd.manifest.split_config.{key}',
                    ))
                elif not isinstance(sc[key], int):
                    errors.append(YAMLError(
                        message=f"'{key}' must be an int",
                        path=f'sfd.manifest.split_config.{key}',
                    ))
                elif key == 'train' and sc[key] <= 0:
                    errors.append(YAMLError(
                        message=f"'train' must be > 0 (got {sc[key]})",
                        path='sfd.manifest.split_config.train',
                        suggestion='Use a positive integer e.g. train: 8',
                    ))
                elif key in ('val', 'test') and sc[key] < 0:
                    errors.append(YAMLError(
                        message=f"'{key}' must be >= 0 (got {sc[key]})",
                        path=f'sfd.manifest.split_config.{key}',
                    ))
        else:
            sd = manifest['split_dates']
            if not isinstance(sd, dict):
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
        if not isinstance(scaler, dict):
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

        params = (yaml_dict.get('sfd') or {}).get('params')
        if not isinstance(params, dict):
            errors.append(YAMLError(
                message="'sfd.params' must be a mapping",
                path='sfd.params',
                suggestion='Each key maps to a list of values, e.g. lookback: [12, 24, 48]',
            ))
            return

        for key, values in params.items():
            if not isinstance(values, list) or len(values) == 0:
                errors.append(YAMLError(
                    message=f"Parameter '{key}' must be a non-empty list of values",
                    path=f'sfd.params.{key}',
                    suggestion=f'Add at least one value, e.g. {key}: [1]',
                ))


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
        if not isinstance(cal, dict):
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

        sfd = yaml_dict.get('sfd') or {}
        manifest = (sfd.get('manifest') or {})
        cal = manifest.get('calibration')
        if not isinstance(cal, dict):
            return

        sfd_params = set(sfd.get('params') or {})

        for section_name in ('probability_calibration', 'threshold_function'):
            section = cal.get(section_name)
            if section is None:
                continue
            if not isinstance(section, dict):
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
            if not isinstance(params, dict):
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

    '''
    Cross-validate sfd.params against manifest {param_name} references.

    Forward check (errors): every {ref} in the manifest must be defined in sfd.params.
    Reverse check (warnings): every key in sfd.params must be referenced in the manifest,
    architecture signature, or a meta-param implied by a manifest block.

    NOTE: If the architecture accepts **kwargs coverage cannot be determined — check is skipped.
    '''

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              warnings: list[YAMLError]) -> None:

        sfd = yaml_dict.get('sfd') or {}
        sfd_params = set((sfd.get('params') or {}).keys())
        manifest = sfd.get('manifest') or {}
        manifest_refs = self._extract_param_refs(manifest)

        if not sfd_params and not manifest_refs:
            return

        arch_path = manifest.get('reference_architecture')
        arch_params: set[str] = set()
        if isinstance(arch_path, str) and is_resolvable(arch_path):
            try:
                sig = inspect.signature(resolve(arch_path))
                for name, param in sig.parameters.items():
                    if name == 'data':
                        continue
                    if param.kind == inspect.Parameter.VAR_KEYWORD:
                        return
                    arch_params.add(name)
            except (TypeError, ValueError):
                return

        meta_params: set[str] = set()
        if manifest.get('calibration'):
            meta_params.update({'use_calibration', 'use_threshold'})
        if manifest.get('indicators') or manifest.get('features'):
            meta_params.add('feature_groups')
        if isinstance(manifest.get('feature_ablation'), dict):
            fa = manifest['feature_ablation']
            meta_params.add(fa.get('drop_count_key') or 'feature_drop_count')
            meta_params.add(fa.get('seed_key') or 'feature_drop_seed')
        if isinstance(manifest.get('scaler'), dict):
            fp = manifest['scaler'].get('from_params')
            if isinstance(fp, str):
                meta_params.add(fp)
        if isinstance(manifest.get('pca_compression'), dict):
            pca = manifest['pca_compression']
            meta_params.add(pca.get('enabled_param') or 'auto_pca')
            meta_params.add(pca.get('n_components_param') or 'pca_k')

        for ref in sorted(manifest_refs):
            if ref not in sfd_params and ref not in arch_params and ref not in meta_params:
                errors.append(YAMLError(
                    message=f"Manifest references '{{{ref}}}' but '{ref}' is not defined in sfd.params",
                    path='sfd.manifest',
                    suggestion=f"Add '{ref}' to sfd.params",
                ))

        valid = manifest_refs | arch_params | meta_params

        for key in sfd_params:
            if key not in valid:
                warnings.append(YAMLError(
                    message=f"Param '{key}' in sfd.params is not referenced in the manifest or architecture",
                    path=f'sfd.params.{key}',
                    suggestion=f"Remove it or add a {{{key}}} reference in the manifest",
                ))

    @staticmethod
    def _extract_param_refs(obj: Any) -> set[str]:

        refs: set[str] = set()
        ParamCoverage._walk_refs(obj, refs)
        return refs

    @staticmethod
    def _walk_refs(obj: Any, refs: set[str]) -> None:

        if isinstance(obj, str):
            for match in _PARAM_REF_RE.finditer(obj):
                refs.add(match.group(1))
        elif isinstance(obj, dict):
            for v in obj.values():
                ParamCoverage._walk_refs(v, refs)
        elif isinstance(obj, list):
            for item in obj:
                ParamCoverage._walk_refs(item, refs)


class BlockSpec:

    '''Emit an error for each listed path that is present but not a mapping.'''

    def __init__(self, *paths: str) -> None:
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
        self._path = path
        self._fields = fields

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        found, block = get_at(yaml_dict, self._path)
        if not found or not isinstance(block, dict):
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


_UEL_TYPE_CHECKS: list[tuple[str, type, str | None]] = [
    ('search_strategy',    dict, 'Use search_strategy:\n  type: random'),
    ('output_path',        str,  None),
    ('prep_each_round',    bool, None),
    ('feedback_interval',  int,  None),
    ('checkpoint_interval', int, None),
]


class UelSpec:

    '''Type checks for optional uel fields that the CLI assumes are specific types.'''

    def check(self,
              yaml_dict: dict[str, Any],
              errors: list[YAMLError],
              _warnings: list[YAMLError]) -> None:

        uel = yaml_dict.get('uel') or {}
        for field, expected, suggestion in _UEL_TYPE_CHECKS:
            if field not in uel:
                continue
            value = uel[field]
            if not isinstance(value, expected):
                errors.append(YAMLError(
                    message=f"'uel.{field}' must be a {expected.__name__} (got {type(value).__name__})",
                    path=f'uel.{field}',
                    suggestion=suggestion,
                ))


class RuleEngine:

    '''Run a list of rules against a yaml_dict, collecting errors and warnings.'''

    def __init__(self, rules: list[Rule]) -> None:
        self._rules = rules

    def run(self,
            yaml_dict: dict[str, Any],
            errors: list[YAMLError],
            warnings: list[YAMLError]) -> None:

        for rule in self._rules:
            rule.check(yaml_dict, errors, warnings)

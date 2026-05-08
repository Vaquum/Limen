from dataclasses import dataclass
from dataclasses import field
from typing import Any

from limen.yaml.errors import YAMLError
from limen.yaml.resolver import is_resolvable
from limen.yaml.schema import DATA_SOURCE_REQUIRED
from limen.yaml.schema import MANIFEST_OPTIONAL_SHARED
from limen.yaml.schema import MANIFEST_REQUIRED
from limen.yaml.schema import METADATA_OPTIONAL
from limen.yaml.schema import METADATA_REQUIRED
from limen.yaml.schema import ML_MANIFEST_OPTIONAL
from limen.yaml.schema import ML_MANIFEST_REQUIRED
from limen.yaml.schema import RULE_BASED_MANIFEST_OPTIONAL
from limen.yaml.schema import RULE_BASED_MANIFEST_REQUIRED
from limen.yaml.schema import SFD_REQUIRED
from limen.yaml.schema import SPLIT_CONFIG_REQUIRED
from limen.yaml.schema import STRATEGY_REQUIRED
from limen.yaml.schema import TARGET_OPTIONAL
from limen.yaml.schema import TARGET_REQUIRED
from limen.yaml.schema import TOP_LEVEL_REQUIRED
from limen.yaml.schema import UEL_OPTIONAL
from limen.yaml.schema import UEL_REQUIRED
from limen.yaml.schema import VALID_MANIFEST_TYPES
from limen.yaml.schema import VALID_MODES
from limen.yaml.schema import VALID_OUTPUT_FORMATS
from limen.yaml.schema import VALID_SEARCH_STRATEGY_TYPES
from limen.yaml.schema import VERSION


@dataclass
class ValidationResult:

    '''Result of a YAML validation pass.'''

    valid: bool
    errors: list[YAMLError] = field(default_factory=list)
    warnings: list[YAMLError] = field(default_factory=list)
    mode: str = 'development'


def validate(yaml_dict: dict[str, Any]) -> ValidationResult:

    '''
    Validate a parsed YAML experiment dict.

    Collects all errors without fail-fast. Returns a ValidationResult
    with valid=True only when no errors are present.

    Args:
        yaml_dict (dict): Parsed YAML dict from parser.parse()

    Returns:
        ValidationResult: Validation outcome with errors and warnings

    '''

    errors: list[YAMLError] = []
    warnings: list[YAMLError] = []

    _check_top_level(yaml_dict, errors)
    if errors:
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    _check_schema_version(yaml_dict, errors, warnings)
    mode = _check_metadata(yaml_dict, errors, warnings)
    _check_sfd(yaml_dict, errors, warnings)
    _check_uel(yaml_dict, errors, warnings)

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        mode=mode,
    )


def _check_top_level(d: dict[str, Any], errors: list[YAMLError]) -> None:

    for key in TOP_LEVEL_REQUIRED:
        if key not in d:
            errors.append(YAMLError(
                message=f"Missing required field '{key}'",
                path=key,
                suggestion=f"Add '{key}' to the top level of your YAML file",
            ))


def _check_schema_version(d: dict[str, Any],
                           errors: list[YAMLError],
                           warnings: list[YAMLError]) -> None:

    version = d.get('schema_version')
    if not isinstance(version, str):
        errors.append(YAMLError(
            message="'schema_version' must be a string",
            path='schema_version',
            suggestion=f'Set schema_version: "{VERSION}"',
        ))
        return

    if version != VERSION:
        warnings.append(YAMLError(
            message=f"schema_version '{version}' != current version '{VERSION}'",
            path='schema_version',
            suggestion=f'Update schema_version to "{VERSION}"',
        ))


def _check_metadata(d: dict[str, Any],
                    errors: list[YAMLError],
                    warnings: list[YAMLError]) -> str:

    meta = d.get('metadata')
    if not isinstance(meta, dict):
        errors.append(YAMLError(
            message="'metadata' must be a mapping",
            path='metadata',
        ))
        return 'development'

    for key in METADATA_REQUIRED:
        if key not in meta:
            errors.append(YAMLError(
                message=f"Missing required field '{key}'",
                path=f'metadata.{key}',
            ))

    known = METADATA_REQUIRED | METADATA_OPTIONAL
    for key in meta:
        if key not in known:
            warnings.append(YAMLError(
                message=f"Unknown metadata field '{key}'",
                path=f'metadata.{key}',
            ))

    mode = meta.get('mode', 'development')
    if mode not in VALID_MODES:
        errors.append(YAMLError(
            message=f"Invalid mode '{mode}'",
            path='metadata.mode',
            suggestion=f"Use one of: {', '.join(sorted(VALID_MODES))}",
        ))

    return str(mode) if mode in VALID_MODES else 'development'


def _check_sfd(d: dict[str, Any],
               errors: list[YAMLError],
               warnings: list[YAMLError]) -> None:

    sfd = d.get('sfd')
    if not isinstance(sfd, dict):
        errors.append(YAMLError(message="'sfd' must be a mapping", path='sfd'))
        return

    for key in SFD_REQUIRED:
        if key not in sfd:
            errors.append(YAMLError(
                message=f"Missing required field '{key}'",
                path=f'sfd.{key}',
            ))

    if 'manifest' in sfd:
        _check_manifest(sfd['manifest'], errors, warnings)

    if 'params' in sfd:
        _check_params(sfd['params'], errors)


def _check_manifest(manifest: Any,
                    errors: list[YAMLError],
                    warnings: list[YAMLError]) -> None:

    if not isinstance(manifest, dict):
        errors.append(YAMLError(message="'sfd.manifest' must be a mapping", path='sfd.manifest'))
        return

    for key in MANIFEST_REQUIRED:
        if key not in manifest:
            errors.append(YAMLError(
                message=f"Missing required field '{key}'",
                path=f'sfd.manifest.{key}',
            ))

    manifest_type = manifest.get('type')
    if manifest_type not in VALID_MANIFEST_TYPES:
        errors.append(YAMLError(
            message=f"Invalid manifest type '{manifest_type}'",
            path='sfd.manifest.type',
            suggestion=f"Use one of: {', '.join(sorted(VALID_MANIFEST_TYPES))}",
        ))
        return

    _check_data_source(manifest, errors)
    _check_split_config(manifest, errors)
    _check_reference_architecture(manifest, errors)

    if manifest_type == 'ml':
        _check_ml_manifest(manifest, errors, warnings)
    elif manifest_type == 'rule_based':
        _check_rule_based_manifest(manifest, errors, warnings)

    known = (
        MANIFEST_REQUIRED
        | MANIFEST_OPTIONAL_SHARED
        | (ML_MANIFEST_REQUIRED | ML_MANIFEST_OPTIONAL if manifest_type == 'ml'
           else RULE_BASED_MANIFEST_REQUIRED | RULE_BASED_MANIFEST_OPTIONAL)
    )
    for key in manifest:
        if key not in known:
            warnings.append(YAMLError(
                message=f"Unknown manifest field '{key}'",
                path=f'sfd.manifest.{key}',
            ))


def _check_data_source(manifest: dict[str, Any], errors: list[YAMLError]) -> None:

    for section in ('data_source', 'test_data_source'):
        src = manifest.get(section)
        if src is None:
            if section == 'data_source':
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

        for key in DATA_SOURCE_REQUIRED:
            if key not in src:
                errors.append(YAMLError(
                    message=f"Missing required field '{key}'",
                    path=f'sfd.manifest.{section}.{key}',
                ))

        method = src.get('method')
        if isinstance(method, str) and not is_resolvable(method):
            errors.append(YAMLError(
                message=f"Cannot resolve data source method '{method}'",
                path=f'sfd.manifest.{section}.method',
                suggestion='Path must be within an allowed limen.* namespace',
            ))


def _check_split_config(manifest: dict[str, Any], errors: list[YAMLError]) -> None:

    split = manifest.get('split_config')
    if not isinstance(split, dict):
        errors.append(YAMLError(
            message="'split_config' must be a mapping",
            path='sfd.manifest.split_config',
        ))
        return

    for key in SPLIT_CONFIG_REQUIRED:
        if key not in split:
            errors.append(YAMLError(
                message=f"Missing required field '{key}'",
                path=f'sfd.manifest.split_config.{key}',
            ))
        elif not isinstance(split[key], int):
            errors.append(YAMLError(
                message=f"'{key}' must be an integer",
                path=f'sfd.manifest.split_config.{key}',
            ))


def _check_reference_architecture(manifest: dict[str, Any], errors: list[YAMLError]) -> None:

    arch = manifest.get('reference_architecture')
    if not isinstance(arch, str):
        errors.append(YAMLError(
            message="'reference_architecture' must be a string path",
            path='sfd.manifest.reference_architecture',
            suggestion="e.g. 'limen.sfd.reference_architecture.logreg_binary'",
        ))
        return

    if not is_resolvable(arch):
        errors.append(YAMLError(
            message=f"Cannot resolve reference_architecture '{arch}'",
            path='sfd.manifest.reference_architecture',
            suggestion='Path must be within an allowed limen.* namespace',
        ))


def _check_ml_manifest(manifest: dict[str, Any],
                        errors: list[YAMLError],
                        _warnings: list[YAMLError]) -> None:

    for key in ML_MANIFEST_REQUIRED:
        if key not in manifest:
            errors.append(YAMLError(
                message=f"Missing required field '{key}' for type 'ml'",
                path=f'sfd.manifest.{key}',
            ))

    if 'target' in manifest:
        _check_target(manifest['target'], errors)

    for section in ('indicators', 'features'):
        items = manifest.get(section)
        if items is not None:
            _check_func_list(items, f'sfd.manifest.{section}', errors)


def _check_target(target: Any, errors: list[YAMLError]) -> None:

    if not isinstance(target, dict):
        errors.append(YAMLError(message="'target' must be a mapping", path='sfd.manifest.target'))
        return

    for key in TARGET_REQUIRED:
        if key not in target:
            errors.append(YAMLError(
                message=f"Missing required field '{key}'",
                path=f'sfd.manifest.target.{key}',
            ))

    known = TARGET_REQUIRED | TARGET_OPTIONAL
    for key in target:
        if key not in known:
            errors.append(YAMLError(
                message=f"Unknown target field '{key}'",
                path=f'sfd.manifest.target.{key}',
            ))

    cls = target.get('class')
    if isinstance(cls, str) and not is_resolvable(cls):
        errors.append(YAMLError(
            message=f"Cannot resolve target class '{cls}'",
            path='sfd.manifest.target.class',
            suggestion='Path must be within an allowed limen.* namespace',
        ))


def _check_func_list(items: Any, path: str, errors: list[YAMLError]) -> None:

    if not isinstance(items, list):
        errors.append(YAMLError(message=f"'{path}' must be a list", path=path))
        return

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(YAMLError(
                message='Each entry must be a mapping with a func key',
                path=f'{path}[{i}]',
            ))
            continue
        if 'func' not in item:
            errors.append(YAMLError(
                message="Missing required field 'func'",
                path=f'{path}[{i}].func',
            ))
            continue
        func = item['func']
        if isinstance(func, str) and not is_resolvable(func):
            errors.append(YAMLError(
                message=f"Cannot resolve func '{func}'",
                path=f'{path}[{i}].func',
                suggestion='Path must be within an allowed limen.* namespace',
            ))


def _check_rule_based_manifest(manifest: dict[str, Any],
                                errors: list[YAMLError],
                                warnings: list[YAMLError]) -> None:

    for key in RULE_BASED_MANIFEST_REQUIRED:
        if key not in manifest:
            errors.append(YAMLError(
                message=f"Missing required field '{key}' for type 'rule_based'",
                path=f'sfd.manifest.{key}',
            ))

    strategy = manifest.get('strategy')
    if strategy is not None:
        _check_strategy(strategy, errors)

    ml_only = ML_MANIFEST_REQUIRED | ML_MANIFEST_OPTIONAL
    for key in ml_only:
        if key in manifest:
            warnings.append(YAMLError(
                message=f"Field '{key}' is not used in rule_based manifests",
                path=f'sfd.manifest.{key}',
            ))


def _check_strategy(strategy: Any, errors: list[YAMLError]) -> None:

    if not isinstance(strategy, dict):
        errors.append(YAMLError(
            message="'strategy' must be a mapping",
            path='sfd.manifest.strategy',
        ))
        return

    for key in STRATEGY_REQUIRED:
        if key not in strategy:
            errors.append(YAMLError(
                message=f"Missing required field '{key}'",
                path=f'sfd.manifest.strategy.{key}',
            ))

    conditions = strategy.get('conditions')
    if isinstance(conditions, list):
        for i, cond in enumerate(conditions):
            if not isinstance(cond, dict):
                errors.append(YAMLError(
                    message='Each condition must be a mapping',
                    path=f'sfd.manifest.strategy.conditions[{i}]',
                ))
                continue
            for key in ('name', 'type', 'expression'):
                if key not in cond:
                    errors.append(YAMLError(
                        message=f"Missing required field '{key}'",
                        path=f'sfd.manifest.strategy.conditions[{i}].{key}',
                    ))

    entry = strategy.get('entry')
    if entry is not None and not isinstance(entry, str):
        errors.append(YAMLError(
            message="'entry' must be a string",
            path='sfd.manifest.strategy.entry',
        ))


def _check_params(params: Any, errors: list[YAMLError]) -> None:

    if not isinstance(params, dict):
        errors.append(YAMLError(
            message="'sfd.params' must be a mapping",
            path='sfd.params',
            suggestion='Each key maps to a list of values, e.g. lookback: [12, 24, 48]',
        ))
        return

    for key, values in params.items():
        if not isinstance(values, list):
            errors.append(YAMLError(
                message=f"Parameter '{key}' must be a list of values",
                path=f'sfd.params.{key}',
                suggestion=f'Change to {key}: [{values}]',
            ))


def _check_uel(d: dict[str, Any],
               errors: list[YAMLError],
               warnings: list[YAMLError]) -> None:

    uel = d.get('uel')
    if not isinstance(uel, dict):
        errors.append(YAMLError(message="'uel' must be a mapping", path='uel'))
        return

    for key in UEL_REQUIRED:
        if key not in uel:
            errors.append(YAMLError(
                message=f"Missing required field '{key}'",
                path=f'uel.{key}',
            ))

    n = uel.get('n_permutations')
    if n is not None and not isinstance(n, int):
        errors.append(YAMLError(
            message="'n_permutations' must be an integer",
            path='uel.n_permutations',
        ))

    search = uel.get('search_strategy')
    if search is not None:
        if not isinstance(search, dict) or 'type' not in search:
            errors.append(YAMLError(
                message="'search_strategy' must be a mapping with a 'type' field",
                path='uel.search_strategy',
                suggestion='Use type: random or type: grid',
            ))
        elif search['type'] not in VALID_SEARCH_STRATEGY_TYPES:
            errors.append(YAMLError(
                message=f"Invalid search_strategy type '{search['type']}'",
                path='uel.search_strategy.type',
                suggestion=f"Use one of: {', '.join(sorted(VALID_SEARCH_STRATEGY_TYPES))}",
            ))

    output_format = uel.get('output_format')
    if output_format is not None and output_format not in VALID_OUTPUT_FORMATS:
        errors.append(YAMLError(
            message=f"Invalid output_format '{output_format}'",
            path='uel.output_format',
            suggestion=f"Use one of: {', '.join(sorted(VALID_OUTPUT_FORMATS))}",
        ))

    known = UEL_REQUIRED | UEL_OPTIONAL
    for key in uel:
        if key not in known:
            warnings.append(YAMLError(
                message=f"Unknown uel field '{key}'",
                path=f'uel.{key}',
            ))

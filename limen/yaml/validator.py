from dataclasses import dataclass
from dataclasses import field
from typing import Any

from limen.yaml.errors import YAMLError
from limen.yaml.rules import CalibrationCrossRef
from limen.yaml.rules import CalibrationPresence
from limen.yaml.rules import NameSlug
from limen.yaml.rules import ConditionsList
from limen.yaml.rules import DataSource
from limen.yaml.rules import FuncList
from limen.yaml.rules import NoUnknownKeys
from limen.yaml.rules import OneOf
from limen.yaml.rules import ParamCoverage
from limen.yaml.rules import Required
from limen.yaml.rules import RuleEngine
from limen.yaml.rules import SchemaVersion
from limen.yaml.rules import SfdParams
from limen.yaml.rules import SplitSpec
from limen.yaml.rules import WarnIfPresent
from limen.yaml.rules import When
from limen.yaml.rules import get_at
from limen.yaml.schema import CALIBRATION_FUNC_OPTIONAL
from limen.yaml.schema import CALIBRATION_FUNC_REQUIRED
from limen.yaml.schema import CALIBRATION_OPTIONAL
from limen.yaml.schema import DATA_SOURCE_OPTIONAL
from limen.yaml.schema import DATA_SOURCE_REQUIRED
from limen.yaml.schema import FEATURE_ABLATION_OPTIONAL
from limen.yaml.schema import MANIFEST_OPTIONAL_SHARED
from limen.yaml.schema import MANIFEST_REQUIRED
from limen.yaml.schema import METADATA_OPTIONAL
from limen.yaml.schema import METADATA_REQUIRED
from limen.yaml.schema import ML_MANIFEST_OPTIONAL
from limen.yaml.schema import ML_MANIFEST_REQUIRED
from limen.yaml.schema import PCA_COMPRESSION_OPTIONAL
from limen.yaml.schema import RULE_BASED_MANIFEST_OPTIONAL
from limen.yaml.schema import RULE_BASED_MANIFEST_REQUIRED
from limen.yaml.schema import SCALER_EXPLICIT_OPTIONAL
from limen.yaml.schema import SCALER_EXPLICIT_REQUIRED
from limen.yaml.schema import SCALER_FROM_PARAMS_REQUIRED
from limen.yaml.schema import SEARCH_STRATEGY_REQUIRED
from limen.yaml.schema import SPLIT_CONFIG_REQUIRED
from limen.yaml.schema import SPLIT_DATES_REQUIRED
from limen.yaml.schema import STRATEGY_REQUIRED
from limen.yaml.schema import TARGET_OPTIONAL
from limen.yaml.schema import TARGET_REQUIRED
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


_TOP_LEVEL_ENGINE = RuleEngine([
    Required('schema_version', str, suggestion=f'Set schema_version: "{VERSION}"'),
    Required('metadata', dict),
    Required('sfd', dict),
    Required('uel', dict),
])

_MAIN_ENGINE = RuleEngine([

    SchemaVersion(VERSION),

    Required('metadata.name', str),
    NameSlug(),
    Required('metadata.mode', str),
    OneOf('metadata.mode', VALID_MODES),
    NoUnknownKeys('metadata', METADATA_REQUIRED | METADATA_OPTIONAL),

    Required('sfd.manifest', dict),
    Required('sfd.params', dict),
    Required('sfd.manifest.type', str),
    OneOf('sfd.manifest.type', VALID_MANIFEST_TYPES),

    DataSource(),
    NoUnknownKeys('sfd.manifest.data_source', DATA_SOURCE_REQUIRED | DATA_SOURCE_OPTIONAL),
    NoUnknownKeys('sfd.manifest.test_data_source', DATA_SOURCE_REQUIRED | DATA_SOURCE_OPTIONAL),

    SplitSpec(),
    NoUnknownKeys('sfd.manifest.split_config', SPLIT_CONFIG_REQUIRED),
    NoUnknownKeys('sfd.manifest.split_dates', SPLIT_DATES_REQUIRED),

    Required('sfd.manifest.reference_architecture', str),

    When('sfd.manifest.type', 'ml', [
        Required('sfd.manifest.target', dict),
        Required('sfd.manifest.target.name', str),
        Required('sfd.manifest.target.class', str),
        NoUnknownKeys('sfd.manifest.target', TARGET_REQUIRED | TARGET_OPTIONAL, severity='error'),
        FuncList('sfd.manifest.indicators'),
        FuncList('sfd.manifest.features'),
        NoUnknownKeys('sfd.manifest.scaler',
                      SCALER_EXPLICIT_REQUIRED | SCALER_EXPLICIT_OPTIONAL | SCALER_FROM_PARAMS_REQUIRED),
        NoUnknownKeys('sfd.manifest.feature_ablation', FEATURE_ABLATION_OPTIONAL),
        NoUnknownKeys('sfd.manifest.pca_compression', PCA_COMPRESSION_OPTIONAL),
        NoUnknownKeys('sfd.manifest.calibration', CALIBRATION_OPTIONAL),
        NoUnknownKeys('sfd.manifest.calibration.probability_calibration',
                      CALIBRATION_FUNC_REQUIRED | CALIBRATION_FUNC_OPTIONAL),
        NoUnknownKeys('sfd.manifest.calibration.threshold_function',
                      CALIBRATION_FUNC_REQUIRED | CALIBRATION_FUNC_OPTIONAL),
        NoUnknownKeys(
            'sfd.manifest',
            MANIFEST_REQUIRED | MANIFEST_OPTIONAL_SHARED | ML_MANIFEST_REQUIRED | ML_MANIFEST_OPTIONAL,
        ),
    ]),

    When('sfd.manifest.type', 'rule_based', [
        Required('sfd.manifest.strategy', dict),
        Required('sfd.manifest.strategy.conditions', list),
        Required('sfd.manifest.strategy.entry', str),
        ConditionsList(),
        NoUnknownKeys('sfd.manifest.strategy', STRATEGY_REQUIRED),
        WarnIfPresent(
            'sfd.manifest',
            ML_MANIFEST_REQUIRED | ML_MANIFEST_OPTIONAL,
            message_template="Field '{key}' is not used in rule_based manifests",
        ),
        NoUnknownKeys(
            'sfd.manifest',
            MANIFEST_REQUIRED | MANIFEST_OPTIONAL_SHARED | RULE_BASED_MANIFEST_REQUIRED | RULE_BASED_MANIFEST_OPTIONAL,
        ),
    ]),

    SfdParams(),
    CalibrationPresence(),
    CalibrationCrossRef(),
    ParamCoverage(),

    Required('uel.n_permutations', int),
    OneOf('uel.search_strategy.type', VALID_SEARCH_STRATEGY_TYPES),
    NoUnknownKeys('uel.search_strategy', SEARCH_STRATEGY_REQUIRED),
    OneOf('uel.output_format', VALID_OUTPUT_FORMATS),
    NoUnknownKeys('uel', UEL_REQUIRED | UEL_OPTIONAL),
])


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

    _TOP_LEVEL_ENGINE.run(yaml_dict, errors, warnings)
    if errors:
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    _MAIN_ENGINE.run(yaml_dict, errors, warnings)

    _, mode = get_at(yaml_dict, 'metadata.mode')
    resolved_mode = str(mode) if mode in VALID_MODES else 'development'

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        mode=resolved_mode,
    )

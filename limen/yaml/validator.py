from limen.yaml._assessment_spec import check_assessment
from limen.yaml._fitted_binding import fitted_errors
from limen.yaml._metadata_spec import check_metadata
from limen.yaml._schema_version import schema_errors
from limen.yaml._boolean_spec import BooleanControlsSpec
from dataclasses import dataclass, field
from typing import Any, cast

from limen.yaml.errors import YAMLError
from limen.yaml._objective_spec import check_objective_spec
from limen.yaml._feature_shim_spec import FeatureShimSpec
from limen.yaml._recording_spec import RecordingSpec
from limen.yaml._temporal_spec import TemporalSpec
from limen.yaml.rules import BacktestCostSpec
from limen.yaml.rules import AcceptanceSpec
from limen.yaml.rules import BlockSpec
from limen.yaml.rules import CalibrationCrossRef, CalibrationPresence
from limen.yaml.rules import ConditionsList
from limen.yaml.rules import DecoderLookbackSpec
from limen.yaml.rules import DataSource
from limen.yaml.rules import FuncList
from limen.yaml.rules import NameSlug
from limen.yaml.rules import NoUnknownKeys
from limen.yaml.rules import OneOf
from limen.yaml.rules import ParamCoverage, ParamKeyFields
from limen.yaml.rules import PruningStrategiesSpec
from limen.yaml.rules import Required
from limen.yaml.rules import RequiredColumnsSpec
from limen.yaml.rules import Resolvable
from limen.yaml.rules import Rule, RuleEngine
from limen.yaml.rules import ScalerSpec
from limen.yaml.rules import SingleFuncBlock
from limen.yaml.rules import UelSpec
from limen.yaml.rules import SfdParams
from limen.yaml.rules import SplitSpec
from limen.yaml.rules import When
from limen.yaml.rules import get_at
from limen.yaml.schema import BACKTEST_OPTIONAL
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
from limen.yaml.schema import RULE_YAML_REJECTIONS
from limen.yaml.schema import SCALER_EXPLICIT_OPTIONAL
from limen.yaml.schema import SCALER_EXPLICIT_REQUIRED
from limen.yaml.schema import SCALER_FROM_PARAMS_REQUIRED
from limen.yaml.schema import SEARCH_STRATEGY_OPTIONAL
from limen.yaml.schema import SEARCH_STRATEGY_REQUIRED
from limen.yaml.schema import SFD_REQUIRED, TOP_LEVEL_REQUIRED
from limen.yaml._structural_keys import PLACEMENT_REJECTIONS
from limen.yaml.schema import SPLIT_DATES_OPTIONAL
from limen.yaml.schema import SPLIT_DATES_REQUIRED
from limen.yaml.schema import STRATEGY_REQUIRED
from limen.yaml.schema import TARGET_OPTIONAL
from limen.yaml.schema import TARGET_REQUIRED
from limen.yaml.schema import UEL_OPTIONAL, UEL_YAML_REJECTIONS
from limen.yaml.schema import UEL_REQUIRED
from limen.yaml.schema import VALID_MANIFEST_TYPES
from limen.yaml.schema import VALID_MODES
from limen.yaml.schema import VALID_OUTPUT_FORMATS
from limen.yaml.schema import VALID_SEARCH_STRATEGY_TYPES


@dataclass
class ValidationResult:

    '''Result of a YAML validation pass.'''

    valid: bool
    errors: list[YAMLError] = field(default_factory=list[YAMLError])
    warnings: list[YAMLError] = field(default_factory=list[YAMLError])
    mode: str = 'development'


_TOP_LEVEL_ENGINE = RuleEngine(cast(list[Rule], [
    Required('metadata', dict),
    Required('sfd', dict),
    Required('uel', dict),
]))

_MAIN_ENGINE = RuleEngine(cast(list[Rule], [
    BooleanControlsSpec(),
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

    SplitSpec(),
    AcceptanceSpec(),
    FeatureShimSpec(),
    RequiredColumnsSpec(),
    NoUnknownKeys('sfd.manifest.split_dates', SPLIT_DATES_REQUIRED | SPLIT_DATES_OPTIONAL, severity='error'),

    Required('sfd.manifest.reference_architecture', str),
    Resolvable('sfd.manifest.reference_architecture'),

    BlockSpec('sfd.manifest.backtest'),
    NoUnknownKeys('sfd.manifest.backtest', BACKTEST_OPTIONAL, severity='error'),
    BacktestCostSpec(),

    When('sfd.manifest.type', 'ml', cast(list[Rule], [
        Required('sfd.manifest.target', dict),
        Required('sfd.manifest.target.name', str),
        Required('sfd.manifest.target.class', str),
        Resolvable('sfd.manifest.target.class'),
        NoUnknownKeys('sfd.manifest.target', TARGET_REQUIRED | TARGET_OPTIONAL, severity='error'),
        FuncList('sfd.manifest.indicators'),
        FuncList('sfd.manifest.features'),
        SingleFuncBlock('sfd.manifest.pre_split_data_selector'),
        SingleFuncBlock('sfd.manifest.bar_formation'),
        SingleFuncBlock('sfd.manifest.data_dict_extension'),
        NoUnknownKeys('sfd.manifest.data_dict_extension', {'func'}, severity='error'),
        NoUnknownKeys('sfd.manifest.scaler',
                      SCALER_EXPLICIT_REQUIRED | SCALER_EXPLICIT_OPTIONAL | SCALER_FROM_PARAMS_REQUIRED),
        ScalerSpec(),
        DecoderLookbackSpec(),
        BlockSpec(
            'sfd.manifest.feature_ablation',
            'sfd.manifest.pca_compression',
            'sfd.manifest.params_override',
            'sfd.manifest.metrics_params',
        ),
        NoUnknownKeys('sfd.manifest.feature_ablation', FEATURE_ABLATION_OPTIONAL),
        ParamKeyFields('sfd.manifest.feature_ablation', 'drop_count_key', 'seed_key'),
        NoUnknownKeys('sfd.manifest.pca_compression', PCA_COMPRESSION_OPTIONAL, severity='error'),
        ParamKeyFields('sfd.manifest.pca_compression', 'enabled_param', 'n_components_param', 'scaler_param_name', 'component_prefix'),
        NoUnknownKeys('sfd.manifest.calibration', CALIBRATION_OPTIONAL),
        NoUnknownKeys('sfd.manifest.calibration.probability_calibration',
                      CALIBRATION_FUNC_REQUIRED | CALIBRATION_FUNC_OPTIONAL),
        NoUnknownKeys('sfd.manifest.calibration.threshold_function',
                      CALIBRATION_FUNC_REQUIRED | CALIBRATION_FUNC_OPTIONAL),
        NoUnknownKeys(
            'sfd.manifest',
            MANIFEST_REQUIRED | MANIFEST_OPTIONAL_SHARED | ML_MANIFEST_REQUIRED | ML_MANIFEST_OPTIONAL | {'acceptance'},
        ),
    ])),

    When('sfd.manifest.type', 'rule_based', cast(list[Rule], [
        Required('sfd.manifest.strategy', dict),
        Required('sfd.manifest.strategy.conditions', list),
        Required('sfd.manifest.strategy.entry', str),
        ConditionsList(),
        TemporalSpec(),
        FuncList('sfd.manifest.indicators'),
        NoUnknownKeys('sfd.manifest.strategy', STRATEGY_REQUIRED),
        NoUnknownKeys(
            'sfd.manifest',
            MANIFEST_REQUIRED | MANIFEST_OPTIONAL_SHARED | RULE_BASED_MANIFEST_REQUIRED | RULE_BASED_MANIFEST_OPTIONAL | {'acceptance'}, rejected=RULE_YAML_REJECTIONS,
        ),
    ])),

    SfdParams(),
    CalibrationPresence(),
    CalibrationCrossRef(),
    ParamCoverage(),

    Required('uel.n_permutations', int),
    OneOf('uel.search_strategy.type', VALID_SEARCH_STRATEGY_TYPES),
    NoUnknownKeys('uel.search_strategy', SEARCH_STRATEGY_REQUIRED | SEARCH_STRATEGY_OPTIONAL),
    OneOf('uel.output_format', VALID_OUTPUT_FORMATS),
    UelSpec(),
    RecordingSpec(),
    PruningStrategiesSpec(),
    NoUnknownKeys('uel', UEL_REQUIRED | UEL_OPTIONAL, rejected=UEL_YAML_REJECTIONS | PLACEMENT_REJECTIONS['uel']),
    NoUnknownKeys('', TOP_LEVEL_REQUIRED | {'assessment', 'holdout', 'lineage'}, rejected=PLACEMENT_REJECTIONS['']),
    NoUnknownKeys('sfd', SFD_REQUIRED, rejected=PLACEMENT_REJECTIONS['sfd']),
    NoUnknownKeys('lineage', {'id', 'committed_at', 'parent_id'}),
]))


unknown_errors = _MAIN_ENGINE.unknown_errors
unused_errors = ParamCoverage().unused_errors

def validate(yaml_dict: dict[str, Any]) -> ValidationResult:

    '''Collect YAML errors and warnings; validity requires no errors.'''

    errors = schema_errors(yaml_dict)
    warnings: list[YAMLError] = []
    if errors:
        return ValidationResult(valid=False, errors=errors)

    _TOP_LEVEL_ENGINE.run(yaml_dict, errors, warnings)
    if errors:
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    check_metadata(yaml_dict, errors)
    check_assessment(yaml_dict, errors)
    check_objective_spec(yaml_dict, errors)
    _MAIN_ENGINE.run(yaml_dict, errors, warnings)
    errors.extend(fitted_errors(yaml_dict))

    _, mode = get_at(yaml_dict, 'metadata.mode')
    resolved_mode = str(mode) if mode in VALID_MODES else 'development'

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings, mode=resolved_mode)

VERSION = '1.0'

VALID_MODES = {'production', 'development'}
VALID_MANIFEST_TYPES = {'ml', 'rule_based'}
VALID_OUTPUT_FORMATS = {'csv', 'parquet'}
VALID_SEARCH_STRATEGY_TYPES = {'random', 'grid'}

TOP_LEVEL_REQUIRED = {'schema_version', 'metadata', 'sfd', 'uel'}

METADATA_REQUIRED = {'name', 'limen_version', 'mode'}
METADATA_OPTIONAL = {'author', 'created_at', 'description', 'tags'}

SFD_REQUIRED = {'manifest', 'params'}

MANIFEST_REQUIRED = {'type', 'data_source', 'split_config', 'reference_architecture'}
MANIFEST_OPTIONAL_SHARED = {'test_data_source', 'required_columns'}

ML_MANIFEST_REQUIRED = {'target'}
ML_MANIFEST_OPTIONAL = {
    'pre_split_data_selector',
    'bar_formation',
    'indicators',
    'features',
    'scaler',
    'feature_ablation',
    'data_dict_extension',
    'calibration',
    'params_override',
    'metrics_params',
}

RULE_BASED_MANIFEST_REQUIRED = {'strategy'}
RULE_BASED_MANIFEST_OPTIONAL: set[str] = set()

DATA_SOURCE_REQUIRED = {'method', 'params'}

SPLIT_CONFIG_REQUIRED = {'train', 'val', 'test'}

TARGET_REQUIRED = {'name', 'class'}
TARGET_OPTIONAL = {'fit_params', 'transform_params'}

STRATEGY_REQUIRED = {'conditions', 'entry'}
CONDITION_REQUIRED = {'name', 'type', 'expression'}
CONDITION_OPTIONAL = {'params'}
VALID_CONDITION_TYPES = {'polars_lambda'}

SCALER_EXPLICIT_REQUIRED = {'class'}
SCALER_EXPLICIT_OPTIONAL = {'params'}
SCALER_FROM_PARAMS_REQUIRED = {'from_params'}

FEATURE_ABLATION_OPTIONAL = {'drop_count_key', 'seed_key'}

CALIBRATION_OPTIONAL = {'probability_calibration', 'threshold_function'}
CALIBRATION_FUNC_REQUIRED = {'func'}
CALIBRATION_FUNC_OPTIONAL = {'params'}

UEL_REQUIRED = {'n_permutations'}
UEL_OPTIONAL = {
    'search_strategy',
    'pruning_strategies',
    'feedback_interval',
    'checkpoint_interval',
    'experiment_dir',
    'intra_callback',
    'prep_each_round',
    'output_format',
    'output_path',
}
SEARCH_STRATEGY_REQUIRED = {'type'}
PRUNING_STRATEGY_REQUIRED = {'type'}
PRUNING_STRATEGY_OPTIONAL = {'params'}

COMPLEXITY_LOW_MAX = 100
COMPLEXITY_MEDIUM_MAX = 1_000
COMPLEXITY_HIGH_MAX = 10_000

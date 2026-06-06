VERSION = '1.0'

VALID_MODES = {'production', 'development'}
VALID_MANIFEST_TYPES = {'ml', 'rule_based'}
VALID_OUTPUT_FORMATS = {'csv', 'parquet'}
VALID_SEARCH_STRATEGY_TYPES = {'random', 'grid'}

TOP_LEVEL_REQUIRED = {'schema_version', 'metadata', 'sfd', 'uel'}

METADATA_REQUIRED = {'name', 'mode'}
METADATA_OPTIONAL = {'author', 'created_at', 'description', 'tags', 'limen_version'}

SFD_REQUIRED = {'manifest', 'params'}

MANIFEST_REQUIRED = {'type', 'data_source', 'reference_architecture'}
MANIFEST_OPTIONAL_SHARED = {'required_columns', 'split_dates', 'indicators', 'backtest'}

ML_MANIFEST_REQUIRED = {'target'}
ML_MANIFEST_OPTIONAL = {
    'pre_split_data_selector',
    'bar_formation',
    'features',
    'scaler',
    'feature_ablation',
    'data_dict_extension',
    'calibration',
    'pca_compression',
    'params_override',
    'metrics_params',
    'decoder_lookback',
}

RULE_BASED_MANIFEST_REQUIRED = {'strategy'}
RULE_BASED_MANIFEST_OPTIONAL: set[str] = set()

DATA_SOURCE_REQUIRED = {'method'}
DATA_SOURCE_OPTIONAL = {'params'}

SPLIT_CONFIG_REQUIRED = {'train', 'val', 'test'}
SPLIT_DATES_REQUIRED = {'train_start', 'train_end', 'val_start', 'val_end', 'test_start', 'test_end'}

TARGET_REQUIRED = {'name', 'class'}
TARGET_OPTIONAL = {'fit_params', 'transform_params'}

STRATEGY_REQUIRED = {'conditions', 'entry'}
CONDITION_REQUIRED = {'id', 'name'}
CONDITION_OPTIONAL = {'params', 'persistence_n', 'recency_n'}

SCALER_EXPLICIT_REQUIRED = {'class'}
SCALER_EXPLICIT_OPTIONAL: set[str] = set()
SCALER_FROM_PARAMS_REQUIRED = {'from_params'}

FEATURE_ABLATION_OPTIONAL = {'drop_count_key', 'seed_key'}
PCA_COMPRESSION_OPTIONAL = {'enabled_param', 'n_components_param', 'scaler_param_name', 'component_prefix'}
BACKTEST_OPTIONAL = {'fee_bps', 'slip_bps'}

CALIBRATION_OPTIONAL = {'probability_calibration', 'threshold_function'}
CALIBRATION_FUNC_REQUIRED = {'func'}
CALIBRATION_FUNC_OPTIONAL = {'params'}

UEL_REQUIRED = {'n_permutations'}
UEL_OPTIONAL = {
    'search_strategy',
    'pruning_strategies',
    'feedback_interval',
    'checkpoint_interval',
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

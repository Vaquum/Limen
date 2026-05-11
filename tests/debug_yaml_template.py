import json
import tempfile
from pathlib import Path

import polars as pl
import pytest

from limen.calibration import grid_threshold_optimizer
from limen.calibration import sklearn_probability_calibrator
from limen.experiment import RandomStrategy
from limen.experiment.experiment_core import UniversalExperimentLoop
from limen.experiment.manifest_core import MLManifest
from limen.experiment.param_domain import ParamDomain
from limen.metrics.balanced_metric import balanced_metric
from limen.yaml.compiler import CompiledSFD
from limen.yaml.compiler import _resolve_func_params
from limen.yaml.compiler import build_manifest
from limen.yaml.parser import parse
from limen.yaml.resolver import is_resolvable
from limen.yaml.resolver import resolve
from limen.yaml.validator import validate

TEMPLATE = Path(__file__).parent.parent / 'limen/yaml/templates/logreg_binary.yaml'

N_PERMUTATIONS = 10

_RESULTS_DIR: Path | None = None


def _get_results_dir() -> Path:
    global _RESULTS_DIR
    if _RESULTS_DIR is None:
        tmp = Path(tempfile.mkdtemp())
        yaml_dict, _ = parse(TEMPLATE)
        sfd_cfg = yaml_dict.get('sfd', {})
        params = {k: list(v) for k, v in (sfd_cfg.get('params') or {}).items()}
        domain = ParamDomain(params)
        sfd = CompiledSFD(yaml_dict)
        uel = UniversalExperimentLoop(
            sfd=sfd,
            search_strategy=RandomStrategy(domain),
            experiment_dir=tmp,
            test_mode=True,
        )
        uel.run(
            experiment_name='logreg_binary_experiment',
            n_permutations=N_PERMUTATIONS,
            prep_each_round=True,
        )
        _RESULTS_DIR = tmp
    return _RESULTS_DIR


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def test_template_parses_without_errors() -> None:
    _, errors = parse(TEMPLATE)
    assert errors == []


def test_template_top_level_keys_present() -> None:
    yaml_dict, _ = parse(TEMPLATE)
    for key in ('schema_version', 'metadata', 'sfd', 'uel'):
        assert key in yaml_dict


def test_template_metadata_fields() -> None:
    yaml_dict, _ = parse(TEMPLATE)
    meta = yaml_dict['metadata']
    assert meta['name'] == 'logreg_binary_experiment'
    assert meta['mode'] == 'development'


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def test_template_validates_with_no_errors() -> None:
    yaml_dict, _ = parse(TEMPLATE)
    result = validate(yaml_dict)
    assert result.valid
    assert result.errors == []


def test_template_validation_reports_development_mode() -> None:
    yaml_dict, _ = parse(TEMPLATE)
    result = validate(yaml_dict)
    assert result.mode == 'development'


def test_template_manifest_type_is_ml() -> None:
    yaml_dict, _ = parse(TEMPLATE)
    assert yaml_dict['sfd']['manifest']['type'] == 'ml'


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

def test_all_limen_paths_in_template_are_resolvable() -> None:
    limen_paths = [
        'limen.data.HistoricalData.get_spot_klines',
        'limen.indicators.roc',
        'limen.indicators.atr',
        'limen.indicators.ppo',
        'limen.indicators.wilder_rsi',
        'limen.features.fractional_diff',
        'limen.features.vwap',
        'limen.features.kline_imbalance',
        'limen.targets.QuantileBinaryTarget',
        'limen.sfd.reference_architecture.logreg_binary',
        'limen.calibration.sklearn_probability_calibrator',
        'limen.calibration.grid_threshold_optimizer',
        'limen.metrics.balanced_metric.balanced_metric',
    ]
    for path in limen_paths:
        assert is_resolvable(path), f"Not resolvable: {path}"
        assert callable(resolve(path)), f"Resolved but not callable: {path}"


# ---------------------------------------------------------------------------
# _resolve_func_params
# ---------------------------------------------------------------------------

def test_resolve_func_params_resolves_callable_path() -> None:
    params = {'metric': 'limen.metrics.balanced_metric.balanced_metric'}
    resolved = _resolve_func_params(params)
    assert resolved['metric'] is balanced_metric


def test_resolve_func_params_leaves_round_param_key_as_string() -> None:
    params = {'threshold_min': 'threshold_min', 'threshold_max': 'threshold_max'}
    resolved = _resolve_func_params(params)
    assert resolved['threshold_min'] == 'threshold_min'
    assert resolved['threshold_max'] == 'threshold_max'


def test_resolve_func_params_passes_through_non_string_values() -> None:
    params = {'threshold_max': 0.70, 'n_steps': 5}
    resolved = _resolve_func_params(params)
    assert resolved['threshold_max'] == pytest.approx(0.70)
    assert resolved['n_steps'] == 5


# ---------------------------------------------------------------------------
# Compiler — CompiledSFD
# ---------------------------------------------------------------------------

def test_compiled_sfd_name_reflects_experiment() -> None:
    yaml_dict, _ = parse(TEMPLATE)
    sfd = CompiledSFD(yaml_dict)
    assert sfd.__name__ == 'yaml:logreg_binary_experiment'


def test_compiled_sfd_params_contains_all_yaml_keys() -> None:
    yaml_dict, _ = parse(TEMPLATE)
    sfd = CompiledSFD(yaml_dict)
    params = sfd.params()
    expected = {
        'frac_diff_d', 'shift', 'q', 'roc_period', 'penalty',
        'scaler_type', 'feature_groups',
        'use_calibration', 'use_threshold',
        'cal_method', 'threshold_min', 'threshold_max', 'threshold_step',
        'class_weight', 'C', 'max_iter', 'solver', 'tol',
    }
    assert expected.issubset(set(params.keys()))


def test_compiled_sfd_params_are_all_lists() -> None:
    yaml_dict, _ = parse(TEMPLATE)
    sfd = CompiledSFD(yaml_dict)
    for key, values in sfd.params().items():
        assert isinstance(values, list) and len(values) > 0, f"Param '{key}' is not a non-empty list"


def test_compiled_sfd_manifest_is_cached() -> None:
    yaml_dict, _ = parse(TEMPLATE)
    sfd = CompiledSFD(yaml_dict)
    assert sfd.manifest() is sfd.manifest()


# ---------------------------------------------------------------------------
# Compiler — build_manifest
# ---------------------------------------------------------------------------

def test_build_manifest_returns_ml_manifest() -> None:
    yaml_dict, _ = parse(TEMPLATE)
    assert isinstance(build_manifest(yaml_dict), MLManifest)


def test_manifest_calibration_funcs_are_resolved() -> None:
    yaml_dict, _ = parse(TEMPLATE)
    config = build_manifest(yaml_dict).prediction_calibration_config
    assert config is not None
    assert config.calibration_func is sklearn_probability_calibrator
    assert config.threshold_func is grid_threshold_optimizer


def test_manifest_calibration_metric_is_resolved_callable() -> None:
    yaml_dict, _ = parse(TEMPLATE)
    config = build_manifest(yaml_dict).prediction_calibration_config
    # metric is a limen.* path in the YAML — must be resolved to the callable
    assert config.threshold_params['metric'] is balanced_metric


def test_manifest_calibration_threshold_bounds_stay_as_round_param_refs() -> None:
    yaml_dict, _ = parse(TEMPLATE)
    config = build_manifest(yaml_dict).prediction_calibration_config
    # {threshold_min} etc. are {param_name} references — stored as-is for runtime resolution
    assert config.threshold_params['threshold_min'] == '{threshold_min}'
    assert config.threshold_params['threshold_max'] == '{threshold_max}'
    assert config.threshold_params['threshold_step'] == '{threshold_step}'


def test_manifest_calibration_method_stays_as_round_param_ref() -> None:
    yaml_dict, _ = parse(TEMPLATE)
    config = build_manifest(yaml_dict).prediction_calibration_config
    assert config.calibration_params['method'] == '{cal_method}'


# ---------------------------------------------------------------------------
# Integration — results files
# ---------------------------------------------------------------------------

def test_results_dir_contains_expected_files() -> None:
    d = _get_results_dir()
    assert (d / 'results.csv').exists()
    assert (d / 'round_data.jsonl').exists()
    assert (d / 'metadata.json').exists()


def test_results_csv_has_n_permutations_rows() -> None:
    df = pl.read_csv(_get_results_dir() / 'results.csv')
    assert len(df) == N_PERMUTATIONS


def test_results_csv_has_metric_columns() -> None:
    df = pl.read_csv(_get_results_dir() / 'results.csv')
    for col in ('recall', 'precision', 'auc', 'accuracy', 'optimal_threshold', 'val_score'):
        assert col in df.columns, f"Missing metric column: {col}"


def test_results_csv_has_param_columns() -> None:
    df = pl.read_csv(_get_results_dir() / 'results.csv')
    yaml_dict, _ = parse(TEMPLATE)
    for key in yaml_dict['sfd']['params']:
        assert key in df.columns, f"Missing param column: {key}"


def test_results_csv_has_no_duplicate_param_hashes() -> None:
    df = pl.read_csv(_get_results_dir() / 'results.csv')
    hashes = df['_param_hash'].to_list()
    assert len(hashes) == len(set(hashes)), 'Duplicate param hashes — same permutation ran twice'


def test_results_csv_auc_values_are_valid() -> None:
    df = pl.read_csv(_get_results_dir() / 'results.csv')
    for v in df['auc'].to_list():
        assert v is not None and 0.0 <= float(v) <= 1.0, f"Invalid AUC: {v}"


# ---------------------------------------------------------------------------
# Integration — round_params correctness
# ---------------------------------------------------------------------------

def test_round_params_solver_values_are_from_yaml_space() -> None:
    df = pl.read_csv(_get_results_dir() / 'results.csv')
    yaml_dict, _ = parse(TEMPLATE)
    allowed = set(str(v) for v in yaml_dict['sfd']['params']['solver'])
    actual = set(df['solver'].to_list())
    assert actual.issubset(allowed), f"Out-of-space solver values: {actual - allowed}"


def test_round_params_cal_method_values_are_from_yaml_space() -> None:
    df = pl.read_csv(_get_results_dir() / 'results.csv')
    yaml_dict, _ = parse(TEMPLATE)
    allowed = set(str(v) for v in yaml_dict['sfd']['params']['cal_method'])
    actual = set(df['cal_method'].to_list())
    assert actual.issubset(allowed), f"Out-of-space cal_method values: {actual - allowed}"


def test_round_params_scaler_type_values_are_from_yaml_space() -> None:
    df = pl.read_csv(_get_results_dir() / 'results.csv')
    yaml_dict, _ = parse(TEMPLATE)
    allowed = set(str(v) for v in yaml_dict['sfd']['params']['scaler_type'])
    actual = set(df['scaler_type'].to_list())
    assert actual.issubset(allowed), f"Out-of-space scaler_type values: {actual - allowed}"


def test_optimal_threshold_is_within_searched_bounds() -> None:
    df = pl.read_csv(_get_results_dir() / 'results.csv')
    yaml_dict, _ = parse(TEMPLATE)
    # threshold_min ranges from 0.20 to 0.30, threshold_max from 0.60 to 0.70
    min_lower = min(float(v) for v in yaml_dict['sfd']['params']['threshold_min'])
    max_upper = max(float(v) for v in yaml_dict['sfd']['params']['threshold_max'])
    # optimal_threshold is null when use_threshold=False — only check calibrated rounds
    thresholds = [float(t) for t in df['optimal_threshold'].to_list() if t is not None]
    assert len(thresholds) > 0, 'No rounds with threshold optimization found'
    for t in thresholds:
        assert min_lower <= t <= max_upper, f"Threshold {t} outside [{min_lower}, {max_upper}]"


# ---------------------------------------------------------------------------
# Integration — round_data.jsonl
# ---------------------------------------------------------------------------

def test_round_data_has_n_permutations_entries() -> None:
    lines = (_get_results_dir() / 'round_data.jsonl').read_text().strip().splitlines()
    assert len(lines) == N_PERMUTATIONS


def test_round_data_entries_have_required_keys() -> None:
    lines = (_get_results_dir() / 'round_data.jsonl').read_text().strip().splitlines()
    for i, line in enumerate(lines):
        entry = json.loads(line)
        for key in ('round_id', 'round_params', 'preds', 'alignment'):
            assert key in entry, f"Entry {i} missing key '{key}'"


def test_round_data_round_ids_are_sequential() -> None:
    lines = (_get_results_dir() / 'round_data.jsonl').read_text().strip().splitlines()
    ids = [json.loads(line)['round_id'] for line in lines]
    assert ids == list(range(N_PERMUTATIONS))


def test_round_data_round_params_contain_all_yaml_param_keys() -> None:
    yaml_dict, _ = parse(TEMPLATE)
    expected_keys = set(yaml_dict['sfd']['params'].keys())
    lines = (_get_results_dir() / 'round_data.jsonl').read_text().strip().splitlines()
    for i, line in enumerate(lines):
        rp = json.loads(line)['round_params']
        missing = expected_keys - set(rp.keys())
        assert not missing, f"Round {i} round_params missing keys: {missing}"


def test_round_data_preds_are_binary() -> None:
    lines = (_get_results_dir() / 'round_data.jsonl').read_text().strip().splitlines()
    for i, line in enumerate(lines):
        preds = json.loads(line)['preds']
        assert len(preds) > 0, f"Round {i} has empty preds"
        assert set(preds).issubset({0, 1}), f"Round {i} has non-binary preds: {set(preds)}"


def test_round_data_alignment_has_datetime_keys() -> None:
    lines = (_get_results_dir() / 'round_data.jsonl').read_text().strip().splitlines()
    for i, line in enumerate(lines):
        alignment = json.loads(line)['alignment']
        assert 'first_test_datetime' in alignment, f"Round {i} missing first_test_datetime"
        assert 'last_test_datetime' in alignment, f"Round {i} missing last_test_datetime"


# ---------------------------------------------------------------------------
# Integration — metadata.json
# ---------------------------------------------------------------------------

def test_metadata_json_sfd_module_is_yaml_prefixed() -> None:
    meta = json.loads((_get_results_dir() / 'metadata.json').read_text())
    assert meta['sfd_module'] == 'yaml:logreg_binary_experiment'


def test_metadata_json_has_limen_version_and_created_at() -> None:
    meta = json.loads((_get_results_dir() / 'metadata.json').read_text())
    assert 'limen_version' in meta
    assert 'created_at' in meta

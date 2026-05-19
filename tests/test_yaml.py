from datetime import date
from textwrap import dedent

from limen.calibration import grid_threshold_optimizer
from limen.calibration import sklearn_probability_calibrator
from limen.experiment.manifest_core import AblationConfig
from limen.experiment.manifest_core import MLManifest
from limen.experiment.manifest_core import RuleBasedManifest
from limen.metrics.balanced_metric import balanced_metric
from limen.yaml.compiler import CompiledSFD
from limen.yaml.compiler import _resolve_func_params
from limen.yaml.compiler import build_manifest
from limen.yaml.errors import GitError
from limen.yaml.errors import ValidationError
from limen.yaml.errors import YAMLError
from limen.yaml.parser import parse
from limen.yaml.resolver import is_resolvable
from limen.yaml.resolver import resolve
from limen.yaml.validator import validate

_MINIMAL_ML_YAML = dedent('''\
    schema_version: "1.0"
    metadata:
      name: test_exp
      mode: development
    sfd:
      manifest:
        type: ml
        data_source:
          method: limen.data.HistoricalData.get_spot_klines
          params:
            kline_size: 3600
        test_data_source:
          method: limen.data.HistoricalData.get_spot_klines
          params:
            kline_size: 7200
            n_rows: 500
        split_config:
          train: 8
          val: 1
          test: 2
        indicators:
          - func: limen.indicators.roc
            params:
              period: "{roc_period}"
              group: momentum
        features:
          - func: limen.features.fractional_diff
            params:
              d: "{frac_diff_d}"
              cols:
                - close
        target:
          name: quantile_flag
          class: limen.targets.QuantileBinaryTarget
          fit_params:
            source_column: "roc_{roc_period}"
            quantile: "{q}"
          transform_params:
            shift: "{shift}"
        scaler:
          from_params: scaler_type
        feature_ablation:
          drop_count_key: feature_drop_count
          seed_key: feature_drop_seed
        reference_architecture: limen.sfd.reference_architecture.logreg_binary
        calibration:
          probability_calibration:
            func: limen.calibration.sklearn_probability_calibrator
            params:
              method: "{cal_method}"
          threshold_function:
            func: limen.calibration.grid_threshold_optimizer
            params:
              metric: limen.metrics.balanced_metric.balanced_metric
              threshold_min: "{threshold_min}"
              threshold_max: "{threshold_max}"
      params:
        roc_period: [1, 4]
        frac_diff_d: [0.0]
        q: [0.35]
        shift: [-1]
        scaler_type: [logreg]
        feature_drop_count: [0]
        feature_drop_seed: [42]
        use_calibration: [true, false]
        use_threshold: [true, false]
        cal_method: [isotonic]
        threshold_min: [0.20]
        threshold_max: [0.70]
        threshold_step: [0.05]
        penalty: [l2]
        class_weight: [0.5]
        C: [1.0]
        max_iter: [100]
        solver: [lbfgs]
        tol: [0.01]
    uel:
      n_permutations: 5
      search_strategy:
        type: random
      output_format: csv
''')

_MINIMAL_RULE_BASED_YAML = dedent('''\
    schema_version: "1.0"
    metadata:
      name: rule_exp
      mode: development
    sfd:
      manifest:
        type: rule_based
        data_source:
          method: limen.data.HistoricalData.get_spot_klines
          params:
            kline_size: 3600
        split_config:
          train: 8
          val: 1
          test: 2
        strategy:
          conditions:
            - id: rsi_low
              name: rsi_low
              type: threshold
              column: rsi
              operator: lt
              value: 30
          entry: rsi_low
        reference_architecture: limen.sfd.reference_architecture.rule_based
      params:
        dummy_param: [1, 2]
    uel:
      n_permutations: 2
      search_strategy:
        type: random
      output_format: csv
''')


def test_parse_valid_yaml_returns_dict_and_no_errors() -> None:
    yaml_dict, errors = parse(_MINIMAL_ML_YAML)
    assert isinstance(yaml_dict, dict)
    assert errors == []


def test_parse_malformed_yaml_returns_errors() -> None:
    _, errors = parse('sfd: [unclosed')
    assert len(errors) > 0


def test_parse_single_line_string_parses_as_yaml_not_path() -> None:
    yaml_dict, errors = parse('schema_version: "1.0"')
    assert errors == []
    assert yaml_dict.get('schema_version') == '1.0'


def test_is_resolvable_returns_true_for_limen_path() -> None:
    assert is_resolvable('limen.indicators.roc')
    assert is_resolvable('limen.data.HistoricalData.get_spot_klines')
    assert is_resolvable('limen.calibration.grid_threshold_optimizer')


def test_is_resolvable_returns_false_for_non_limen_path() -> None:
    assert not is_resolvable('os.path.join')
    assert not is_resolvable('not_a_module.foo')
    assert not is_resolvable('{threshold_min}')


def test_is_resolvable_returns_false_for_namespace_prefix_collision() -> None:
    assert not is_resolvable('limen.database.something')
    assert not is_resolvable('limen.datax.something')


def test_resolve_returns_callable_for_limen_path() -> None:
    assert callable(resolve('limen.indicators.roc'))


def test_validate_passes_valid_ml_yaml() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    result = validate(yaml_dict)
    assert result.valid
    assert result.errors == []


def test_validate_error_for_missing_required_field() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    del yaml_dict['sfd']['manifest']['split_config']
    result = validate(yaml_dict)
    assert not result.valid
    assert any('split_config' in e.message for e in result.errors)


def test_validate_error_for_calibration_param_ref_not_in_sfd_params() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['calibration']['threshold_function']['params']['bogus'] = '{nonexistent_param}'
    result = validate(yaml_dict)
    assert not result.valid
    assert any('nonexistent_param' in e.message for e in result.errors)


def test_validate_no_error_when_calibration_param_is_limen_path() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    result = validate(yaml_dict)
    assert result.valid


def test_validate_warning_for_unused_sfd_param() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['params']['orphan_param'] = [1, 2, 3]
    result = validate(yaml_dict)
    assert any('orphan_param' in w.message for w in result.warnings)


def test_resolve_func_params_resolves_limen_path_to_callable() -> None:
    resolved = _resolve_func_params({'metric': 'limen.metrics.balanced_metric.balanced_metric'})
    assert resolved['metric'] is balanced_metric


def test_resolve_func_params_preserves_round_param_ref_as_string() -> None:
    params = {'threshold_min': '{threshold_min}', 'threshold_max': '{threshold_max}'}
    resolved = _resolve_func_params(params)
    assert resolved['threshold_min'] == '{threshold_min}'
    assert resolved['threshold_max'] == '{threshold_max}'


def test_build_manifest_data_source_method_is_callable_with_correct_params() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    manifest = build_manifest(yaml_dict)
    cfg = manifest.data_source_config
    assert callable(cfg.method)
    assert cfg.params['kline_size'] == 3600


def test_build_manifest_test_data_source_method_is_callable() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    manifest = build_manifest(yaml_dict)
    cfg = manifest.test_data_source_config
    assert callable(cfg.method)
    assert cfg.params['n_rows'] == 500


def test_build_manifest_split_config_tuple_is_correct() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    assert build_manifest(yaml_dict).split_config == (8, 1, 2)


def test_build_manifest_indicator_round_param_ref_passes_through_as_string() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    manifest = build_manifest(yaml_dict)
    roc_entry = next(e for e in manifest.feature_transforms if 'roc' in e.func.__name__)
    assert roc_entry.params['period'] == '{roc_period}'


def test_build_manifest_feature_round_param_ref_passes_through_as_string() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    manifest = build_manifest(yaml_dict)
    fd_entry = next(e for e in manifest.feature_transforms if 'fractional' in e.func.__name__)
    assert fd_entry.params['d'] == '{frac_diff_d}'


def test_build_manifest_target_fit_params_preserve_refs_and_template_strings() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    fit_params = build_manifest(yaml_dict).target_class_config.fit_params
    assert fit_params['quantile'] == '{q}'
    assert fit_params['source_column'] == 'roc_{roc_period}'


def test_build_manifest_target_transform_params_preserve_round_param_refs() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    assert build_manifest(yaml_dict).target_class_config.transform_params['shift'] == '{shift}'


def test_build_manifest_scaler_from_params_is_configured() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    manifest = build_manifest(yaml_dict)
    assert manifest.scaler is not None
    fitted_params_list, _, _ = manifest.scaler
    assert any('scaler_type' in str(params) for _, _, params in fitted_params_list)


def test_build_manifest_feature_ablation_keys_match_yaml() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    ablation = build_manifest(yaml_dict).ablation_config
    assert isinstance(ablation, AblationConfig)
    assert ablation.drop_count_key == 'feature_drop_count'
    assert ablation.seed_key == 'feature_drop_seed'


def test_build_manifest_reference_architecture_is_callable() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    assert callable(build_manifest(yaml_dict).architecture_function)


def test_build_manifest_calibration_funcs_are_resolved_to_callables() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    config = build_manifest(yaml_dict).prediction_calibration_config
    assert config.calibration_func is sklearn_probability_calibrator
    assert config.threshold_func is grid_threshold_optimizer


def test_build_manifest_calibration_limen_path_param_resolved_to_callable() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    tp = build_manifest(yaml_dict).prediction_calibration_config.threshold_params
    assert tp['metric'] is balanced_metric


def test_build_manifest_calibration_round_param_refs_stored_as_strings() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    config = build_manifest(yaml_dict).prediction_calibration_config
    assert config.threshold_params['threshold_min'] == '{threshold_min}'
    assert config.threshold_params['threshold_max'] == '{threshold_max}'
    assert config.calibration_params['method'] == '{cal_method}'


def test_build_manifest_rule_based_returns_rule_based_manifest() -> None:
    yaml_dict, _ = parse(_MINIMAL_RULE_BASED_YAML)
    assert isinstance(build_manifest(yaml_dict), RuleBasedManifest)


def test_build_manifest_rule_based_strategy_and_architecture_wired() -> None:
    yaml_dict, _ = parse(_MINIMAL_RULE_BASED_YAML)
    manifest = build_manifest(yaml_dict)
    assert manifest.strategy is not None
    assert callable(manifest.architecture_function)


def test_compiled_sfd_name_is_yaml_prefixed() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    assert CompiledSFD(yaml_dict).__name__ == 'yaml:test_exp'


def test_compiled_sfd_params_are_non_empty_lists() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    for key, values in CompiledSFD(yaml_dict).params().items():
        assert isinstance(values, list) and len(values) > 0, f"Param '{key}' is not a non-empty list"


def test_compiled_sfd_manifest_is_cached() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    sfd = CompiledSFD(yaml_dict)
    assert sfd.manifest() is sfd.manifest()


def test_compiled_sfd_manifest_is_ml_manifest() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    assert isinstance(CompiledSFD(yaml_dict).manifest(), MLManifest)


def test_validate_error_when_both_split_config_and_split_dates_present() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['split_dates'] = {
        'train_start': '2022-01-01', 'train_end': '2022-07-01',
        'val_start': '2022-07-01', 'val_end': '2022-10-01',
        'test_start': '2022-10-01', 'test_end': '2023-01-01',
    }
    result = validate(yaml_dict)
    assert not result.valid
    assert any('split_config' in e.message and 'split_dates' in e.message for e in result.errors)


def test_validate_error_when_neither_split_config_nor_split_dates_present() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    del yaml_dict['sfd']['manifest']['split_config']
    result = validate(yaml_dict)
    assert not result.valid
    assert any('split_config' in e.message or 'split_dates' in e.message for e in result.errors)


def test_build_manifest_split_dates_calls_set_split_dates() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    del yaml_dict['sfd']['manifest']['split_config']
    yaml_dict['sfd']['manifest']['split_dates'] = {
        'train_start': '2022-01-01', 'train_end': '2022-07-01',
        'val_start': '2022-07-01', 'val_end': '2022-10-01',
        'test_start': '2022-10-01', 'test_end': '2023-01-01',
    }
    manifest = build_manifest(yaml_dict)
    assert manifest.split_dates is not None
    assert manifest.split_dates[0] == date.fromisoformat('2022-01-01')
    assert manifest.split_dates[5] == date.fromisoformat('2023-01-01')


def test_build_manifest_pca_compression_configured() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['pca_compression'] = {}
    manifest = build_manifest(yaml_dict)
    assert isinstance(manifest, MLManifest)
    assert manifest.pca_compression_config is not None


def test_parse_empty_yaml_returns_error() -> None:
    _, errors = parse('# no content\n')
    assert any('empty' in e.message.lower() for e in errors)


def test_parse_non_mapping_root_returns_error() -> None:
    _, errors = parse('- item1\n- item2\n')
    assert len(errors) > 0


def test_validation_error_carries_errors_and_formats_message() -> None:
    errors = [YAMLError(message='missing field', path='sfd.manifest')]
    exc = ValidationError(errors)
    assert exc.errors is errors
    assert 'missing field' in str(exc)


def test_git_error_carries_path_and_message() -> None:
    exc = GitError(path='exp.yaml', message='not a git repo')
    assert exc.path == 'exp.yaml'
    assert 'not a git repo' in str(exc)


def test_validate_passes_valid_yaml_with_split_dates() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    del yaml_dict['sfd']['manifest']['split_config']
    yaml_dict['sfd']['manifest']['split_dates'] = {
        'train_start': '2022-01-01', 'train_end': '2022-07-01',
        'val_start': '2022-07-01', 'val_end': '2022-10-01',
        'test_start': '2022-10-01', 'test_end': '2023-01-01',
    }
    result = validate(yaml_dict)
    assert result.valid


def test_validate_error_for_invalid_split_date_format() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    del yaml_dict['sfd']['manifest']['split_config']
    yaml_dict['sfd']['manifest']['split_dates'] = {
        'train_start': '01/01/2022', 'train_end': '2022-07-01',
        'val_start': '2022-07-01', 'val_end': '2022-10-01',
        'test_start': '2022-10-01', 'test_end': '2023-01-01',
    }
    result = validate(yaml_dict)
    assert not result.valid
    assert any('train_start' in e.message for e in result.errors)


def test_validate_error_for_split_config_train_zero() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['split_config']['train'] = 0
    result = validate(yaml_dict)
    assert not result.valid
    assert any('train' in e.message for e in result.errors)


def test_validate_error_for_split_config_negative_val() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['split_config']['val'] = -1
    result = validate(yaml_dict)
    assert not result.valid
    assert any('val' in e.message for e in result.errors)


def test_validate_error_for_split_dates_out_of_order() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    del yaml_dict['sfd']['manifest']['split_config']
    yaml_dict['sfd']['manifest']['split_dates'] = {
        'train_start': '2022-07-01', 'train_end': '2022-01-01',
        'val_start': '2022-07-01', 'val_end': '2022-10-01',
        'test_start': '2022-10-01', 'test_end': '2023-01-01',
    }
    result = validate(yaml_dict)
    assert not result.valid
    assert any('non-decreasing' in e.message for e in result.errors)


def test_validate_warning_for_unknown_key_in_data_source() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['data_source']['typo_key'] = 'value'
    result = validate(yaml_dict)
    assert any('typo_key' in w.message for w in result.warnings)


def test_validate_warning_for_unknown_key_in_split_config() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['split_config']['typo_key'] = 1
    result = validate(yaml_dict)
    assert any('typo_key' in w.message for w in result.warnings)


def test_validate_passes_valid_rule_based_yaml() -> None:
    yaml_dict, _ = parse(_MINIMAL_RULE_BASED_YAML)
    result = validate(yaml_dict)
    assert result.valid
    assert result.errors == []


def test_validate_error_for_empty_sfd_param_list() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['params']['roc_period'] = []
    result = validate(yaml_dict)
    assert not result.valid
    assert any('roc_period' in e.message for e in result.errors)


def test_validate_error_for_missing_mode() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    del yaml_dict['metadata']['mode']
    result = validate(yaml_dict)
    assert not result.valid
    assert any('mode' in e.message for e in result.errors)


def test_validate_error_for_unsafe_metadata_name() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['metadata']['name'] = '../attack'
    result = validate(yaml_dict)
    assert not result.valid
    assert any('metadata.name' in e.path for e in result.errors)


def test_validate_no_warning_for_literal_calibration_param() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['calibration']['probability_calibration']['params']['method'] = 'isotonic'
    result = validate(yaml_dict)
    assert not any('isotonic' in w.message for w in result.warnings)


def test_validate_error_for_empty_calibration_block() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['calibration'] = {}
    result = validate(yaml_dict)
    assert not result.valid
    assert any('calibration' in e.message for e in result.errors)


def test_validate_error_for_unresolvable_reference_architecture() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['reference_architecture'] = 'limen.sfd.reference_architecture.nonexistent_arch'
    result = validate(yaml_dict)
    assert not result.valid
    assert any('reference_architecture' in e.path for e in result.errors)


def test_validate_error_for_unresolvable_target_class() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['target']['class'] = 'limen.targets.NoSuchTarget'
    result = validate(yaml_dict)
    assert not result.valid
    assert any('target.class' in e.path for e in result.errors)


def test_validate_error_for_scaler_missing_both_from_params_and_class() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['scaler'] = {'unknown_key': 'value'}
    result = validate(yaml_dict)
    assert not result.valid
    assert any('scaler' in e.path and 'from_params' in e.message for e in result.errors)


def test_validate_error_for_scaler_with_both_from_params_and_class() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['scaler'] = {
        'from_params': 'scaler_type',
        'class': 'limen.scalers.StandardScaler',
    }
    result = validate(yaml_dict)
    assert not result.valid
    assert any('scaler' in e.path and 'both' in e.message for e in result.errors)


def test_validate_error_for_calibration_func_missing() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['calibration']['probability_calibration'] = {'params': {}}
    result = validate(yaml_dict)
    assert not result.valid
    assert any('probability_calibration.func' in e.path for e in result.errors)


def test_validate_error_for_data_source_params_not_a_mapping() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['data_source']['params'] = ['not', 'a', 'dict']
    result = validate(yaml_dict)
    assert not result.valid
    assert any('data_source.params' in e.path for e in result.errors)

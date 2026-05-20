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


def test_parse_empty_yaml_returns_error() -> None:
    _, errors = parse('# no content\n')
    assert any('empty' in e.message.lower() for e in errors)


def test_parse_non_mapping_root_returns_error() -> None:
    _, errors = parse('- item1\n- item2\n')
    assert len(errors) > 0


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


def test_resolve_returns_correct_object_types() -> None:
    import inspect
    assert callable(resolve('limen.indicators.roc'))
    assert inspect.ismodule(resolve('limen.indicators'))


def test_validation_error_carries_errors_and_formats_message() -> None:
    errors = [YAMLError(message='missing field', path='sfd.manifest')]
    exc = ValidationError(errors)
    assert exc.errors is errors
    assert 'missing field' in str(exc)


def test_git_error_carries_path_and_message() -> None:
    exc = GitError(path='exp.yaml', message='not a git repo')
    assert exc.path == 'exp.yaml'
    assert 'not a git repo' in str(exc)


def test_resolve_func_params_resolves_limen_path_to_callable() -> None:
    resolved = _resolve_func_params({'metric': 'limen.metrics.balanced_metric.balanced_metric'})
    assert resolved['metric'] is balanced_metric


def test_resolve_func_params_preserves_round_param_ref_as_string() -> None:
    params = {'threshold_min': '{threshold_min}', 'threshold_max': '{threshold_max}'}
    resolved = _resolve_func_params(params)
    assert resolved['threshold_min'] == '{threshold_min}'
    assert resolved['threshold_max'] == '{threshold_max}'


def test_resolve_func_params_preserves_non_limen_string_on_resolution_failure() -> None:
    resolved = _resolve_func_params({'key': 'some_literal_value'})
    assert resolved['key'] == 'some_literal_value'


def test_resolve_func_params_raises_on_unresolvable_limen_path() -> None:
    from limen.yaml.errors import ResolutionError
    try:
        _resolve_func_params({'metric': 'limen.metrics.nonexistent_function'})
        assert False, 'Expected ResolutionError'
    except ResolutionError:
        pass


def test_resolve_func_params_raises_on_limen_module_path() -> None:
    try:
        _resolve_func_params({'metric': 'limen.metrics'})
        assert False, 'Expected ValueError'
    except ValueError:
        pass


def test_validate_passes_valid_ml_yaml() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    result = validate(yaml_dict)
    assert result.valid
    assert result.errors == []


def test_validate_passes_valid_rule_based_yaml() -> None:
    yaml_dict, _ = parse(_MINIMAL_RULE_BASED_YAML)
    result = validate(yaml_dict)
    assert result.valid
    assert result.errors == []


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


def test_validate_error_for_missing_required_field() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    del yaml_dict['sfd']['manifest']['split_config']
    result = validate(yaml_dict)
    assert not result.valid
    assert any('split_config' in e.message for e in result.errors)


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


def test_validate_error_for_split_config_invalid_value() -> None:
    for key, value in [('train', 0), ('val', -1)]:
        yaml_dict, _ = parse(_MINIMAL_ML_YAML)
        yaml_dict['sfd']['manifest']['split_config'][key] = value
        result = validate(yaml_dict)
        assert not result.valid
        assert any(key in e.message for e in result.errors)


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


def test_validate_error_for_empty_sfd_param_list() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['params']['roc_period'] = []
    result = validate(yaml_dict)
    assert not result.valid
    assert any('roc_period' in e.message for e in result.errors)


def test_validate_warning_for_unused_sfd_param() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['params']['orphan_param'] = [1, 2, 3]
    result = validate(yaml_dict)
    assert any('orphan_param' in w.message for w in result.warnings)


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


def test_validate_error_for_callable_path_resolving_to_module() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['reference_architecture'] = 'limen.sfd.reference_architecture'
    assert any('module' in e.message for e in validate(yaml_dict).errors)

    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['data_source']['method'] = 'limen.data'
    assert any('module' in e.message for e in validate(yaml_dict).errors)

    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['indicators'][0]['func'] = 'limen.indicators'
    assert any('module' in e.message for e in validate(yaml_dict).errors)


def test_validate_error_for_func_not_a_string() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['data_source']['method'] = 42
    assert any('data_source.method' in e.path for e in validate(yaml_dict).errors)

    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['indicators'][0]['func'] = 123
    assert any('indicators' in e.path for e in validate(yaml_dict).errors)

    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['pre_split_data_selector'] = {'func': 99}
    assert any('pre_split_data_selector' in e.path for e in validate(yaml_dict).errors)

    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['calibration'] = {'probability_calibration': {'func': 42, 'params': {}}}
    assert any('calibration' in e.path for e in validate(yaml_dict).errors)


def test_validate_error_for_func_block_missing_func() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['calibration']['probability_calibration'] = {'params': {}}
    assert any('probability_calibration.func' in e.path for e in validate(yaml_dict).errors)

    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['pre_split_data_selector'] = {'params': {}}
    assert any('pre_split_data_selector.func' in e.path for e in validate(yaml_dict).errors)

    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['bar_formation'] = {'params': {}}
    assert any('bar_formation.func' in e.path for e in validate(yaml_dict).errors)

    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['data_dict_extension'] = {}
    assert any('data_dict_extension.func' in e.path for e in validate(yaml_dict).errors)


def test_validate_error_for_func_params_containing_module_path() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['indicators'][0]['params']['metric'] = 'limen.metrics'
    assert any('indicators' in e.path for e in validate(yaml_dict).errors)

    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['pre_split_data_selector'] = {
        'func': 'limen.data.HistoricalData.get_spot_klines',
        'params': {'metric': 'limen.metrics'},
    }
    assert any('pre_split_data_selector' in e.path for e in validate(yaml_dict).errors)


def test_validate_error_for_data_source_params_not_a_mapping() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['data_source']['params'] = ['not', 'a', 'dict']
    result = validate(yaml_dict)
    assert not result.valid
    assert any('data_source.params' in e.path for e in result.errors)


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


def test_validate_error_for_scaler_not_a_mapping() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['scaler'] = 'logreg'
    result = validate(yaml_dict)
    assert not result.valid
    assert any('scaler' in e.path and 'mapping' in e.message for e in result.errors)


def test_validate_error_for_scaler_from_params_invalid_value() -> None:
    for value in ['', 42]:
        yaml_dict, _ = parse(_MINIMAL_ML_YAML)
        yaml_dict['sfd']['manifest']['scaler'] = {'from_params': value}
        result = validate(yaml_dict)
        assert not result.valid
        assert any('scaler' in e.path for e in result.errors)


def test_validate_error_for_calibration_not_a_mapping() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['calibration'] = 'isotonic'
    result = validate(yaml_dict)
    assert not result.valid
    assert any('calibration' in e.path and 'mapping' in e.message for e in result.errors)


def test_validate_error_for_calibration_section_not_a_mapping() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['calibration'] = {'probability_calibration': 'isotonic'}
    result = validate(yaml_dict)
    assert not result.valid
    assert any('probability_calibration' in e.path for e in result.errors)


def test_validate_error_for_calibration_params_not_a_mapping() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['calibration'] = {
        'probability_calibration': {
            'func': 'limen.calibration.sklearn_probability_calibrator',
            'params': 'isotonic',
        }
    }
    result = validate(yaml_dict)
    assert not result.valid
    assert any('probability_calibration.params' in e.path for e in result.errors)


def test_validate_error_for_empty_calibration_block() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['calibration'] = {}
    result = validate(yaml_dict)
    assert not result.valid
    assert any('calibration' in e.message for e in result.errors)


def test_validate_error_for_calibration_param_ref_not_in_sfd_params() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['calibration']['threshold_function']['params']['bogus'] = '{nonexistent_param}'
    result = validate(yaml_dict)
    assert not result.valid
    assert any('nonexistent_param' in e.message for e in result.errors)


def test_validate_no_warning_for_literal_calibration_param() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['calibration']['probability_calibration']['params']['method'] = 'isotonic'
    result = validate(yaml_dict)
    assert not any('isotonic' in w.message for w in result.warnings)


def test_validate_error_for_optional_manifest_block_not_a_mapping() -> None:
    for block, value in [
        ('feature_ablation', 'all'),
        ('pca_compression', 'auto'),
        ('params_override', 'bad'),
        ('metrics_params', 42),
    ]:
        yaml_dict, _ = parse(_MINIMAL_ML_YAML)
        yaml_dict['sfd']['manifest'][block] = value
        result = validate(yaml_dict)
        assert not result.valid, f'{block} should require a mapping'
        assert any(block in e.path for e in result.errors)


def test_validate_error_for_pca_compression_unknown_key() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['pca_compression'] = {'enabled_param': 'auto_pca', 'bogus_key': 1}
    result = validate(yaml_dict)
    assert not result.valid
    assert any('bogus_key' in e.message for e in result.errors)


def test_validate_error_for_feature_ablation_key_field_invalid_value() -> None:
    for field_name, value in [('drop_count_key', 123), ('drop_count_key', ''), ('seed_key', None)]:
        yaml_dict, _ = parse(_MINIMAL_ML_YAML)
        yaml_dict['sfd']['manifest']['feature_ablation'] = {field_name: value}
        result = validate(yaml_dict)
        assert not result.valid, f'feature_ablation.{field_name}={value!r} should be invalid'
        assert any(field_name in e.path for e in result.errors)


def test_validate_error_for_pca_compression_key_field_invalid_value() -> None:
    for field_name, value in [('enabled_param', 123), ('enabled_param', ''), ('n_components_param', False)]:
        yaml_dict, _ = parse(_MINIMAL_ML_YAML)
        yaml_dict['sfd']['manifest']['pca_compression'] = {field_name: value}
        result = validate(yaml_dict)
        assert not result.valid, f'pca_compression.{field_name}={value!r} should be invalid'
        assert any(field_name in e.path for e in result.errors)


def test_validate_warning_for_unknown_key_in_func_block() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['indicators'][0]['unexpected_key'] = 'value'
    result = validate(yaml_dict)
    assert any('unexpected_key' in w.message for w in result.warnings)


def test_validate_no_warning_for_func_block_with_only_known_keys() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    result = validate(yaml_dict)
    assert not any('indicators[0]' in w.path and 'Unknown' in w.message for w in result.warnings)


def test_validate_no_warning_for_include_if_on_indicator() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['indicators'][0]['include_if'] = 'use_roc'
    result = validate(yaml_dict)
    assert not any('include_if' in w.message for w in result.warnings)


def test_validate_error_for_include_if_invalid_value() -> None:
    for value in [123, '']:
        yaml_dict, _ = parse(_MINIMAL_ML_YAML)
        yaml_dict['sfd']['manifest']['features'][0]['include_if'] = value
        result = validate(yaml_dict)
        assert not result.valid, f'include_if={value!r} should be invalid'
        assert any('include_if' in e.path for e in result.errors)


def test_validate_error_for_required_columns_not_a_list() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['required_columns'] = 'close'
    result = validate(yaml_dict)
    assert not result.valid
    assert any('required_columns' in e.path for e in result.errors)


def test_validate_error_for_data_dict_extension_unknown_key() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['data_dict_extension'] = {
        'func': 'limen.data.HistoricalData.get_spot_klines',
        'params': {},
        'extra_key': 'bad',
    }
    result = validate(yaml_dict)
    assert not result.valid
    assert any('data_dict_extension' in e.path for e in result.errors)


def test_validate_error_for_uel_search_strategy_wrong_type() -> None:
    for value in ['random', None]:
        yaml_dict, _ = parse(_MINIMAL_ML_YAML)
        yaml_dict['uel']['search_strategy'] = value
        result = validate(yaml_dict)
        assert not result.valid
        assert any('search_strategy' in e.path for e in result.errors)


def test_validate_error_for_uel_output_path_wrong_type() -> None:
    for value in [123, None]:
        yaml_dict, _ = parse(_MINIMAL_ML_YAML)
        yaml_dict['uel']['output_path'] = value
        result = validate(yaml_dict)
        assert not result.valid
        assert any('output_path' in e.path for e in result.errors)


def test_validate_error_for_uel_prep_each_round_wrong_type() -> None:
    for value in ['yes', None]:
        yaml_dict, _ = parse(_MINIMAL_ML_YAML)
        yaml_dict['uel']['prep_each_round'] = value
        result = validate(yaml_dict)
        assert not result.valid
        assert any('prep_each_round' in e.path for e in result.errors)


def test_validate_error_for_uel_feedback_interval_wrong_type() -> None:
    for value in ['100', None, True]:
        yaml_dict, _ = parse(_MINIMAL_ML_YAML)
        yaml_dict['uel']['feedback_interval'] = value
        result = validate(yaml_dict)
        assert not result.valid, f'feedback_interval={value!r} should be invalid'
        assert any('feedback_interval' in e.path for e in result.errors)


def test_validate_error_for_uel_checkpoint_interval_wrong_type() -> None:
    for value in ['500', None, True]:
        yaml_dict, _ = parse(_MINIMAL_ML_YAML)
        yaml_dict['uel']['checkpoint_interval'] = value
        result = validate(yaml_dict)
        assert not result.valid, f'checkpoint_interval={value!r} should be invalid'
        assert any('checkpoint_interval' in e.path for e in result.errors)


def test_validate_error_for_manifest_ref_not_in_sfd_params() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['indicators'][0]['params']['period'] = '{unknown_param}'
    result = validate(yaml_dict)
    assert not result.valid
    assert any('unknown_param' in e.message for e in result.errors)


def test_validate_no_error_when_manifest_ref_is_in_sfd_params() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['indicators'][0]['params']['period'] = '{new_param}'
    yaml_dict['sfd']['params']['new_param'] = [1, 2, 3]
    result = validate(yaml_dict)
    assert not any('new_param' in e.message for e in result.errors)


def test_validate_no_unused_param_warning_for_pca_defaults() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['pca_compression'] = {}
    yaml_dict['sfd']['params']['auto_pca'] = [True, False]
    yaml_dict['sfd']['params']['pca_k'] = [5, 10]
    result = validate(yaml_dict)
    assert not any(w.path in ('sfd.params.auto_pca', 'sfd.params.pca_k') for w in result.warnings)


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


def test_build_manifest_indicator_include_if_stored_on_entry() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['indicators'][0]['include_if'] = 'use_roc'
    manifest = build_manifest(yaml_dict)
    roc_entry = next(e for e in manifest.feature_transforms if 'roc' in e.func.__name__)
    assert roc_entry.include_if == 'use_roc'


def test_build_manifest_indicator_include_if_none_when_absent() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    manifest = build_manifest(yaml_dict)
    roc_entry = next(e for e in manifest.feature_transforms if 'roc' in e.func.__name__)
    assert roc_entry.include_if is None


def test_build_manifest_feature_include_if_stored_on_entry() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['features'][0]['include_if'] = 'use_frac_diff'
    yaml_dict['sfd']['params']['use_frac_diff'] = [True, False]
    manifest = build_manifest(yaml_dict)
    fd_entry = next(e for e in manifest.feature_transforms if 'fractional' in e.func.__name__)
    assert fd_entry.include_if == 'use_frac_diff'


def test_validate_no_error_when_include_if_key_is_in_sfd_params() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['indicators'][0]['include_if'] = 'use_roc'
    yaml_dict['sfd']['params']['use_roc'] = [True, False]
    result = validate(yaml_dict)
    assert result.valid
    assert not any('use_roc' in e.message for e in result.errors)


def test_validate_error_when_include_if_key_not_in_sfd_params() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['indicators'][0]['include_if'] = 'use_roc'
    result = validate(yaml_dict)
    assert not result.valid
    assert any('use_roc' in e.message for e in result.errors)


def test_validate_error_when_include_if_param_values_are_not_bool() -> None:
    for bad_values in [[1, 0], ['yes', 'no'], [1]]:
        yaml_dict, _ = parse(_MINIMAL_ML_YAML)
        yaml_dict['sfd']['manifest']['indicators'][0]['include_if'] = 'use_roc'
        yaml_dict['sfd']['params']['use_roc'] = bad_values
        result = validate(yaml_dict)
        assert not result.valid, f'use_roc={bad_values!r} should be invalid'
        assert any('use_roc' in e.path for e in result.errors)


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


def test_build_manifest_pca_compression_configured() -> None:
    yaml_dict, _ = parse(_MINIMAL_ML_YAML)
    yaml_dict['sfd']['manifest']['pca_compression'] = {}
    manifest = build_manifest(yaml_dict)
    assert isinstance(manifest, MLManifest)
    assert manifest.pca_compression_config is not None


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

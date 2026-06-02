from pathlib import Path
from unittest.mock import MagicMock

from limen.yaml.compiler import CompiledSFD
from limen.yaml.parser import parse
from limen.yaml.profiler import ProfileResult
from limen.yaml.profiler import _classify_error
from limen.yaml.profiler import _complexity_rating
from limen.yaml.profiler import make_covering_array
from limen.yaml.profiler import profile

_TEMPLATES_DIR = Path(__file__).resolve().parents[1] / 'limen' / 'yaml' / 'templates'


def test_complexity_rating_low() -> None:
    assert _complexity_rating(1) == 'low'
    assert _complexity_rating(100) == 'low'


def test_complexity_rating_medium() -> None:
    assert _complexity_rating(101) == 'medium'
    assert _complexity_rating(1000) == 'medium'


def test_complexity_rating_high() -> None:
    assert _complexity_rating(1001) == 'high'
    assert _complexity_rating(10000) == 'high'


def test_complexity_rating_extreme() -> None:
    assert _complexity_rating(10001) == 'extreme'
    assert _complexity_rating(10 ** 9) == 'extreme'


def test_covering_array_covers_all_values() -> None:
    params = {
        'a': [1, 2, 3],
        'b': ['x', 'y'],
        'c': [True, False, True, False],
    }
    array = make_covering_array(params)
    assert len(array) == 4
    for key, values in params.items():
        covered = {row[key] for row in array}
        assert covered == set(values), f"param {key!r} not fully covered"


def test_covering_array_length_equals_max_cardinality() -> None:
    params = {'x': [1, 2, 3, 4, 5], 'y': [10, 20]}
    assert len(make_covering_array(params)) == 5


def test_covering_array_empty_params_returns_empty() -> None:
    assert make_covering_array({}) == []


def test_covering_array_single_param() -> None:
    params = {'p': [10, 20, 30]}
    array = make_covering_array(params)
    assert len(array) == 3
    assert {row['p'] for row in array} == {10, 20, 30}


def test_covering_array_is_reproducible_with_same_seed() -> None:
    params = {'a': [1, 2, 3], 'b': ['x', 'y', 'z']}
    assert make_covering_array(params, seed=7) == make_covering_array(params, seed=7)


def test_covering_array_differs_with_different_seeds() -> None:
    params = {'a': [1, 2, 3, 4, 5], 'b': ['x', 'y', 'z', 'w', 'v']}
    assert make_covering_array(params, seed=1) != make_covering_array(params, seed=2)


def test_covering_array_columns_are_independently_shuffled() -> None:
    # Without independent shuffle, all params would share the same rank order.
    params = {'a': [1, 2, 3, 4, 5], 'b': [10, 20, 30, 40, 50]}
    array = make_covering_array(params, seed=42)
    a_order = [row['a'] for row in array]
    b_order = [row['b'] for row in array]
    # rank-order comparison: identical rank order ↔ columns shuffled as one
    a_ranks = sorted(range(len(a_order)), key=lambda i: a_order[i])
    b_ranks = sorted(range(len(b_order)), key=lambda i: b_order[i])
    assert a_ranks != b_ranks


def test_covering_array_all_rows_have_all_keys() -> None:
    params = {'x': [1, 2], 'y': ['a', 'b', 'c'], 'z': [True]}
    array = make_covering_array(params)
    for row in array:
        assert set(row.keys()) == {'x', 'y', 'z'}


def test_covering_array_values_are_from_original_lists() -> None:
    params = {'p': [0.1, 0.5, 1.0], 'q': ['foo', 'bar']}
    array = make_covering_array(params)
    for row in array:
        assert row['p'] in params['p']
        assert row['q'] in params['q']


def test_classify_error_small_dataset() -> None:
    msg = _classify_error(ValueError('Found array with 0 sample(s)'))
    assert 'n_rows' in msg


def test_classify_error_class_diversity() -> None:
    msg = _classify_error(ValueError('The number of classes has to be at least 2'))
    assert 'class diversity' in msg


def test_classify_error_nan() -> None:
    msg = _classify_error(ValueError('Input contains NaN'))
    assert 'NaN' in msg


def test_classify_error_generic() -> None:
    msg = _classify_error(RuntimeError('something unexpected'))
    assert 'RuntimeError' in msg
    assert 'unexpected' in msg


def test_classify_error_no_false_positive_on_numeric_1_in_message() -> None:
    # "1" appears in learning_rate value — must not match class-diversity branch
    msg = _classify_error(ValueError('XGBClassifier: learning_rate must be > 0.1'))
    assert 'class diversity' not in msg


def test_classify_error_no_false_positive_on_information_in_message() -> None:
    # "inf" appears inside "information" — must not match NaN/Inf branch
    msg = _classify_error(ValueError('For more information see sklearn docs'))
    assert 'NaN' not in msg
    assert 'Inf' not in msg


def test_profile_static_fields_correct_for_logreg_template() -> None:
    template = _TEMPLATES_DIR / 'logreg_binary.yaml'
    yaml_dict, _ = parse(template.read_text(encoding='utf-8'))
    sfd = CompiledSFD(yaml_dict)
    result = profile(sfd)

    assert isinstance(result, ProfileResult)
    assert result.total_permutations > 0
    assert result.complexity_rating in ('low', 'medium', 'high', 'extreme')
    assert set(result.param_cardinalities.keys()) == set(yaml_dict['sfd']['params'].keys())
    for key, values in yaml_dict['sfd']['params'].items():
        assert result.param_cardinalities[key] == len(list(values))


def test_profile_total_permutations_is_product_of_cardinalities() -> None:
    template = _TEMPLATES_DIR / 'logreg_binary.yaml'
    yaml_dict, _ = parse(template.read_text(encoding='utf-8'))
    sfd = CompiledSFD(yaml_dict)
    result = profile(sfd)

    expected = 1
    for v in result.param_cardinalities.values():
        expected *= v
    assert result.total_permutations == expected


def test_profile_warns_when_test_data_source_absent() -> None:
    from textwrap import dedent
    yaml_str = dedent('''\
        schema_version: "1.0"
        metadata:
          name: no_test_src
          mode: development
        sfd:
          manifest:
            type: ml
            data_source:
              method: limen.data.HistoricalData.get_spot_klines
              params:
                kline_size: 3600
            split_dates:
              train_start: "2024-01-01"
              train_end: "2024-09-30"
              val_start: "2024-10-01"
              val_end: "2024-11-30"
              test_start: "2024-12-01"
              test_end: "2024-12-31"
            target:
              name: quantile_flag
              class: limen.targets.QuantileBinaryTarget
              fit_params:
                source_column: close
                quantile: 0.5
              transform_params:
                shift: -1
            reference_architecture: limen.sfd.reference_architecture.logreg_binary
          params:
            max_iter: [100]
            solver: [lbfgs]
            penalty: [l2]
            dual: [false]
            tol: [0.01]
            C: [1.0]
            fit_intercept: [true]
            intercept_scaling: [1.0]
            class_weight: [0.5]
            random_state: [42]
            multi_class: [deprecated]
            verbose: [0]
            warm_start: [false]
            n_jobs: [-1]
            l1_ratio: [null]
        uel:
          n_permutations: 5
          search_strategy:
            type: random
          output_format: csv
    ''')
    yaml_dict, _ = parse(yaml_str)
    sfd = CompiledSFD(yaml_dict)
    result = profile(sfd)

    assert any('test_data_source' in w for w in result.warnings)
    assert result.sample_permutations_attempted == 0
    assert result.sample_time_seconds_per_permutation is None


def _make_mock_manifest(fail: bool = False) -> MagicMock:
    manifest = MagicMock()
    manifest.test_data_source_config = MagicMock()
    manifest.fetch_test_data.return_value = MagicMock()
    manifest.prepare_data.return_value = {'x_train': [], 'y_train': []}
    if fail:
        manifest.run_model.side_effect = ValueError('Found array with 0 sample(s)')
    else:
        manifest.run_model.return_value = {'metric': 0.8}
    return manifest


def _make_mock_sfd(params: dict, manifest: MagicMock) -> MagicMock:
    sfd = MagicMock(spec=CompiledSFD)
    sfd.params.return_value = params
    sfd.manifest.return_value = manifest
    return sfd


def test_profile_runtime_records_timing_when_all_permutations_succeed() -> None:
    params = {'a': [1, 2, 3], 'b': ['x', 'y', 'z']}
    sfd = _make_mock_sfd(params, _make_mock_manifest())
    result = profile(sfd)

    assert result.sample_permutations_attempted == 3
    assert result.sample_permutations_completed == 3
    assert result.sample_time_seconds_per_permutation is not None
    assert result.sample_time_seconds_per_permutation >= 0.0
    assert result.errors == []


def test_profile_runtime_records_errors_when_all_permutations_fail() -> None:
    params = {'a': [1, 2, 3], 'b': ['x', 'y', 'z']}
    sfd = _make_mock_sfd(params, _make_mock_manifest(fail=True))
    result = profile(sfd)

    assert result.sample_permutations_completed == 0
    assert result.sample_time_seconds_per_permutation is None
    assert len(result.errors) == 3
    assert all('n_rows' in e for e in result.errors)


def test_profile_runtime_partial_failure_timing_from_completed_only() -> None:
    params = {'a': [1, 2, 3]}
    manifest = _make_mock_manifest()
    # Fail on second call only
    manifest.run_model.side_effect = [
        {'metric': 0.8},
        ValueError('Found array with 0 sample(s)'),
        {'metric': 0.7},
    ]
    sfd = _make_mock_sfd(params, manifest)
    result = profile(sfd)

    assert result.sample_permutations_completed == 2
    assert result.sample_permutations_attempted == 3
    assert result.sample_time_seconds_per_permutation is not None
    assert len(result.errors) == 1
    assert any('2 of 3' in w or '1 of 3' in w for w in result.warnings)


def test_profile_runtime_fetch_data_failure_reported_as_error() -> None:
    params = {'a': [1, 2]}
    manifest = _make_mock_manifest()
    manifest.fetch_test_data.side_effect = ConnectionError('network down')
    sfd = _make_mock_sfd(params, manifest)
    result = profile(sfd)

    assert any('Failed to load test data' in e for e in result.errors)
    assert result.sample_permutations_attempted == 0

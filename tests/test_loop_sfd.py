# ruff: noqa: E402
import json
import os
import sys
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

# Allow running this file directly: ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from limen import UniversalExperimentLoop
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import RandomStrategy
from limen.scalers import LinearScaler
from limen.sfd.loop import LoopSFD
from limen.sfd.loop.meta import (
    LABEL_TARGET_COLUMNS,
    SCALER_NAME_MAP,
    get_target_column,
)
from limen.sfd.loop.progress import make_progress_callback
from limen.sfd.loop.reference_defaults import get_reference_architecture_params
from limen.sfd.loop.registry import (
    FEATURE_REGISTRY,
    INDICATOR_REGISTRY,
    MODEL_REGISTRY,
    SCALER_REGISTRY,
)
from limen.sfd.reference_architecture import logreg_binary as _logreg_binary_func


_FIXTURE_PATH = Path(__file__).parent / 'fixtures' / 'loop_logreg_binary.json'


def _load_payload() -> dict:
    return json.loads(_FIXTURE_PATH.read_text())


def test_registry_indicator_resolution():
    from limen.indicators import bbands

    assert INDICATOR_REGISTRY['bbands'] is bbands


def test_registry_feature_resolution():
    from limen.features import atr_percent_sma
    from limen.features.forward_breakout_target import forward_breakout_target

    assert FEATURE_REGISTRY['atr_percent_sma'] is atr_percent_sma
    # forward_breakout_target is not in limen.features.__all__ — it reaches
    # the registry via the manual additions fallback in registry.py
    assert FEATURE_REGISTRY['forward_breakout_target'] is forward_breakout_target


def test_registry_model_resolution():
    assert MODEL_REGISTRY['logreg_binary'] is _logreg_binary_func


def test_registry_model_excludes_classes():
    # ReferenceModel and concrete classes should NOT be in MODEL_REGISTRY
    assert 'ReferenceModel' not in MODEL_REGISTRY
    assert 'LogRegBinary' not in MODEL_REGISTRY


def test_registry_scaler_has_linear():
    assert 'linear' in SCALER_REGISTRY
    assert SCALER_REGISTRY['linear'] is LinearScaler


def test_registry_unknown_name_raises():
    try:
        _ = INDICATOR_REGISTRY['this_does_not_exist_xyz']
    except KeyError:
        return
    raise AssertionError('Expected KeyError for unknown indicator')


def test_label_meta_known():
    assert get_target_column('forward_breakout_target') == 'forward_breakout'


def test_label_meta_fallback_to_name():
    assert get_target_column('quantile_flag') == 'quantile_flag'


def test_label_meta_table_has_known_label():
    assert 'forward_breakout_target' in LABEL_TARGET_COLUMNS


def test_scaler_name_map_completeness():
    assert SCALER_NAME_MAP['LinearScaler'] == 'linear'
    assert SCALER_NAME_MAP['LogRegScaler'] == 'logreg'
    assert SCALER_NAME_MAP['RobustScaler'] == 'robust'
    assert SCALER_NAME_MAP['RankGaussScaler'] == 'rank_gauss'


def test_reference_defaults_logreg_binary_extracts_model_params():
    defaults = get_reference_architecture_params('logreg_binary', _logreg_binary_func)

    # All keys must be valid logreg_binary signature parameter names
    import inspect
    sig_names = set(inspect.signature(_logreg_binary_func).parameters)
    for key in defaults:
        assert key in sig_names, f"key '{key}' is not in logreg_binary signature"

    # Must include the actual model hyperparams
    for expected in ('C', 'solver', 'penalty', 'tol', 'max_iter'):
        assert expected in defaults, f"missing model param '{expected}'"

    # Must NOT include data-prep params from the foundational SFD
    for excluded in ('frac_diff_d', 'roc_period', 'q', 'shift', 'feature_groups',
                     'scaler_type'):
        assert excluded not in defaults, (
            f"data-prep param '{excluded}' should not leak into reference defaults"
        )


def test_reference_defaults_constructor_override():
    payload = _load_payload()
    custom = {'C': [0.5, 1.0], 'solver': ['liblinear']}
    sfd = LoopSFD(payload, reference_architecture_params=custom)

    params = sfd.params()
    assert params['C'] == [0.5, 1.0]
    assert params['solver'] == ['liblinear']


def test_loop_sfd_name_attribute():
    sfd = LoopSFD(_load_payload())
    assert sfd.__name__ == 'loop_logreg_binary'


def test_loop_sfd_params_filters_metadata_keys():
    sfd = LoopSFD(_load_payload())
    params = sfd.params()

    # UI form metadata is filtered out
    for key in params:
        assert not key.endswith('_selected_items'), (
            f"selected_items key leaked: {key}"
        )
    assert 'scaling_method' not in params

    # Split keys are filtered
    for key in params:
        assert not key.startswith('input_split_'), f"split key leaked: {key}"


def test_loop_sfd_params_filters_arch_prefixed_keys():
    sfd = LoopSFD(_load_payload())
    params = sfd.params()

    # input_logreg_binary_* keys must not be in the merged params
    for key in params:
        assert not key.startswith('input_logreg_binary_'), (
            f"reference architecture prefix key leaked: {key}"
        )


def test_loop_sfd_params_includes_component_namespaced_keys():
    sfd = LoopSFD(_load_payload())
    params = sfd.params()

    expected = [
        'indicator_bbands_period',
        'indicator_bbands_nb_dev_up',
        'feature_atr_percent_sma_period',
        'label_forward_breakout_target_forward_periods',
        'label_forward_breakout_target_threshold',
        'label_forward_breakout_target_shift',
    ]
    for key in expected:
        assert key in params, f"expected component key missing: {key}"


def test_loop_sfd_params_excludes_dropped_categories():
    sfd = LoopSFD(_load_payload())
    params = sfd.params()

    # reference_architecture is top-level UI metadata, not a model hyperparam
    assert 'reference_architecture' not in params

    # scaler_* keys came from selectedItems UI metadata; we use scalingMethod
    for key in params:
        assert not key.startswith('scaler_'), f"scaler_* key leaked: {key}"

    # transforms are intentionally ignored in this iteration
    for key in params:
        assert not key.startswith('transform_'), f"transform_* key leaked: {key}"


def test_loop_sfd_params_includes_unnamespaced_model_keys():
    sfd = LoopSFD(_load_payload())
    params = sfd.params()

    for expected in ('C', 'solver', 'penalty', 'tol', 'max_iter'):
        assert expected in params, f"expected model param missing: {expected}"


def test_loop_sfd_manifest_split_config():
    sfd = LoopSFD(_load_payload())
    m = sfd.manifest()
    assert m.split_config == (70, 15, 15)


def test_loop_sfd_manifest_target_column():
    sfd = LoopSFD(_load_payload())
    m = sfd.manifest()
    assert m.target_column == 'forward_breakout'


def test_loop_sfd_manifest_model_function():
    sfd = LoopSFD(_load_payload())
    m = sfd.manifest()
    assert m.model_function is _logreg_binary_func


def test_loop_sfd_manifest_scaler_set():
    sfd = LoopSFD(_load_payload())
    m = sfd.manifest()
    assert m.scaler is not None


def test_loop_sfd_manifest_feature_transforms_order():
    sfd = LoopSFD(_load_payload())
    m = sfd.manifest()

    feature_funcs = [entry.func for entry in m.feature_transforms]
    feature_names = [getattr(f, '__name__', repr(f)) for f in feature_funcs]

    # Indicators and features go into feature_transforms in declaration order
    assert 'bbands' in feature_names
    assert 'atr_percent_sma' in feature_names
    assert feature_names.index('bbands') < feature_names.index('atr_percent_sma')


def test_loop_sfd_ignores_payload_transforms():
    # The payload's transforms[] array is intentionally ignored in this
    # iteration. Verify mad_transform from the fixture appears in neither
    # feature_transforms nor target_transforms.
    sfd = LoopSFD(_load_payload())
    m = sfd.manifest()

    feature_names = [
        getattr(e.func, '__name__', '') for e in m.feature_transforms
    ]
    target_names = [
        getattr(e[1], '__name__', '') for e in m.target_transforms
    ]
    assert 'mad_transform' not in feature_names
    assert 'mad_transform' not in target_names


def test_loop_sfd_manifest_label_in_target_transforms():
    sfd = LoopSFD(_load_payload())
    m = sfd.manifest()

    funcs = [entry[1] for entry in m.target_transforms]
    func_names = [getattr(f, '__name__', repr(f)) for f in funcs]
    assert 'forward_breakout_target' in func_names


def test_loop_sfd_param_wiring_uses_namespaced_reference():
    sfd = LoopSFD(_load_payload())
    m = sfd.manifest()

    bbands_entry = next(
        e for e in m.feature_transforms
        if getattr(e.func, '__name__', '') == 'bbands'
    )
    # Each bbands param should be wired to its namespaced parameterSpace key
    assert bbands_entry.params['period'] == 'indicator_bbands_period'
    assert bbands_entry.params['nb_dev_up'] == 'indicator_bbands_nb_dev_up'


def test_loop_sfd_param_wiring_label_params():
    sfd = LoopSFD(_load_payload())
    m = sfd.manifest()

    # Target transforms are tuples (fitted_params, func, params)
    label_entry = next(
        e for e in m.target_transforms
        if getattr(e[1], '__name__', '') == 'forward_breakout_target'
    )
    label_params = label_entry[2]
    assert label_params['forward_periods'] == 'label_forward_breakout_target_forward_periods'
    assert label_params['threshold'] == 'label_forward_breakout_target_threshold'
    assert label_params['shift'] == 'label_forward_breakout_target_shift'


def test_loop_sfd_param_wiring_literal_when_not_in_param_space():
    payload = _load_payload()
    # Add a bbands param that is NOT in parameterSpace
    payload['indicators'][0]['params']['extra_literal'] = 99
    sfd = LoopSFD(payload)
    m = sfd.manifest()

    bbands_entry = next(
        e for e in m.feature_transforms
        if getattr(e.func, '__name__', '') == 'bbands'
    )
    assert bbands_entry.params['extra_literal'] == 99


def test_loop_sfd_unknown_scaler_raises():
    payload = _load_payload()
    payload['scaler']['scalingMethod'] = 'NotARealScaler'
    sfd = LoopSFD(payload)
    try:
        sfd.manifest()
    except ValueError as e:
        assert 'Unknown scaler' in str(e)
        return
    raise AssertionError('Expected ValueError for unknown scaler')


def test_loop_sfd_unknown_reference_architecture_raises():
    payload = _load_payload()
    payload['referenceArchitecture'] = 'not_a_real_arch'
    try:
        LoopSFD(payload)
    except KeyError:
        return
    raise AssertionError('Expected KeyError for unknown reference architecture')


def test_progress_callback_writes_json():
    with TemporaryDirectory() as tmpdir:
        progress_file = Path(tmpdir) / 'progress.json'
        callback = make_progress_callback(progress_file, total=10)

        fake_log = pl.DataFrame({'a': [1, 2, 3]})
        callback(fake_log, None)

        assert progress_file.exists()
        data = json.loads(progress_file.read_text())
        assert data['completed'] == 3
        assert data['total'] == 10
        assert data['percent'] == 30.0
        assert 'updated_at' in data


def test_progress_callback_handles_none_log():
    with TemporaryDirectory() as tmpdir:
        progress_file = Path(tmpdir) / 'progress.json'
        callback = make_progress_callback(progress_file, total=10)
        callback(None, None)
        data = json.loads(progress_file.read_text())
        assert data['completed'] == 0
        assert data['percent'] == 0.0


def test_progress_callback_zero_total_no_div_by_zero():
    with TemporaryDirectory() as tmpdir:
        progress_file = Path(tmpdir) / 'progress.json'
        callback = make_progress_callback(progress_file, total=0)
        callback(pl.DataFrame({'a': [1]}), None)
        data = json.loads(progress_file.read_text())
        assert data['percent'] == 0.0


def _build_quantile_flag_payload(col: str = 'roc_1', q: float = 0.5) -> dict:
    # Minimal payload using quantile_flag as the label. We add a roc
    # indicator so the fitted target can threshold on a mean-reverting
    # column (`roc_1`) rather than a monotone one like `close` — this
    # keeps test/train label distributions balanced enough for binary
    # metrics to compute a full 2x2 confusion matrix.
    return {
        'parameterSpace': {
            'indicator_roc_period': [1],
            'label_quantile_flag_col': [col],
            'label_quantile_flag_q': [q],
        },
        'referenceArchitecture': 'logreg_binary',
        'inputData': {'splitRatios': {'train': 70, 'val': 15, 'test': 15}},
        'indicators': [{'name': 'roc', 'params': {'period': 1}}],
        'features': [],
        'labels': [{'name': 'quantile_flag', 'params': {'col': col, 'q': q}}],
        'scaler': {'scalingMethod': 'LinearScaler'},
    }


def test_quantile_flag_compiles_as_fitted_transform():
    sfd = LoopSFD(_build_quantile_flag_payload())
    m = sfd.manifest()

    assert m.target_column == 'quantile_flag'
    assert len(m.target_transforms) == 1

    fitted_params, func, base_params = m.target_transforms[0]

    # Fitted param is wired to compute_quantile_cutoff with col + q references
    assert len(fitted_params) == 1
    fp_name, fp_func, fp_params = fitted_params[0]
    assert fp_name == '_quantile_cutoff'
    assert fp_func.__name__ == 'compute_quantile_cutoff'
    assert fp_params['col'] == 'label_quantile_flag_col'
    assert fp_params['q'] == 'label_quantile_flag_q'

    # Transform func is quantile_flag with col + fitted cutoff injection
    assert func.__name__ == 'quantile_flag'
    assert base_params['col'] == 'label_quantile_flag_col'
    assert base_params['cutoff'] == '_quantile_cutoff'


def test_forward_breakout_label_remains_plain_add_transform():
    # Regression guard: the non-fitted label path must not route through
    # the fitted-label branch.
    sfd = LoopSFD(_load_payload())
    m = sfd.manifest()
    fitted_params, func, _base_params = m.target_transforms[0]
    assert fitted_params == []
    assert func.__name__ == 'forward_breakout_target'


def test_quantile_flag_params_routes_through_round_params():
    sfd = LoopSFD(_build_quantile_flag_payload(col='roc_1', q=0.3))
    params = sfd.params()
    assert params['label_quantile_flag_col'] == ['roc_1']
    assert params['label_quantile_flag_q'] == [0.3]


def test_quantile_flag_end_to_end_with_uel():
    os.environ['LOOP_ENV'] = 'test'

    payload = _build_quantile_flag_payload(col='roc_1', q=0.5)
    sfd = LoopSFD(payload)

    domain = ParamDomain(sfd.params())
    strategy = RandomStrategy(domain, seed=42)

    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir) / 'exp'
        experiment_dir.mkdir(parents=True)

        progress_file = experiment_dir / 'progress.json'
        progress_cb = make_progress_callback(progress_file, total=2)

        uel = UniversalExperimentLoop(
            sfd=sfd,
            search_strategy=strategy,
            feedback_interval=1,
            checkpoint_interval=1,
            experiment_dir=experiment_dir,
            intra_callback=progress_cb,
        )

        uel.run(
            experiment_name=str(experiment_dir / 'run'),
            n_permutations=2,
        )

        assert (experiment_dir / 'results.csv').exists()
        assert (experiment_dir / 'metadata.json').exists()
        assert progress_file.exists()
        data = json.loads(progress_file.read_text())
        assert data['total'] == 2


def test_loop_sfd_end_to_end_with_uel():
    # Force test data source path
    os.environ['LOOP_ENV'] = 'test'

    payload = _load_payload()
    sfd = LoopSFD(payload)

    domain = ParamDomain(sfd.params())
    strategy = RandomStrategy(domain, seed=42)

    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir) / 'exp'
        experiment_dir.mkdir(parents=True)

        progress_file = experiment_dir / 'progress.json'
        progress_cb = make_progress_callback(progress_file, total=2)

        uel = UniversalExperimentLoop(
            sfd=sfd,
            search_strategy=strategy,
            feedback_interval=1,
            checkpoint_interval=1,
            experiment_dir=experiment_dir,
            intra_callback=progress_cb,
        )

        uel.run(
            experiment_name=str(experiment_dir / 'run'),
            n_permutations=2,
        )

        assert (experiment_dir / 'results.csv').exists()
        assert (experiment_dir / 'metadata.json').exists()
        assert (experiment_dir / 'checkpoint.json').exists()

        assert progress_file.exists()
        data = json.loads(progress_file.read_text())
        assert data['completed'] >= 1
        assert data['total'] == 2


_TESTS = [
    test_registry_indicator_resolution,
    test_registry_feature_resolution,
    test_registry_model_resolution,
    test_registry_model_excludes_classes,
    test_registry_scaler_has_linear,
    test_registry_unknown_name_raises,
    test_label_meta_known,
    test_label_meta_fallback_to_name,
    test_label_meta_table_has_known_label,
    test_scaler_name_map_completeness,
    test_reference_defaults_logreg_binary_extracts_model_params,
    test_reference_defaults_constructor_override,
    test_loop_sfd_name_attribute,
    test_loop_sfd_params_filters_metadata_keys,
    test_loop_sfd_params_filters_arch_prefixed_keys,
    test_loop_sfd_params_includes_component_namespaced_keys,
    test_loop_sfd_params_excludes_dropped_categories,
    test_loop_sfd_params_includes_unnamespaced_model_keys,
    test_loop_sfd_manifest_split_config,
    test_loop_sfd_manifest_target_column,
    test_loop_sfd_manifest_model_function,
    test_loop_sfd_manifest_scaler_set,
    test_loop_sfd_manifest_feature_transforms_order,
    test_loop_sfd_ignores_payload_transforms,
    test_loop_sfd_manifest_label_in_target_transforms,
    test_loop_sfd_param_wiring_uses_namespaced_reference,
    test_loop_sfd_param_wiring_label_params,
    test_loop_sfd_param_wiring_literal_when_not_in_param_space,
    test_loop_sfd_unknown_scaler_raises,
    test_loop_sfd_unknown_reference_architecture_raises,
    test_progress_callback_writes_json,
    test_progress_callback_handles_none_log,
    test_progress_callback_zero_total_no_div_by_zero,
    test_quantile_flag_compiles_as_fitted_transform,
    test_forward_breakout_label_remains_plain_add_transform,
    test_quantile_flag_params_routes_through_round_params,
    test_quantile_flag_end_to_end_with_uel,
    test_loop_sfd_end_to_end_with_uel,
]


if __name__ == '__main__':
    failed = 0
    for test in _TESTS:
        try:
            test()
            sys.stdout.write(f"PASS {test.__name__}\n")
        except Exception as e:
            failed += 1
            sys.stdout.write(f"FAIL {test.__name__}: {e}\n")
            traceback.print_exc()
    if failed:
        sys.stdout.write(f"\n{failed} test(s) failed\n")
        sys.exit(1)
    sys.stdout.write(f"\nAll {len(_TESTS)} tests passed\n")

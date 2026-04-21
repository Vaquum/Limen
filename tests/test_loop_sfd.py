# ruff: noqa: E402
import json
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
from limen.data import HistoricalData
from limen.experiment.param_domain import ParamDomain
from limen.experiment.param_search import RandomStrategy
from limen.scalers import LinearScaler
from limen.sfd.loop import LoopSFD
from limen.sfd.loop.meta import (
    DATA_SOURCE_REGISTRY,
    LABEL_TARGET_COLUMNS,
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
from limen.sfd.loop.run import (
    _annotation_accepts_int,
    _coerce_string_params_by_signature,
    _fetch_data_from_payload,
)
from limen.sfd.reference_architecture import logreg_binary as _logreg_binary_func


_FIXTURE_PATH = Path(__file__).parent / 'fixtures' / 'loop_logreg_binary.json'


def _load_payload() -> dict:
    return json.loads(_FIXTURE_PATH.read_text())


def test_registry_indicator_resolution():
    from limen.indicators import bbands

    assert INDICATOR_REGISTRY['bbands'] is bbands


def test_registry_feature_resolution():
    from limen.features import atr_percent_sma, forward_breakout_target

    assert FEATURE_REGISTRY['atr_percent_sma'] is atr_percent_sma
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


def test_data_source_registry_known_methods():
    assert DATA_SOURCE_REGISTRY['get_spot_klines'] is HistoricalData.get_spot_klines


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

    # scaler_* keys (legacy selectedItems metadata) are stripped defensively
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
    payload['scaler'] = {
        'selectedItems': [{'name': 'not_a_real_scaler', 'params': {}}],
    }
    sfd = LoopSFD(payload)
    try:
        sfd.manifest()
    except ValueError as e:
        assert 'Unknown scaler' in str(e)
        return
    raise AssertionError('Expected ValueError for unknown scaler')


def test_loop_sfd_scaler_read_from_selected_items():
    payload = _load_payload()
    payload['scaler'] = {
        'selectedItems': [{'name': 'logreg', 'params': {}}],
    }
    # Drop any legacy scalingMethod field if present
    payload['scaler'].pop('scalingMethod', None)
    sfd = LoopSFD(payload)
    m = sfd.manifest()
    assert m.scaler is not None


def test_loop_sfd_no_scaler_when_selected_items_empty():
    payload = _load_payload()
    payload['scaler'] = {'selectedItems': []}
    payload['scaler'].pop('scalingMethod', None)
    sfd = LoopSFD(payload)
    m = sfd.manifest()
    assert m.scaler is None


def test_loop_sfd_data_source_from_payload_selected_items():
    payload = _load_payload()
    payload['inputData']['selectedItems'] = [{
        'name': 'data_source',
        'params': {
            'method': 'get_spot_klines',
            'kline_size': 7200,
            'start_date_limit': '2024-06-01',
        },
    }]
    sfd = LoopSFD(payload)
    assert sfd._data_source_config['method'] is HistoricalData.get_spot_klines
    assert sfd._data_source_config['params']['kline_size'] == 7200
    assert sfd._data_source_config['params']['start_date_limit'] == '2024-06-01'


def test_loop_sfd_data_source_constructor_override_wins_over_payload():
    payload = _load_payload()
    payload['inputData']['selectedItems'] = [{
        'name': 'data_source',
        'params': {'method': 'get_spot_klines', 'kline_size': 7200},
    }]
    custom = {
        'method': HistoricalData.get_spot_klines,
        'params': {'kline_size': 900},
    }
    sfd = LoopSFD(payload, data_source_config=custom)
    assert sfd._data_source_config['method'] is HistoricalData.get_spot_klines
    assert sfd._data_source_config['params']['kline_size'] == 900


def test_loop_sfd_data_source_falls_back_to_default_when_payload_empty():
    payload = _load_payload()
    payload['inputData']['selectedItems'] = []
    sfd = LoopSFD(payload)
    assert sfd._data_source_config['method'] is HistoricalData.get_spot_klines
    assert sfd._data_source_config['params']['kline_size'] == 3600


def test_loop_sfd_unknown_data_source_method_raises():
    payload = _load_payload()
    payload['inputData']['selectedItems'] = [{
        'name': 'data_source',
        'params': {'method': 'not_a_real_method'},
    }]
    try:
        LoopSFD(payload)
    except ValueError as e:
        assert 'Unknown data source method' in str(e)
        return
    raise AssertionError('Expected ValueError for unknown data source method')


def test_loop_sfd_params_excludes_input_data_source_keys():
    payload = _load_payload()
    payload['parameterSpace']['input_data_source_method'] = ['get_spot_klines']
    payload['parameterSpace']['input_data_source_kline_size'] = [3600]
    sfd = LoopSFD(payload)
    params = sfd.params()
    for key in params:
        assert not key.startswith('input_data_source_'), f"leaked: {key}"


class _HistoricalDataStub:

    '''Test double for HistoricalData that records method calls.'''

    last_method: str | None = None
    last_fetch_kwargs: dict | None = None

    def __init__(self):
        self.data = pl.DataFrame({
            'datetime': ['2025-01-01T00:00:00'],
            'open': [100.0], 'high': [101.0], 'low': [99.0], 'close': [100.5],
            'volume': [1000.0],
        })

    def get_spot_klines(self, **kwargs):
        type(self).last_method = 'get_spot_klines'
        type(self).last_fetch_kwargs = dict(kwargs)


def _reset_hd_stub():
    _HistoricalDataStub.last_method = None
    _HistoricalDataStub.last_fetch_kwargs = None


def _patch_historical_data(monkey_attr_owner, stub_cls=_HistoricalDataStub):
    import limen.sfd.loop.run as _run_mod
    original = _run_mod.HistoricalData
    _run_mod.HistoricalData = stub_cls
    return original


def _restore_historical_data(original):
    import limen.sfd.loop.run as _run_mod
    _run_mod.HistoricalData = original


def test_fetch_data_strips_none_params():
    _reset_hd_stub()
    original = _patch_historical_data(None)
    try:
        payload = {
            'inputData': {
                'selectedItems': [{
                    'name': 'data_source',
                    'params': {
                        'method': 'get_spot_klines',
                        'kline_size': 3600,
                        'n_rows': None,
                        'start_date_limit': None,
                    },
                }],
            },
        }
        _fetch_data_from_payload(payload)
        assert 'n_rows' not in _HistoricalDataStub.last_fetch_kwargs
        assert 'start_date_limit' not in _HistoricalDataStub.last_fetch_kwargs
        assert _HistoricalDataStub.last_fetch_kwargs == {'kline_size': 3600}
    finally:
        _restore_historical_data(original)


def test_fetch_data_unknown_method_raises():
    try:
        _fetch_data_from_payload({
            'inputData': {
                'selectedItems': [{
                    'name': 'data_source',
                    'params': {'method': 'not_a_real_method'},
                }],
            },
        })
    except ValueError as e:
        assert 'Unknown data source method' in str(e)
        return
    raise AssertionError('Expected ValueError for unknown method')


def test_fetch_data_missing_method_key_raises():
    try:
        _fetch_data_from_payload({
            'inputData': {
                'selectedItems': [{
                    'name': 'data_source',
                    'params': {'kline_size': 3600},
                }],
            },
        })
    except ValueError as e:
        assert 'method' in str(e)
        return
    raise AssertionError('Expected ValueError for missing method key')


def test_fetch_data_no_data_source_item_raises():
    try:
        _fetch_data_from_payload({
            'inputData': {'selectedItems': []},
        })
    except ValueError as e:
        assert 'data_source' in str(e)
        return
    raise AssertionError('Expected ValueError for missing data_source item')


def test_annotation_accepts_int_handles_unions_and_plain():
    def _plain(n: int) -> None: ...
    def _optional(n: int | None = None) -> None: ...
    def _string(s: str) -> None: ...
    def _union(n: int | str = 1) -> None: ...
    def _unannotated(n) -> None: ...

    import inspect
    assert _annotation_accepts_int(inspect.signature(_plain).parameters['n'].annotation)
    assert _annotation_accepts_int(inspect.signature(_optional).parameters['n'].annotation)
    assert not _annotation_accepts_int(inspect.signature(_string).parameters['s'].annotation)
    assert _annotation_accepts_int(inspect.signature(_union).parameters['n'].annotation)
    assert not _annotation_accepts_int(inspect.signature(_unannotated).parameters['n'].annotation)


def test_coerce_string_params_by_signature_converts_int_strings():
    # Uses the real HistoricalData.get_spot_klines signature:
    #   kline_size: int, n_rows: int | None, start_date_limit: str | None
    coerced = _coerce_string_params_by_signature(
        HistoricalData.get_spot_klines,
        {
            'kline_size': '3600',
            'n_rows': '5000',
            'start_date_limit': '2025-01-01',
        },
    )
    assert coerced['kline_size'] == 3600
    assert isinstance(coerced['kline_size'], int)
    assert coerced['n_rows'] == 5000
    assert isinstance(coerced['n_rows'], int)
    # String-annotated param is left alone
    assert coerced['start_date_limit'] == '2025-01-01'
    assert isinstance(coerced['start_date_limit'], str)


def test_coerce_string_params_leaves_non_string_values_alone():
    coerced = _coerce_string_params_by_signature(
        HistoricalData.get_spot_klines,
        {'kline_size': 3600, 'n_rows': None},
    )
    assert coerced['kline_size'] == 3600
    assert coerced['n_rows'] is None


def test_coerce_string_params_ignores_unknown_param_names():
    coerced = _coerce_string_params_by_signature(
        HistoricalData.get_spot_klines,
        {'kline_size': '3600', 'bogus_param': 'some_value'},
    )
    assert coerced['kline_size'] == 3600
    assert coerced['bogus_param'] == 'some_value'


def test_coerce_string_params_raises_clear_error_on_bad_string():
    try:
        _coerce_string_params_by_signature(
            HistoricalData.get_spot_klines,
            {'kline_size': 'not_a_number'},
        )
    except ValueError as e:
        msg = str(e)
        assert 'kline_size' in msg
        assert 'not_a_number' in msg
        assert 'get_spot_klines' in msg
        return
    raise AssertionError('Expected ValueError for non-numeric string')


class _TypedHistoricalDataStub:

    '''Stub with typed get_spot_klines so coercion logic exercises signature inspection.'''

    last_fetch_kwargs: dict | None = None

    def __init__(self):
        self.data = pl.DataFrame({
            'datetime': ['2025-01-01T00:00:00'],
            'open': [100.0], 'high': [101.0], 'low': [99.0], 'close': [100.5],
            'volume': [1000.0],
        })

    def get_spot_klines(self,
                        n_rows: int | None = None,
                        kline_size: int = 1,
                        start_date_limit: str | None = None) -> None:
        type(self).last_fetch_kwargs = {
            'n_rows': n_rows,
            'kline_size': kline_size,
            'start_date_limit': start_date_limit,
        }


def test_fetch_data_coerces_kline_size_string_through_full_path():
    # Verify the coercion is wired into _fetch_data_from_payload, not just
    # the helper in isolation.
    _TypedHistoricalDataStub.last_fetch_kwargs = None
    original = _patch_historical_data(None, stub_cls=_TypedHistoricalDataStub)
    try:
        payload = {
            'inputData': {
                'selectedItems': [{
                    'name': 'data_source',
                    'params': {
                        'method': 'get_spot_klines',
                        'kline_size': '3600',
                        'start_date_limit': '2025-01-01',
                    },
                }],
            },
        }
        _fetch_data_from_payload(payload)
        kwargs = _TypedHistoricalDataStub.last_fetch_kwargs
        assert kwargs is not None
        assert kwargs['kline_size'] == 3600
        assert isinstance(kwargs['kline_size'], int)
        assert kwargs['start_date_limit'] == '2025-01-01'
    finally:
        _restore_historical_data(original)


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
            'C': [0.01, 0.1],
            'solver': ['lbfgs'],
            'penalty': ['l2'],
            'tol': [0.001],
            'max_iter': [100],
        },
        'referenceArchitecture': 'logreg_binary',
        'inputData': {'splitRatios': {'train': 70, 'val': 15, 'test': 15}},
        'indicators': [{'name': 'roc', 'params': {'period': 1}}],
        'features': [],
        'labels': [{'name': 'quantile_flag', 'params': {'col': col, 'q': q}}],
        'scaler': {'selectedItems': [{'name': 'linear', 'params': {}}]},
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


def test_component_alias_map_built_from_payload():
    payload = {
        'referenceArchitecture': 'logreg_binary',
        'parameterSpace': {},
        'inputData': {'splitRatios': {'train': 70, 'val': 15, 'test': 15}},
        'indicators': [
            {'name': 'roc', 'params': {'price_col': 'close', 'period': 10}},
        ],
        'features': [
            {'name': 'atr_percent_sma', 'params': {'period': 14}},
        ],
        'labels': [
            {'name': 'quantile_flag', 'params': {'col': 'close', 'q': 0.5}},
        ],
    }
    sfd = LoopSFD(payload)
    expected = {
        'roc_price_col': 'indicator_roc_price_col',
        'roc_period': 'indicator_roc_period',
        'atr_percent_sma_period': 'feature_atr_percent_sma_period',
        'quantile_flag_col': 'label_quantile_flag_col',
        'quantile_flag_q': 'label_quantile_flag_q',
    }
    assert sfd._component_alias_map == expected


def test_rewrite_format_string_translates_namespaced_placeholder():
    payload = {
        'referenceArchitecture': 'logreg_binary',
        'parameterSpace': {},
        'inputData': {'splitRatios': {'train': 70, 'val': 15, 'test': 15}},
        'indicators': [{'name': 'roc', 'params': {'period': 10}}],
        'features': [],
        'labels': [],
    }
    sfd = LoopSFD(payload)
    assert sfd._rewrite_format_string('roc_{roc_period}') == 'roc_{indicator_roc_period}'


def test_rewrite_format_string_leaves_unknown_placeholder_alone():
    payload = {
        'referenceArchitecture': 'logreg_binary',
        'parameterSpace': {},
        'inputData': {'splitRatios': {'train': 70, 'val': 15, 'test': 15}},
        'indicators': [{'name': 'roc', 'params': {'period': 10}}],
        'features': [],
        'labels': [],
    }
    sfd = LoopSFD(payload)
    # 'indicator_roc_period' isn't an alias key (it's the namespaced target,
    # not the un-namespaced source), so the placeholder passes through.
    # At runtime str.format will resolve it directly against round_params.
    assert sfd._rewrite_format_string('roc_{indicator_roc_period}') == 'roc_{indicator_roc_period}'
    # Unknown placeholder: passes through unchanged; would raise KeyError
    # at runtime str.format time (same failure mode as pre-fix).
    assert sfd._rewrite_format_string('{totally_unknown}') == '{totally_unknown}'


def test_rewrite_format_string_leaves_plain_string_alone():
    payload = {
        'referenceArchitecture': 'logreg_binary',
        'parameterSpace': {},
        'inputData': {'splitRatios': {'train': 70, 'val': 15, 'test': 15}},
        'indicators': [{'name': 'roc', 'params': {'period': 10}}],
        'features': [],
        'labels': [],
    }
    sfd = LoopSFD(payload)
    assert sfd._rewrite_format_string('close') == 'close'
    assert sfd._rewrite_format_string('roc_1') == 'roc_1'


def test_wire_params_bypasses_parameter_space_for_format_strings():
    # Even when label_quantile_flag_col IS in parameterSpace, a format-string
    # valued col must be kept as a literal (post-rewrite) rather than routed
    # through round_params — otherwise _resolve_params would return the
    # round_params value terminally without applying .format().
    payload = {
        'referenceArchitecture': 'logreg_binary',
        'parameterSpace': {
            'indicator_roc_period': [10, 14, 20],
            'label_quantile_flag_col': ['roc_{roc_period}'],
            'label_quantile_flag_q': [0.95],
        },
        'inputData': {'splitRatios': {'train': 70, 'val': 15, 'test': 15}},
        'indicators': [{'name': 'roc', 'params': {'price_col': 'close', 'period': 10}}],
        'features': [],
        'labels': [
            {'name': 'quantile_flag', 'params': {'col': 'roc_{roc_period}', 'q': 0.95}},
        ],
    }
    sfd = LoopSFD(payload)
    m = sfd.manifest()
    fp, _func, bp = m.target_transforms[0]
    # The literal format string should survive as-is (rewritten), NOT
    # collapse to the parameterSpace reference 'label_quantile_flag_col'.
    assert bp['col'] == 'roc_{indicator_roc_period}'
    # q is not a format string, so it still routes through round_params as
    # a reference.
    assert fp[0][2]['col'] == 'roc_{indicator_roc_period}'
    assert fp[0][2]['q'] == 'label_quantile_flag_q'


def test_failing_payload_end_to_end_with_uel():
    # Uses the real payload shape that was originally failing:
    #   - label_quantile_flag_col IS in parameterSpace
    #   - col uses the un-namespaced {roc_period} placeholder
    # After the _wire_params fix, the format string is rewritten and
    # bypasses the parameterSpace-reference path, so .format() resolves
    # to 'roc_<period>' at runtime.
    #
    # The fixture has kline_size as a string in selectedItems (web-form
    # payload). run.py coerces this before passing data= to UEL. In the
    # test we mirror that: pre-fetch via the stub and pass data= directly.
    fixture_path = Path(__file__).parent / 'fixtures' / 'loop_template_test_failing.json'
    payload = json.loads(fixture_path.read_text())

    sfd = LoopSFD(payload)
    domain = ParamDomain(sfd.params())
    strategy = RandomStrategy(domain, seed=42)

    # Build synthetic data large enough to survive indicator NaN dropping
    # (roc period up to 20, atr_percent_sma period up to 20) and a
    # 70/10/20 split. Use a cyclical price series so the roc indicator
    # oscillates, ensuring both label classes appear in train and test
    # even at q=0.95.
    import numpy as np
    n = 500
    t = np.linspace(0, 10 * np.pi, n)
    close = 100.0 + 20.0 * np.sin(t) + np.cumsum(np.random.default_rng(42).normal(0, 0.2, n))
    rng = np.random.default_rng(0)
    synthetic_data = pl.DataFrame({
        'datetime': pl.date_range(
            pl.date(2024, 1, 1), pl.date(2025, 5, 14), interval='1d', eager=True,
        ).cast(pl.Datetime('us', 'UTC')),
        'open': close - rng.uniform(0, 0.5, n),
        'high': close + rng.uniform(0, 1, n),
        'low': close - rng.uniform(0, 1, n),
        'close': close,
        'volume': rng.uniform(1000, 5000, n),
    })

    with TemporaryDirectory() as tmpdir:
        experiment_dir = Path(tmpdir) / 'exp'
        experiment_dir.mkdir(parents=True)

        progress_file = experiment_dir / 'progress.json'
        progress_cb = make_progress_callback(progress_file, total=3)

        uel = UniversalExperimentLoop(
            sfd=sfd,
            data=synthetic_data,
            search_strategy=strategy,
            feedback_interval=1,
            checkpoint_interval=1,
            experiment_dir=experiment_dir,
            intra_callback=progress_cb,
        )

        uel.run(
            experiment_name=str(experiment_dir / 'run'),
            n_permutations=3,
        )

        assert (experiment_dir / 'results.csv').exists()
        assert (experiment_dir / 'metadata.json').exists()
        assert progress_file.exists()
        data = json.loads(progress_file.read_text())
        assert data['total'] == 3


def test_quantile_flag_end_to_end_with_uel():
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
    test_data_source_registry_known_methods,
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
    test_loop_sfd_scaler_read_from_selected_items,
    test_loop_sfd_no_scaler_when_selected_items_empty,
    test_loop_sfd_data_source_from_payload_selected_items,
    test_loop_sfd_data_source_constructor_override_wins_over_payload,
    test_loop_sfd_data_source_falls_back_to_default_when_payload_empty,
    test_loop_sfd_unknown_data_source_method_raises,
    test_loop_sfd_params_excludes_input_data_source_keys,
    test_fetch_data_strips_none_params,
    test_fetch_data_unknown_method_raises,
    test_fetch_data_missing_method_key_raises,
    test_fetch_data_no_data_source_item_raises,
    test_annotation_accepts_int_handles_unions_and_plain,
    test_coerce_string_params_by_signature_converts_int_strings,
    test_coerce_string_params_leaves_non_string_values_alone,
    test_coerce_string_params_ignores_unknown_param_names,
    test_coerce_string_params_raises_clear_error_on_bad_string,
    test_fetch_data_coerces_kline_size_string_through_full_path,
    test_loop_sfd_unknown_reference_architecture_raises,
    test_progress_callback_writes_json,
    test_progress_callback_handles_none_log,
    test_progress_callback_zero_total_no_div_by_zero,
    test_quantile_flag_compiles_as_fitted_transform,
    test_forward_breakout_label_remains_plain_add_transform,
    test_quantile_flag_params_routes_through_round_params,
    test_component_alias_map_built_from_payload,
    test_rewrite_format_string_translates_namespaced_placeholder,
    test_rewrite_format_string_leaves_unknown_placeholder_alone,
    test_rewrite_format_string_leaves_plain_string_alone,
    test_wire_params_bypasses_parameter_space_for_format_strings,
    test_failing_payload_end_to_end_with_uel,
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

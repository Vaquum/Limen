from contextlib import nullcontext
from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from limen.features.price_lines import price_lines
from limen.features.quantile_price_lines import quantile_price_lines
from limen.targets.tradeline_long_binary import TradelineLongBinaryTarget
from limen.utils._computation_cache import computation_reuse, current_cache
from limen.utils.find_price_lines import find_price_lines, price_line_arrays

_FIXTURE = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'


@pytest.fixture
def bars():
    return pl.read_parquet(_FIXTURE, n_rows=512)


def _detector_calls(monkeypatch):
    module = import_module('limen.utils.find_price_lines')
    original = module._compute_price_line_arrays
    calls = []

    def counted(*args):
        calls.append((len(args[0]), args[1], args[2]))
        return original(*args)

    monkeypatch.setattr(module, '_compute_price_line_arrays', counted)
    return calls


def _features(frame, quantile=0.75, research=False):
    return quantile_price_lines(
        price_lines(frame, 24, .001, include_research_only=research),
        24, .001, quantile, include_research_only=research,
    )


def test_features_and_targets_share_only_identical_detector_inputs(bars, monkeypatch):
    expected = _features(bars)
    target = TradelineLongBinaryTarget(bars, 'entry', 24, .001, 75)
    calls = _detector_calls(monkeypatch)
    with computation_reuse() as cache:
        actual = _features(bars.lazy()).collect()
        first = TradelineLongBinaryTarget(bars, 'entry', 24, .001, 75)
        second = TradelineLongBinaryTarget(bars, 'entry', 24, .001, 65)
        assert len(calls) == 1
        assert first is not second
        assert first.threshold == target.threshold
        assert cache.primitives.bytes > 0
        assert cache.features.bytes > 0
    assert current_cache() is None
    assert cache.primitives.bytes == cache.features.bytes == 0
    assert_frame_equal(actual, expected, check_exact=True)


def test_reuse_preserves_source_columns_schema_and_changed_parameters(bars, monkeypatch):
    expected = _features(bars)
    wider = bars.with_columns(pl.col('volume').alias('recorded_volume'))
    calls = _detector_calls(monkeypatch)
    with computation_reuse():
        first = _features(bars)
        repeated = _features(wider.lazy()).collect()
        changed = _features(bars, quantile=.8, research=True)
        assert len(calls) == 1
    assert_frame_equal(first, expected, check_exact=True)
    assert_frame_equal(repeated.select(expected.columns), expected, check_exact=True)
    assert repeated['recorded_volume'].equals(bars['volume'])
    assert_frame_equal(changed, _features(bars, quantile=.8, research=True), check_exact=True)


def test_changed_source_order_warmup_and_mutation_cannot_hit_stale_arrays(bars, monkeypatch):
    close = bars['close'].to_numpy().copy()
    expected = price_line_arrays(close, 24, .001)
    calls = _detector_calls(monkeypatch)
    with computation_reuse():
        arrays = price_line_arrays(close, 24, .001)
        arrays[0][2][:] = arrays[0][2][::-1]
        again = price_line_arrays(close, 24, .001)
        for actual, reference in zip(again[0], expected[0], strict=True):
            np.testing.assert_array_equal(actual, reference)
        close[0], close[1] = close[1], close[0]
        price_line_arrays(close, 24, .001)
        price_line_arrays(close[::-1], 24, .001)
        price_line_arrays(close[24:], 24, .001)
        price_line_arrays(close, 48, .001)
        price_line_arrays(close, 24, .002)
        assert len(calls) == 6


def test_cached_feature_output_isolated_from_mutation(bars):
    expected = _features(bars)
    with computation_reuse():
        first = _features(bars)
        first.replace_column(first.get_column_index('trending_score'), first['trending_score'].reverse())
        assert_frame_equal(_features(bars), expected, check_exact=True)


def test_budget_evicts_entries_and_oversized_computations_remain_fresh(bars, monkeypatch):
    close = bars['close'].head(32).to_numpy()
    with computation_reuse() as cache:
        price_line_arrays(close, 24, 10.0)
        weight = cache.primitives.bytes
    calls = _detector_calls(monkeypatch)
    with computation_reuse(max_bytes=weight * 2) as cache:
        for threshold in (10.0, 20.0, 10.0):
            price_line_arrays(close, 24, threshold)
            assert cache.primitives.bytes <= cache.primitives.max_bytes
            assert len(cache.primitives.entries) == 1
        assert len(calls) == 3
    with computation_reuse(max_bytes=0) as cache:
        _features(bars)
        _features(bars)
        assert cache.primitives.bytes == cache.features.bytes == 0
        assert not cache.primitives.entries and not cache.features.entries


def test_errors_and_typed_parameters_survive_hits(bars):
    close = bars['close'].to_numpy()
    with computation_reuse():
        price_line_arrays(close, 24, .001)
        with pytest.raises(TypeError):
            price_line_arrays(close, 24.0, .001)
        with pytest.raises(ValueError, match='max_duration_hours'):
            price_line_arrays(close, 1, .001)
        with pytest.raises(ValueError, match='min_height_pct'):
            price_line_arrays(close, 24, 0.0)
        with pytest.raises(ValueError, match='big_move_lookback_hours'):
            price_lines(bars, 24, .001, big_move_lookback_hours=0)
        with pytest.raises(ValueError, match='include_research_only'):
            price_lines(bars, 24, .001, include_research_only=1)
    with pytest.raises(ValueError, match='nonnegative integer'):
        with computation_reuse(max_bytes=-1):
            pytest.fail('Negative cache budget was accepted')


def test_expired_lazy_context_and_nested_invocations(bars, monkeypatch):
    calls = _detector_calls(monkeypatch)
    with computation_reuse() as outer:
        expected = _features(bars)
        with computation_reuse() as inner:
            delayed = _features(bars.lazy())
            _features(bars)
            assert len(calls) == 2
        assert current_cache() is outer
        assert not inner.primitives.entries
        assert_frame_equal(delayed.collect(), expected, check_exact=True)
        assert not inner.primitives.entries and not inner.features.entries
        assert len(calls) == 4
    assert current_cache() is None


def test_dictionary_output_contract_matches_shared_arrays(bars):
    close = bars['close'].to_numpy()
    expected = find_price_lines(close, 24, .001)
    with computation_reuse():
        assert find_price_lines(close, 24, .001) == expected
        assert find_price_lines(close, 24, .001) == expected


def test_random_sweep_reuses_transforms_with_fresh_fitted_state(tmp_path, bars, monkeypatch):
    from limen.experiment import UniversalExperimentLoop
    from limen.experiment.manifest_core import MLManifest
    from limen.yaml import CompiledSFD, build_search_strategy
    from tests.test_record_model_outputs import _config

    config = _config()
    manifest = config['sfd']['manifest']
    manifest['split_dates'] = {
        'train_start': '2024-01-01', 'train_end': '2024-01-12',
        'val_start': '2024-01-12', 'val_end': '2024-01-16',
        'test_start': '2024-01-16', 'test_end': '2024-01-21',
        'test_predict_guard': False,
    }
    manifest['features'] = [
        {'func': 'limen.features.price_lines', 'params': {'max_duration_hours': 24,
         'min_height_pct': .001, 'include_research_only': False}},
        {'func': 'limen.features.quantile_price_lines', 'params': {'max_duration_hours': 24,
         'min_height_pct': .001, 'include_research_only': False}},
    ]
    manifest['scaler'] = {'class': 'limen.scalers.RobustScaler'}
    config['uel']['search_strategy'] = {'type': 'random', 'seed': 42}
    expected_rows = None
    for enabled in (False, True):
        module = import_module('limen.experiment.experiment_core')
        fitted = []
        original = MLManifest.prepare_data

        def capture(*args, **kwargs):
            data = original(*args, **kwargs)
            fitted.append(data['_fitted_params'])
            return data

        with monkeypatch.context() as hooks:
            calls = _detector_calls(hooks)
            hooks.setattr(MLManifest, 'prepare_data', capture)
            if not enabled:
                hooks.setattr(module, 'computation_reuse', nullcontext)
            loop = UniversalExperimentLoop(
                sfd=CompiledSFD(config), data=bars, experiment_dir=tmp_path / str(enabled),
                yaml_reference=config, search_strategy=build_search_strategy(config),
            )
            loop.run('reuse', n_permutations=2, prep_each_round=True, progress_bar=False)
        assert len(calls) == (3 if enabled else 12)
        assert len(fitted) == 2
        assert fitted[0]['_scaler'] is not fitted[1]['_scaler']
        rows = loop.experiment_log.drop('execution_time')
        if expected_rows is None:
            expected_rows = rows
        else:
            assert_frame_equal(rows, expected_rows, check_exact=True)


def test_preparation_recomputes_changed_carried_context_without_sharing_fits(bars):
    from limen.experiment import MLManifest
    from limen.indicators.window_return import window_return
    from limen.scalers.causal_rolling_robust_scaler import CausalRollingRobustScaler
    from limen.targets.quantile_binary import QuantileBinaryTarget

    manifest = (MLManifest().set_split_config(3, 1, 1)
        .add_indicator(window_return, period=1)
        .add_feature(price_lines, max_duration_hours=24, min_height_pct=.001, include_research_only=False)
        .with_target_label('entry', QuantileBinaryTarget,
                           fit_params={'source_column': 'ret_1', 'quantile': .5},
                           transform_params={'shift': -1})
        .set_scaler(CausalRollingRobustScaler, extra_params={'window': 'context', 'min_samples': 3}))
    expected = [manifest.prepare_data(bars, {'context': context}) for context in (6, 12)]
    with computation_reuse():
        actual = [manifest.prepare_data(bars, {'context': context}) for context in (6, 12)]
    for left, right in zip(actual, expected, strict=True):
        for name, value in right.items():
            if isinstance(value, pl.DataFrame):
                assert_frame_equal(left[name], value, check_exact=True)
        assert left['_fitted_params']['_scaler'] is not right['_fitted_params']['_scaler']
    assert actual[0]['_fitted_params']['_scaler'] is not actual[1]['_fitted_params']['_scaler']

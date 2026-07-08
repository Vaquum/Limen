import math
import numpy as np
import polars as pl
import pytest

from limen.indicators._bbands import stddev_from_var, stddev_using_precalc_ma
from limen.data.utils.random_slice import random_slice
from limen.data.utils.splits import split_data_to_prep_output, split_random, split_sequential
from limen.indicators._ema import ema_talib_default_segment, ema_talib_segment_with_k
from limen.indicators.ht_phasor import _ht_phasor_from_values, ht_phasor
from limen.indicators.ht_sine import _ht_sine_from_values, ht_sine
from limen.indicators.price_change_pct import price_change_pct
from limen.indicators.returns import returns
from limen.indicators.rsi_sma import rsi_sma
from limen.utils.data_dict_to_numpy import data_dict_to_numpy


def test_ema_segment_returns_empty_when_requested_range_is_before_lookback() -> None:
    out_beg_idx, values = ema_talib_segment_with_k(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        period=3,
        k=0.5,
        start_idx=0,
        end_idx=1,
    )

    assert out_beg_idx == 2
    assert values.size == 0


def test_ema_segment_matches_manual_recursive_values() -> None:
    out_beg_idx, values = ema_talib_segment_with_k(
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        period=3,
        k=0.5,
        start_idx=2,
        end_idx=4,
    )

    assert out_beg_idx == 2
    assert values.tolist() == pytest.approx([2.0, 3.0, 4.0])


def test_default_ema_segment_uses_standard_talib_constant() -> None:
    manual = ema_talib_segment_with_k(
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        period=3,
        k=2.0 / 4.0,
        start_idx=2,
        end_idx=4,
    )
    default = ema_talib_default_segment(
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        period=3,
        start_idx=2,
        end_idx=4,
    )

    assert default[0] == manual[0]
    assert default[1].tolist() == pytest.approx(manual[1].tolist())


def test_stddev_from_var_matches_manual_rolling_std_and_returns_empty_before_lookback() -> None:
    start_idx, empty = stddev_from_var(
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        start_idx=0,
        end_idx=1,
        period=3,
    )

    assert start_idx == 2
    assert empty.size == 0

    start_idx, values = stddev_from_var(
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        start_idx=0,
        end_idx=4,
        period=3,
    )

    assert start_idx == 2
    assert values.tolist() == pytest.approx([math.sqrt(2.0 / 3.0)] * 3)


def test_stddev_using_precalc_ma_matches_manual_window_std() -> None:
    values = stddev_using_precalc_ma(
        np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.asarray([np.nan, np.nan, 2.0, 3.0, 4.0]),
        movavg_beg_idx=2,
        movavg_nb_element=3,
        period=3,
    )

    assert values.tolist() == pytest.approx([math.sqrt(2.0 / 3.0)] * 3)


def test_ht_phasor_short_inputs_return_nan_only_and_wrapper_uses_custom_price_col() -> None:
    inphase, quadrature = _ht_phasor_from_values(np.arange(10, dtype=float))

    assert np.isnan(inphase).all()
    assert np.isnan(quadrature).all()

    long_prices = np.linspace(1.0, 100.0, 120)
    result = ht_phasor(pl.DataFrame({'price': long_prices}), price_col='price')
    inphase = result['ht_phasor_inphase'].to_numpy()
    quadrature = result['ht_phasor_quadrature'].to_numpy()

    assert result.columns == ['price', 'ht_phasor_inphase', 'ht_phasor_quadrature']
    assert np.isnan(inphase[:32]).all()
    assert np.isnan(quadrature[:32]).all()
    assert np.isfinite(inphase[32:]).all()
    assert np.isfinite(quadrature[32:]).all()


def test_ht_phasor_constant_signal_stays_zero_after_lookback_and_matches_wrapper() -> None:
    prices = np.full(120, 100.0)

    direct_inphase, direct_quadrature = _ht_phasor_from_values(prices)
    wrapped = ht_phasor(pl.DataFrame({'price': prices}), price_col='price')

    wrapped_inphase = wrapped['ht_phasor_inphase'].to_numpy()
    wrapped_quadrature = wrapped['ht_phasor_quadrature'].to_numpy()

    assert np.isnan(direct_inphase[:32]).all()
    assert np.isnan(direct_quadrature[:32]).all()
    assert np.allclose(direct_inphase[32:], 0.0)
    assert np.allclose(direct_quadrature[32:], 0.0)
    assert np.allclose(wrapped_inphase[32:], direct_inphase[32:])
    assert np.allclose(wrapped_quadrature[32:], direct_quadrature[32:])


def test_ht_phasor_oscillating_signal_produces_distinct_components() -> None:
    phase = np.linspace(0.0, 8.0 * math.pi, 256)
    prices = 100.0 + (5.0 * np.sin(phase)) + (0.5 * np.cos(phase * 2.0))

    inphase, quadrature = _ht_phasor_from_values(prices)
    finite_mask = np.isfinite(inphase) & np.isfinite(quadrature)

    assert finite_mask.sum() == 224
    assert np.std(inphase[finite_mask]) > 0.1
    assert np.std(quadrature[finite_mask]) > 0.1
    assert not np.allclose(inphase[finite_mask], quadrature[finite_mask])


def test_ht_sine_short_inputs_return_nan_only_and_wrapper_outputs_bounded_values() -> None:
    sine, lead_sine = _ht_sine_from_values(np.arange(10, dtype=float))

    assert np.isnan(sine).all()
    assert np.isnan(lead_sine).all()

    long_prices = np.linspace(1.0, 100.0, 120)
    result = ht_sine(pl.DataFrame({'price': long_prices}), price_col='price')
    sine = result['ht_sine'].to_numpy()
    lead_sine = result['ht_sine_lead'].to_numpy()

    assert result.columns == ['price', 'ht_sine', 'ht_sine_lead']
    assert np.isnan(sine[:63]).all()
    assert np.isnan(lead_sine[:63]).all()
    assert np.isfinite(sine[63:]).all()
    assert np.isfinite(lead_sine[63:]).all()
    assert np.nanmax(np.abs(sine[63:])) <= 1.0
    assert np.nanmax(np.abs(lead_sine[63:])) <= 1.0


def test_ht_sine_constant_signal_converges_to_stable_bounded_outputs() -> None:
    prices = np.full(200, 100.0)

    sine, lead_sine = _ht_sine_from_values(prices)

    assert np.isnan(sine[:63]).all()
    assert np.isnan(lead_sine[:63]).all()
    assert np.isfinite(sine[63:]).all()
    assert np.isfinite(lead_sine[63:]).all()
    assert np.nanmax(np.abs(sine[63:])) <= 1.0
    assert np.nanmax(np.abs(lead_sine[63:])) <= 1.0
    assert np.ptp(sine[-10:]) == pytest.approx(0.0)
    assert np.ptp(lead_sine[-10:]) == pytest.approx(0.0)


def test_ht_sine_oscillating_signal_matches_wrapper_and_has_phase_advanced_lead() -> None:
    phase = np.linspace(0.0, 8.0 * math.pi, 256)
    prices = 100.0 + (5.0 * np.sin(phase)) + (0.5 * np.cos(phase * 2.0))

    direct_sine, direct_lead = _ht_sine_from_values(prices)
    wrapped = ht_sine(pl.DataFrame({'price': prices}), price_col='price')

    wrapped_sine = wrapped['ht_sine'].to_numpy()
    wrapped_lead = wrapped['ht_sine_lead'].to_numpy()
    finite_mask = np.isfinite(direct_sine) & np.isfinite(direct_lead)

    assert finite_mask.sum() == 193
    assert np.allclose(wrapped_sine[finite_mask], direct_sine[finite_mask])
    assert np.allclose(wrapped_lead[finite_mask], direct_lead[finite_mask])
    assert np.nanmax(np.abs(direct_sine[finite_mask])) <= 1.0
    assert np.nanmax(np.abs(direct_lead[finite_mask])) <= 1.0
    assert not np.allclose(direct_sine[finite_mask], direct_lead[finite_mask])


def test_price_change_pct_computes_percentage_change_from_prior_close() -> None:
    data = pl.DataFrame({'close': [100.0, 110.0, 121.0, 108.9]})
    result = price_change_pct(data, period=1)

    assert result['price_change_pct_1'].to_list()[0] is None
    assert result['price_change_pct_1'].to_list()[1:] == pytest.approx([10.0, 10.0, -10.0])


def test_returns_matches_polars_pct_change_convention() -> None:
    data = pl.DataFrame({'close': [100.0, 110.0, 121.0, 108.9]})
    result = returns(data)

    assert result['returns'].to_list()[0] is None
    assert result['returns'].to_list()[1:] == pytest.approx([0.1, 0.1, -0.1])


def test_rsi_sma_is_neutral_when_recent_gains_and_losses_balance() -> None:
    data = pl.DataFrame({'close': [1.0, 2.0, 1.0, 2.0, 1.0]})
    result = rsi_sma(data, period=2)

    assert result['rsi_sma_2'].to_list()[2:] == pytest.approx([50.0, 50.0, 50.0])


def test_random_slice_rejects_invalid_safe_range_bounds() -> None:
    data = pl.DataFrame({'value': list(range(10))})

    with pytest.raises(ValueError, match=r'safe_range_low must be >= 0\.0'):
        random_slice(data, rows=2, safe_range_low=0.8, safe_range_high=0.2)


def test_random_slice_rejects_slice_that_cannot_fit_safe_range() -> None:
    data = pl.DataFrame({'value': list(range(10))})

    with pytest.raises(ValueError, match='slice size'):
        random_slice(data, rows=6, safe_range_low=0.25, safe_range_high=0.75, seed=1)


def test_random_slice_returns_contiguous_window_inside_safe_range() -> None:
    data = pl.DataFrame({'value': list(range(10))})
    result = random_slice(data, rows=3, safe_range_low=0.2, safe_range_high=0.6, seed=0)
    values = result['value'].to_list()

    assert values == list(range(values[0], values[0] + 3))
    assert 2 <= values[0] <= 3


def test_split_sequential_preserves_order_and_uses_all_rows() -> None:
    data = pl.DataFrame({'value': list(range(10))})
    splits = split_sequential(data, ratios=[6, 2, 2])

    assert [split.height for split in splits] == [6, 2, 2]
    assert splits[0]['value'].to_list() == [0, 1, 2, 3, 4, 5]
    assert splits[1]['value'].to_list() == [6, 7]
    assert splits[2]['value'].to_list() == [8, 9]


def test_split_random_creates_reproducible_partition_with_full_coverage() -> None:
    data = pl.DataFrame({'value': list(range(10))})
    first = split_random(data, ratios=[6, 2, 2], seed=42)
    second = split_random(data, ratios=[6, 2, 2], seed=42)

    assert [split.height for split in first] == [6, 2, 2]
    assert [split['value'].to_list() for split in first] == [split['value'].to_list() for split in second]
    combined = first[0]['value'].to_list() + first[1]['value'].to_list() + first[2]['value'].to_list()
    assert sorted(combined) == list(range(10))


def test_split_data_to_prep_output_builds_expected_alignment_metadata() -> None:
    split_data = [
        pl.DataFrame({'datetime': [1, 2], 'feat': [10, 20], 'target': [0, 1]}),
        pl.DataFrame({'datetime': [3], 'feat': [30], 'target': [0]}),
        pl.DataFrame({'datetime': [5, 6], 'feat': [50, 60], 'target': [1, 0]}),
    ]
    cols = ['datetime', 'feat', 'target']
    original_frames = [frame.clone() for frame in split_data]
    original_ids = [id(frame) for frame in split_data]
    original_columns = [frame.columns for frame in split_data]
    original_cols = list(cols)

    result = split_data_to_prep_output(split_data, cols, all_datetimes=[1, 2, 3, 4, 5, 6])

    assert result['x_train'].columns == ['feat']
    assert result['x_train']['feat'].to_list() == [10, 20]
    assert result['y_test'].to_list() == [1, 0]
    assert result['_alignment']['missing_datetimes'] == [4]
    assert result['_alignment']['first_test_datetime'] == 5
    assert result['_alignment']['last_test_datetime'] == 6
    assert [id(frame) for frame in split_data] == original_ids
    assert [frame.columns for frame in split_data] == original_columns
    assert all(frame.equals(original) for frame, original in zip(split_data, original_frames, strict=True))
    assert cols == original_cols


def test_split_data_to_prep_output_requires_datetime_column() -> None:
    split_data = [
        pl.DataFrame({'datetime': [1], 'feat': [10], 'target': [0]}),
        pl.DataFrame({'datetime': [2], 'feat': [20], 'target': [1]}),
        pl.DataFrame({'datetime': [3], 'feat': [30], 'target': [0]}),
    ]

    with pytest.raises(ValueError, match='SFDs must contain `datetime`'):
        split_data_to_prep_output(split_data, ['feat', 'target'], all_datetimes=[1, 2, 3])


def test_data_dict_to_numpy_converts_requested_polars_entries() -> None:
    data = {
        'x_train': pl.DataFrame({'feature': [1, 2]}),
        'y_train': pl.Series('target', [0, 1]),
        'ignored': pl.DataFrame({'feature': [9]}),
    }

    result = data_dict_to_numpy(data)

    assert set(result) == {'x_train', 'y_train'}
    assert result['x_train'].tolist() == [[1], [2]]
    assert result['y_train'].tolist() == [0, 1]


def test_data_dict_to_numpy_respects_custom_keys_and_preserves_numpy_values() -> None:
    raw_array = np.asarray([9, 8, 7])
    data = {
        'x_extra': raw_array,
        'x_train': pl.DataFrame({'feature': [1, 2]}),
    }

    result = data_dict_to_numpy(data, keys=['x_extra'])

    assert set(result) == {'x_extra'}
    assert result['x_extra'] is raw_array

from datetime import datetime
from datetime import timedelta

import polars as pl
import pytest

from limen.data.utils.compute_data_bars import compute_data_bars
from limen.data.bars import liquidity_bars
from limen.data.bars import trade_bars
from limen.data.bars import volume_bars

from tests.utils.historical_data import get_cached_spot_klines_2h


def validate_bars_output(
        result: pl.DataFrame, expected_aggregation: bool = True):
    assert isinstance(result, pl.DataFrame)
    assert len(result) > 0

    expected_columns = {
        'datetime', 'open', 'high', 'low', 'close', 'volume',
        'no_of_trades', 'liquidity_sum', 'maker_ratio', 'maker_volume',
        'maker_liquidity', 'mean', 'bar_count', 'base_interval'
    }
    assert set(result.columns) == expected_columns

    if len(result) > 1:
        assert (result['datetime']
                .diff()
                .drop_nulls()
                .dt.total_nanoseconds() >= 0).all()

    assert (result['high'] >= result['open']).all()
    assert (result['high'] >= result['low']).all()
    assert (result['high'] >= result['close']).all()
    assert (result['low'] <= result['open']).all()
    assert (result['low'] <= result['high']).all()
    assert (result['low'] <= result['close']).all()

    assert (result['volume'] > 0).all()
    assert (result['no_of_trades'] > 0).all()
    assert (result['liquidity_sum'] > 0).all()
    assert (result['maker_ratio'] >= 0).all()
    assert (result['maker_ratio'] <= 1).all()
    assert (result['maker_volume'] >= 0).all()
    assert (result['maker_liquidity'] >= 0).all()
    assert (result['mean'] >= 0).all()
    assert (result['bar_count'] > 0).all()

    if expected_aggregation:
        total_klines = result['bar_count'].sum()
        avg_klines_per_bar = total_klines / len(result)
        assert avg_klines_per_bar > 1.0, f"No aggregation detected: {avg_klines_per_bar:.1f} klines per bar"


def _make_small_kline_frame() -> pl.DataFrame:
    start = datetime(2025, 1, 1)
    rows = []

    for idx in range(4):
        rows.append({
            'datetime': start + timedelta(hours=idx),
            'open': 10.0 + idx,
            'high': 11.0 + idx,
            'low': 9.0 + idx,
            'close': 10.5 + idx,
            'volume': 100.0 * (idx + 1),
            'no_of_trades': 10 * (idx + 1),
            'liquidity_sum': 1000.0 * (idx + 1),
            'maker_ratio': 0.5,
            'maker_volume': 50.0 * (idx + 1),
            'maker_liquidity': 500.0 * (idx + 1),
            'mean': 10.25 + idx,
        })

    return pl.DataFrame(rows)


def test_volume_bars_basic():
    data = get_cached_spot_klines_2h(5000)
    result = volume_bars(data, volume_threshold=2060000.0)

    validate_bars_output(result, expected_aggregation=True)
    EXPECTED_BASE_INTERVAL = 7200
    MIN_EXPECTED_BARS = 10
    MAX_EXPECTED_BARS = 20
    assert result['base_interval'][0] == EXPECTED_BASE_INTERVAL
    assert MIN_EXPECTED_BARS <= len(result) <= MAX_EXPECTED_BARS, f"Expected 10-20 bars, got {len(result)}"


def test_trade_bars_basic():
    data = get_cached_spot_klines_2h(5000)
    result = trade_bars(data, trade_threshold=29000000)

    validate_bars_output(result, expected_aggregation=True)
    EXPECTED_BASE_INTERVAL = 7200
    MIN_EXPECTED_BARS = 10
    MAX_EXPECTED_BARS = 20
    assert result['base_interval'][0] == EXPECTED_BASE_INTERVAL
    assert MIN_EXPECTED_BARS <= len(result) <= MAX_EXPECTED_BARS, f"Expected 10-20 bars, got {len(result)}"


def test_liquidity_bars_basic():
    data = get_cached_spot_klines_2h(5000)
    result = liquidity_bars(data, liquidity_threshold=32000000000.0)

    validate_bars_output(result, expected_aggregation=True)
    EXPECTED_BASE_INTERVAL = 7200
    MIN_EXPECTED_BARS = 10
    MAX_EXPECTED_BARS = 20
    assert result['base_interval'][0] == EXPECTED_BASE_INTERVAL
    assert MIN_EXPECTED_BARS <= len(result) <= MAX_EXPECTED_BARS, f"Expected 10-20 bars, got {len(result)}"


def test_compute_data_bars_base_and_unknown_modes_return_input_unchanged():
    data = _make_small_kline_frame()

    assert compute_data_bars(data, bar_type='base') is data
    assert compute_data_bars(data, bar_type='unsupported') is data


def test_compute_data_bars_dispatches_to_specific_bar_builders():
    data = _make_small_kline_frame()

    assert compute_data_bars(
        data,
        bar_type='trade',
        trade_threshold=25,
    ).equals(trade_bars(data, 25))
    assert compute_data_bars(
        data,
        bar_type='volume',
        volume_threshold=250.0,
    ).equals(volume_bars(data, 250.0))
    assert compute_data_bars(
        data,
        bar_type='liquidity',
        liquidity_threshold=2500.0,
    ).equals(liquidity_bars(data, 2500.0))


def test_compute_data_bars_requires_threshold_for_selected_bar_type():
    data = _make_small_kline_frame()

    with pytest.raises(ValueError, match='trade_threshold'):
        compute_data_bars(data, bar_type='trade')

    with pytest.raises(ValueError, match='volume_threshold'):
        compute_data_bars(data, bar_type='volume')

    with pytest.raises(ValueError, match='liquidity_threshold'):
        compute_data_bars(data, bar_type='liquidity')

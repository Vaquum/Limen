import polars as pl
from loop.data.bars import volume_bars, trade_bars, liquidity_bars
from loop.tests.utils.get_data import get_klines_data_small


def validate_bars_output(
        result: pl.DataFrame, expected_aggregation: bool = True):
    assert isinstance(result, pl.DataFrame)
    assert len(result) > 0
    
    expected_columns = {
        'datetime', 'open', 'high', 'low', 'close', 'volume', 
        'no_of_trades', 'liquidity_sum', 'maker_ratio', 'maker_volume', 
        'maker_liquidity', 'bar_count', 'base_interval'
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
    assert (result['bar_count'] > 0).all()
    
    if expected_aggregation:
        total_klines = result['bar_count'].sum()
        avg_klines_per_bar = total_klines / len(result)
        assert avg_klines_per_bar > 1.0, f"No aggregation detected: {avg_klines_per_bar:.1f} klines per bar"


def test_volume_bars_basic():
    data = get_klines_data_small()
    result = volume_bars(data, volume_threshold=2060000.0)
    
    validate_bars_output(result, expected_aggregation=True)
    assert result['base_interval'][0] == 7200
    assert 10 <= len(result) <= 20, f'Expected 10-20 bars, got {len(result)}'


def test_trade_bars_basic():
    data = get_klines_data_small()
    result = trade_bars(data, trade_threshold=29000000)
    
    validate_bars_output(result, expected_aggregation=True)
    assert result['base_interval'][0] == 7200
    assert 10 <= len(result) <= 20, f'Expected 10-20 bars, got {len(result)}'


def test_liquidity_bars_basic():
    data = get_klines_data_small()
    result = liquidity_bars(data, liquidity_threshold=32000000000.0)
    
    validate_bars_output(result, expected_aggregation=True)
    assert result['base_interval'][0] == 7200
    assert 10 <= len(result) <= 20, f'Expected 10-20 bars, got {len(result)}'



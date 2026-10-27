from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal
import pytest

from limen.data import HistoricalData
from limen.data.utils import lookback_window, sample_regime_blocks
from limen.features.classify_market_regime import RegimeConfig, classify_market_regime

SOURCE = Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet'


@pytest.fixture(scope='module')
def market():
    return pl.read_parquet(SOURCE, n_rows=2500)


def test_lookback_preserves_recorded_rows_and_explicit_source_end(market):
    original = market.clone()
    expected = market.filter(pl.col('datetime') >= market['datetime'][-1] - timedelta(days=2))
    assert_frame_equal(lookback_window(market, 2), expected)
    with patch('limen.data.historical_data._read_any_file', return_value=market):
        source = HistoricalData().get_spot_klines(kline_size=900, lookback_days=2,
            end_date_limit=market['datetime'][-1].strftime('%Y-%m-%dT%H:%M:%S'))
    assert_frame_equal(source, expected)
    assert_frame_equal(market, original)


@pytest.mark.parametrize('value', [True, 0, -1, float('nan'), float('inf')])
def test_invalid_lookback_refuses_before_acquisition(market, value):
    with pytest.raises(ValueError, match='lookback_days'):
        lookback_window(market, value)
    with patch('limen.data.historical_data._read_any_file') as transport:
        with pytest.raises(ValueError, match='lookback_days'):
            HistoricalData().get_spot_klines(lookback_days=value, end_date_limit='2025-01-03')
        transport.assert_not_called()


def test_lookback_requires_explicit_nonconflicting_source_end():
    for options in ({}, {'end_date_limit': '2025-01-03', 'start_date_limit': '2025-01-01'}):
        with patch('limen.data.historical_data._read_any_file') as transport:
            with pytest.raises(ValueError, match='explicit end_date_limit'):
                HistoricalData().get_spot_klines(lookback_days=2, **options)
            transport.assert_not_called()


def test_regime_sampling_retains_one_original_contiguous_block(market):
    states = classify_market_regime(market.with_columns(pl.lit('input').alias('series_id')),
                                   config=RegimeConfig(version='label3-test'))['market_regime'].to_list()
    blocks = []
    start = 0
    for end in range(1, len(states) + 1):
        if end == len(states) or states[end] != states[start]:
            if states[start] != 'INITIALIZING':
                blocks.append((end - start, states[start]))
            start = end
    length, regime = max(blocks)
    assert length >= 10
    chosen = sample_regime_blocks(market, regime, 10, seed=42, regime_version='label3-test')
    assert_frame_equal(chosen, sample_regime_blocks(market, regime, 10, seed=42, regime_version='label3-test'))
    positions = [index for index, clock in enumerate(market['datetime']) if clock in chosen['datetime']]
    assert positions == list(range(positions[0], positions[-1] + 1))
    assert_frame_equal(chosen, market.slice(positions[0], len(positions)))
    assert set(states[positions[0]:positions[-1] + 1]) == {regime}
    with pytest.raises(ValueError, match='No contiguous'):
        sample_regime_blocks(market, regime, market.height + 1, seed=42, regime_version='label3-test')


@pytest.mark.parametrize('options', [
    {'target_regime': 'INITIALIZING'}, {'target_regime': 'unknown'},
    {'min_block_size': True}, {'min_block_size': 0}, {'seed': True}, {'seed': -1},
])
def test_invalid_regime_sampling_refuses(market, options):
    settings = {'target_regime': 'TREND_UP', 'min_block_size': 10, 'seed': 42, 'regime_version': 'label3-test'}
    settings.update(options)
    with pytest.raises(ValueError):
        sample_regime_blocks(market, **settings)

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from limen.data import HistoricalData
from limen.features.conserved_flux_renormalization import conserved_flux_renormalization

FIXTURE_PATH = Path(__file__).parent / 'fixtures' / 'binance_trades_btcusdt_2025-05-23_90s.zip'
TRADE_COLUMNS = ['trade_id', 'price', 'quantity', 'quote_quantity', 'timestamp', 'is_buyer_maker', '_null']
KLINE_COLUMNS = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'value_sum', 'vwap']
FEATURE_COLUMNS = [
    'flux_rel_std_mean',
    'flux_rel_std_var',
    'entropy_mean',
    'entropy_var',
    'Δflux_rms',
    'Δentropy_rms',
]
KLINE_INTERVAL = '30s'
BASE_WINDOW_S = 5
LEVELS = 3
EXPECTED_BAR_STARTS = [
    datetime(2025, 5, 23, 0, 0, 0),
    datetime(2025, 5, 23, 0, 0, 30),
    datetime(2025, 5, 23, 0, 1, 0),
]


def _fixture_trades() -> pl.DataFrame:
    '''
    Load the checked-in 90-second slice of real BTCUSDT trades.

    Returns:
        pl.DataFrame: The trades with a derived datetime column.
    '''

    historical = HistoricalData()
    historical.get_any_file(str(FIXTURE_PATH), has_header=False, columns=TRADE_COLUMNS)

    return historical.data


def test_conserved_flux_renormalization():
    '''
    The CFR transform must bucket every trade into klines without losing
    flux: per-bar dollar flow reconciles with vwap and volume, totals are
    conserved across the bucketing, the multi-scale features are finite,
    and the output is deterministic.
    '''

    trades_df = _fixture_trades()

    result = conserved_flux_renormalization(
        trades_df,
        kline_interval=KLINE_INTERVAL,
        base_window_s=BASE_WINDOW_S,
        levels=LEVELS,
    )

    assert result.columns == KLINE_COLUMNS + FEATURE_COLUMNS
    assert result.height == len(EXPECTED_BAR_STARTS)
    assert result['datetime'].to_list() == EXPECTED_BAR_STARTS

    assert result['volume'].sum() == pytest.approx(trades_df['quantity'].sum())
    expected_value_sum = (trades_df['price'] * trades_df['quantity']).sum()
    assert result['value_sum'].sum() == pytest.approx(expected_value_sum)
    assert result['value_sum'].sum() == pytest.approx(trades_df['quote_quantity'].sum())

    for row in result.iter_rows(named=True):
        assert row['vwap'] * row['volume'] == pytest.approx(row['value_sum'])
        assert row['low'] <= min(row['open'], row['close'])
        assert row['high'] >= max(row['open'], row['close'])
        assert row['low'] <= row['vwap'] <= row['high']

    features = result.select(FEATURE_COLUMNS)
    assert features.null_count().sum_horizontal().item() == 0
    for column in FEATURE_COLUMNS:
        assert features[column].is_finite().all()
        assert (features[column] >= 0.0).all()

    rerun = conserved_flux_renormalization(
        trades_df,
        kline_interval=KLINE_INTERVAL,
        base_window_s=BASE_WINDOW_S,
        levels=LEVELS,
    )
    assert result.equals(rerun)

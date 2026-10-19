from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal

from limen.backtest.trade_contract import ProductSpec, TradeInputs, TradePolicy, contract_digest, export_trade_contract, source_binding
from limen.experiment._prepare_trade_context import normalize_observations
from limen.features.volatility_barriers import volatility_barriers

FIXTURE = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'


def test_formula_warmup_and_split_boundary_invariance():
    raw = pl.read_parquet(FIXTURE, n_rows=96)
    configured = dict(span=6, regime_span=24, min_periods=3, minimum_bps=1, maximum_bps=1000)
    actual = volatility_barriers(raw, **configured)
    prior = pd.Series(raw['close'].to_list()).pct_change(fill_method=None).shift(1)
    fast = prior.ewm(span=6, adjust=True, min_periods=3).std(bias=False)
    slow = prior.ewm(span=24, adjust=True, min_periods=3).std(bias=False)
    multiplier = np.where(fast / slow < 0.75, 0.8, np.where(fast / slow > 1.25, 1.2, 1.0))
    expected = np.clip(10000 * fast * multiplier * 2, 1, 1000)
    np.testing.assert_allclose(actual['volatility_take_profit_bps'].to_numpy(), expected, equal_nan=True, atol=1e-10)
    assert actual['__barrier_ready__'].to_list() == [False] * 4 + [True] * 92
    for end in (24, 48, 72):
        assert_frame_equal(volatility_barriers(raw.head(end), **configured), actual.head(end))


def test_prefix_causality_and_unconfigured_compatibility():
    raw = pl.read_parquet(FIXTURE, n_rows=16)
    prices = normalize_observations(raw, interval_seconds=3600)
    start, end = int(prices['start_ns'][0]), int(prices['available_at_ns'][-1])
    signals = prices.select('row_id', 'available_at_ns').with_columns(pl.lit(None, dtype=pl.Float64).alias('target'))
    source = source_binding(prices, 'legacy:recorded-hourly-prefix', start, end, 1000000000, 'recorded_hourly_ohlc')
    inputs = TradeInputs(10000, start, end, signals, prices, None, (source,))
    policy = TradePolicy(ProductSpec('cash_spot', 'BTCUSDT', 'BTC', 'USDT', 0.00001, 5), fee_bps=10, slip_bps=5)
    assert contract_digest(export_trade_contract(policy, inputs)) == '9adc51c47e50e023f1d2d364400ba78e615bdf14304ab999ca66adc3b27ca9b7'
    assert 'barrier_spec' not in export_trade_contract(policy, inputs)['policy']
    assert replace(policy, take_profit_bps_column='entry_tp').take_profit_bps_column == 'entry_tp'

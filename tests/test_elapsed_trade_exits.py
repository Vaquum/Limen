from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import polars as pl
import pytest

from limen.backtest.execution_events import execution_events
from limen.backtest.trade_contract import NANOSECONDS, ProductSpec, TradeInputs, TradePolicy, source_binding
from limen.backtest.trade_execution import trade_execution
from tests.test_target_exposure import _case


def _ticks():
    source = Path(__file__).parent / 'fixtures/binance_trades_btcusdt_2025-05-23_90s.zip'
    with ZipFile(source) as archive, archive.open(archive.namelist()[0]) as stream:
        trades = pl.read_csv(stream, has_header=False)
    trades = trades.select(pl.col('column_5').cast(pl.Int64).mul(1000).alias('time'), pl.col('column_2').alias('price')).group_by('time', maintain_order=True).agg(pl.col('price').last())
    trades = trades.filter(pl.col('time') <= trades['time'][0] + 10 * NANOSECONDS)
    times = trades['time']
    points = pl.DataFrame({'row_id': [f'tick:{i}' for i in range(trades.height)], 'start_ns': times, 'end_ns': times, 'open_available_at_ns': times, 'available_at_ns': times, **dict.fromkeys(('open', 'high', 'low', 'close'), trades['price'])})
    start, end = int(times[0]), int(times[-1])
    signals = pl.DataFrame({'row_id': ['entry', 'last'], 'available_at_ns': [start, end], 'target': [0.5, 0.5]})
    binding = source_binding(points, str(source), start, end, 1000, 'recorded Binance trade microsecond timestamps')
    return TradeInputs(10000.0, start, end, signals, points, None, (binding,)), TradePolicy(ProductSpec('linear_perpetual', 'BTCUSDT', 'BTC', 'USDT', 1e-9, 0.0))


def test_irregular_bars_use_elapsed_duration():
    inputs, policy = _case([0.5] * 9)
    observations = inputs.observations[[0, 1, 4, 7, 8]]
    signals = inputs.signals[[0, 1, 4, 7, 8]]
    binding = source_binding(observations, 'recorded irregular close subset', inputs.partition_start_ns, inputs.partition_end_ns, 900 * NANOSECONDS, 'recorded points')
    result = trade_execution(replace(inputs, observations=observations, signals=signals, sources=(binding,)), replace(policy, max_holding_seconds=3600.0))
    assert result.episodes['closed_at_ns'][0] == signals['available_at_ns'][2]
    assert result.episodes['exit_reason'][0] == 'time_stop'
    assert result.fills.height == 2


def test_timer_and_second_execution_source():
    inputs, policy = _ticks()
    policy = replace(policy, max_holding_seconds=1.1, timer_interval_seconds=1.0, max_price_gap_seconds=1.0)
    result = trade_execution(inputs, policy)
    request = result.intents.filter(pl.col('reason') == 'time_stop')['requested_at_ns'][0]
    fill = result.episodes['closed_at_ns'][0]
    deadline = inputs.partition_start_ns + round(1.1 * NANOSECONDS)
    expected_request = -(-deadline // NANOSECONDS) * NANOSECONDS
    assert request == expected_request
    assert fill == inputs.observations.filter(pl.col('available_at_ns') >= request)['available_at_ns'][0]
    assert fill >= request
    assert result.fills.height == 2
    assert len([event for event in execution_events(inputs, policy) if event.kind == 'signal']) == 2


def test_missing_precision_and_equal_time_order():
    inputs, policy = _ticks()
    sparse = inputs.observations[[0, -1]]
    binding = source_binding(sparse, 'recorded sparse endpoints', inputs.partition_start_ns, inputs.partition_end_ns, 1000, 'recorded points')
    with pytest.raises(ValueError, match='gap'):
        trade_execution(replace(inputs, observations=sparse, sources=(binding,)), replace(policy, max_price_gap_seconds=1.0))
    events = execution_events(inputs, replace(policy, timer_interval_seconds=1.0))
    assert events == tuple(sorted(events, key=lambda event: (event.time_ns, {'funding': 0, 'observation': 1, 'timer': 2, 'signal': 3}[event.kind], event.observation_phase == 'open', event.event_id)))
    null_time = inputs.signals.with_columns(pl.lit(None, dtype=pl.Int64).alias('available_at_ns'))
    with pytest.raises(ValueError, match='availability'):
        execution_events(replace(inputs, signals=null_time), policy)
    with pytest.raises(ValueError, match='fingerprint'):
        trade_execution(replace(inputs, observations=inputs.observations.with_columns([(pl.col(key) + 1).alias(key) for key in ('open', 'high', 'low', 'close')])), policy)


def test_disabled_and_resize_deadline():
    inputs, policy = _case([0.5, 0.75, 0.75, 0.75, 0.0, 0.5])
    disabled = trade_execution(inputs, policy)
    assert disabled.episodes['closed_at_ns'][0] == inputs.signals['available_at_ns'][4]
    result = trade_execution(inputs, replace(policy, max_holding_seconds=3600.0))
    assert result.episodes['closed_at_ns'][0] == inputs.signals['available_at_ns'][2]
    assert result.episodes.height == 2
    assert result.episodes['first_fill_ns'][0] == inputs.partition_start_ns
    assert result.fills['episode_id'][0] == result.fills['episode_id'][1]


def test_adjacent_close_precedes_current_open_and_stale_open_is_unusable():
    from limen.experiment._prepare_trade_context import normalize_observations

    source = pl.read_parquet(Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet').with_row_index('index')
    index = source.filter(pl.col('open') != pl.col('close').shift(1))['index'][0]
    prices = normalize_observations(source.slice(index - 1, 2), interval_seconds=900).with_columns(pl.Series('row_id', ['9', '10']))
    start, end = int(prices['start_ns'][0]), int(prices['end_ns'][-1])
    available = int(prices['start_ns'][1])
    signals = pl.DataFrame({'row_id': ['entry'], 'available_at_ns': [available], 'target': [0.5]})
    binding = source_binding(prices, 'recorded adjacent bars', start, end, 900 * NANOSECONDS, 'recorded OHLC')
    inputs, policy = _case([0.5])
    inputs = replace(inputs, partition_start_ns=start, partition_end_ns=end, observations=prices, signals=signals, sources=(binding,))
    events = [event for event in execution_events(inputs, policy) if event.time_ns == available and event.kind == 'observation']
    assert [(event.source_row_id, event.observation_phase) for event in events] == [('9', 'close'), ('10', 'open')]
    ledger = trade_execution(inputs, policy)
    assert ledger.fills['reference_price'][0] == prices['open'][1]
    assert ledger.fills['evidence_row_id'][0] == '10'
    late = prices.with_columns(pl.col('available_at_ns').alias('open_available_at_ns'))
    bound = source_binding(late, 'recorded delayed opens', start, end, 900 * NANOSECONDS, 'delayed OHLC availability')
    assert all(event.observation_phase != 'open' for event in execution_events(replace(inputs, observations=late, sources=(bound,)), policy))

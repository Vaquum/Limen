import math
from dataclasses import replace
from typing import cast

import polars as pl

from limen.backtest.trade_contract import ExecutionEvent, NANOSECONDS, TradeInputs, TradePolicy, finite_number

OBSERVATION_COLUMNS = ('row_id', 'start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns', 'open', 'high', 'low', 'close')
SIGNAL_COLUMNS = ('row_id', 'available_at_ns', 'target')


def validate_observations(inputs: TradeInputs, policy: TradePolicy) -> None:
    if finite_number(inputs.initial_equity, 'initial equity') <= 0 or inputs.partition_end_ns < inputs.partition_start_ns:
        raise ValueError('Positive initial equity and ordered partition bounds are required')
    observations = inputs.observations
    if not observations.height or set(OBSERVATION_COLUMNS) - set(observations.columns):
        raise ValueError('Execution observations require recorded prices and interval/availability timestamps')
    if observations.select(OBSERVATION_COLUMNS).null_count().sum_horizontal()[0] or observations['row_id'].is_duplicated().any():
        raise ValueError('Execution observation identity/values are missing or duplicated')
    previous_end = None
    previous_available = None
    for row in observations.iter_rows(named=True):
        start, end, opened, available = (int(row[key]) for key in ('start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns'))
        prices = [finite_number(row[key], key) for key in ('open', 'high', 'low', 'close')]
        if start > end or opened < start or available < end or available < opened or min(prices) <= 0 or prices[1] < max(prices[0], prices[3]) or prices[2] > min(prices[0], prices[3]):
            raise ValueError('Invalid causal observation interval or OHLC')
        if start == end and len(set(prices)) != 1:
            raise ValueError('Point observation must contain one recorded price')
        if previous_end is not None and start < previous_end:
            raise ValueError('Execution observations overlap or are out of order')
        if previous_available is not None and available < previous_available:
            raise ValueError('Execution availability is out of order')
        if previous_end is not None and policy.max_price_gap_seconds is not None and start - previous_end > policy.max_price_gap_seconds * NANOSECONDS:
            raise ValueError('Execution price gap exceeds declared precision')
        previous_end, previous_available = end, available
    if int(observations['start_ns'][0]) > inputs.partition_start_ns or int(observations['end_ns'][-1]) < inputs.partition_end_ns:
        raise ValueError('Execution source does not cover the declared partition')
    if not inputs.sources:
        raise ValueError('Execution source binding is required')


def execution_events(inputs: TradeInputs, policy: TradePolicy) -> tuple[ExecutionEvent, ...]:
    validate_observations(inputs, policy)
    if set(SIGNAL_COLUMNS) - set(inputs.signals.columns) or inputs.signals['row_id'].is_duplicated().any():
        raise ValueError('Signals require unique row identity, availability and target')
    events: list[ExecutionEvent] = []
    for row in inputs.observations.iter_rows(named=True):
        identity = str(row['row_id'])
        if row['start_ns'] != row['end_ns']:
            events.append(ExecutionEvent(f'price:{identity}:open', int(row['open_available_at_ns']), 'observation', identity, int(row['open_available_at_ns']), 'open'))
        events.append(ExecutionEvent(f'price:{identity}:close', int(row['available_at_ns']), 'observation', identity, int(row['available_at_ns']), 'close'))
    for row in inputs.signals.iter_rows(named=True):
        time = int(row['available_at_ns'])
        events.append(ExecutionEvent(f'signal:{row["row_id"]}', time, 'signal', str(row['row_id']), time))
    if policy.timer_interval_seconds is not None:
        interval = round(policy.timer_interval_seconds * NANOSECONDS)
        phase = round(policy.timer_phase_utc_seconds * NANOSECONDS)
        if interval <= 0:
            raise ValueError('Timer cadence is below timestamp precision')
        first = phase + math.ceil((inputs.partition_start_ns - phase) / interval) * interval
        events.extend(ExecutionEvent(f'timer:{time}', time, 'timer', None, time) for time in range(first, inputs.partition_end_ns + 1, interval))
    # Funding clocks are added by the native funding preparation, independently
    # of prediction/bar clocks.
    if inputs.funding_events is not None and 'kind' in inputs.funding_events.columns:
        for row in inputs.funding_events.iter_rows(named=True):
            time = int(row['time_ns'])
            events.append(ExecutionEvent(f'funding:{row["event_id"]}', time, 'funding', str(row['event_id']), time))
    priority = {'funding': 0, 'observation': 1, 'timer': 2, 'signal': 3}
    selected = [event for event in events if inputs.partition_start_ns <= event.time_ns <= inputs.partition_end_ns]
    selected.append(ExecutionEvent(f'end:{inputs.partition_end_ns}', inputs.partition_end_ns, 'timer', None, inputs.partition_end_ns))
    return tuple(sorted(selected, key=lambda event: (event.time_ns, priority[event.kind], event.event_id)))


def with_predictions(inputs: TradeInputs, predictions: object) -> TradeInputs:
    import numpy as np

    values = np.asarray(predictions, dtype=float)
    if values.ndim != 1 or len(values) != inputs.signals.height:
        raise ValueError('Prediction count does not match causal signal timeline')
    signals = inputs.signals.with_columns(pl.Series('target', cast(list[float], values.tolist())))
    return replace(inputs, signals=signals)


__all__ = ['OBSERVATION_COLUMNS', 'SIGNAL_COLUMNS', 'execution_events', 'validate_observations', 'with_predictions']

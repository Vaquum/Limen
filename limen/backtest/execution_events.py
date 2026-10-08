from dataclasses import replace
from typing import cast

import polars as pl

from limen.backtest.trade_contract import ExecutionEvent, NANOSECONDS, TradeInputs, TradePolicy, finite_number, source_binding

OBSERVATION_COLUMNS: tuple[str, ...] = ('row_id', 'start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns', 'open', 'high', 'low', 'close')
SIGNAL_COLUMNS: tuple[str, ...] = ('row_id', 'available_at_ns', 'target')


def validate_observations(inputs: TradeInputs, policy: TradePolicy) -> None:
    for value in cast(tuple[object, object], (inputs.partition_start_ns, inputs.partition_end_ns)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError('Partition timestamps must be integer UTC nanoseconds')
    if finite_number(inputs.initial_equity, 'initial equity') <= 0 or inputs.partition_end_ns < inputs.partition_start_ns:
        raise ValueError('Positive initial equity and ordered partition bounds are required')
    observations = inputs.observations
    if not observations.height or set(OBSERVATION_COLUMNS) - set(observations.columns):
        raise ValueError('Execution observations require recorded prices and interval/availability timestamps')
    if observations.select(OBSERVATION_COLUMNS).null_count().sum_horizontal()[0] or observations['row_id'].is_duplicated().any():
        raise ValueError('Execution observation identity/values are missing or duplicated')
    if any(observations[name].dtype != pl.Int64 for name in ('start_ns', 'end_ns', 'open_available_at_ns', 'available_at_ns')):
        raise ValueError('Observation timestamps must be integer UTC nanoseconds')
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
        if policy.max_price_gap_seconds is not None and (end - start > policy.max_price_gap_seconds * NANOSECONDS or (previous_end is not None and start - previous_end > policy.max_price_gap_seconds * NANOSECONDS)):
            raise ValueError('Execution price gap exceeds declared precision')
        previous_end, previous_available = end, available
    if int(observations['start_ns'][0]) > inputs.partition_start_ns or int(observations['end_ns'][-1]) < inputs.partition_end_ns:
        raise ValueError('Execution source does not cover the declared partition')
    if not inputs.sources:
        raise ValueError('Execution source binding is required')
    fingerprint = source_binding(observations, '', 0, 0, 1, '').checksum
    if not any(source.checksum == fingerprint for source in inputs.sources):
        raise ValueError('Execution source fingerprint changed')


def execution_events(inputs: TradeInputs, policy: TradePolicy) -> tuple[ExecutionEvent, ...]:
    validate_observations(inputs, policy)
    if set(SIGNAL_COLUMNS) - set(inputs.signals.columns) or inputs.signals['row_id'].is_duplicated().any():
        raise ValueError('Signals require unique row identity, availability and target')
    if inputs.signals['row_id'].null_count() or inputs.signals['available_at_ns'].null_count() or inputs.signals['available_at_ns'].dtype != pl.Int64:
        raise ValueError('Signal identity and integer UTC availability are required')
    if inputs.signals.height and (int(cast(int, inputs.signals['available_at_ns'].min())) < inputs.partition_start_ns or int(cast(int, inputs.signals['available_at_ns'].max())) > inputs.partition_end_ns or not inputs.signals['available_at_ns'].is_sorted()):
        raise ValueError('Signal timeline is outside or out of order in its true partition')
    events: list[ExecutionEvent] = []
    for row in inputs.observations.iter_rows(named=True):
        identity = str(row['row_id'])
        if row['start_ns'] != row['end_ns'] and row['open_available_at_ns'] == row['start_ns']:
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
        first = phase + -(-(inputs.partition_start_ns - phase) // interval) * interval
        events.extend(ExecutionEvent(f'timer:{time}', time, 'timer', None, time) for time in range(first, inputs.partition_end_ns + 1, interval))
    # Funding clocks are added by the native funding preparation, independently
    # of prediction/bar clocks.
    if inputs.funding_events is not None:
        required: tuple[str, ...] = ('event_id', 'time_ns', 'kind', 'start_ns', 'end_ns', 'rate_decimal', 'rate_basis_seconds', 'valuation_price')
        if not set(inputs.funding_events.columns).issuperset(required) or inputs.funding_events['event_id'].is_duplicated().any() or inputs.funding_events.select('event_id', 'time_ns', 'kind', 'start_ns', 'end_ns', 'rate_decimal', 'rate_basis_seconds').null_count().sum_horizontal()[0]:
            raise ValueError('Funding requires validated normalized event clocks and values')
        if any(inputs.funding_events[key].dtype != pl.Int64 for key in ('time_ns', 'start_ns', 'end_ns')) or not set(inputs.funding_events['kind'].to_list()) <= {'payment', 'accrual', 'cash_settlement'}:
            raise ValueError('Funding clocks require integer UTC timestamps and recognized kinds')
        fingerprint = source_binding(inputs.funding_events, '', 0, 0, 1, '').checksum
        if not any(source.checksum == fingerprint for source in inputs.sources):
            raise ValueError('Funding source fingerprint changed')
        for row in inputs.funding_events.iter_rows(named=True):
            time = max(inputs.partition_start_ns, int(row['time_ns'])) if row['kind'] == 'accrual' else int(row['time_ns'])
            events.append(ExecutionEvent(f'funding:{row["event_id"]}', time, 'funding', str(row['event_id']), time))
            if row['kind'] == 'accrual':
                end = int(row['end_ns'])
                events.append(ExecutionEvent(f'funding:{row["event_id"]}:end', end, 'funding', str(row['event_id']), end))
    priority = {'funding': 0, 'observation': 1, 'timer': 2, 'signal': 3}
    source_order = {str(row_id): index for index, row_id in enumerate(inputs.observations['row_id'])}
    signal_order = {str(row_id): index for index, row_id in enumerate(inputs.signals['row_id'])}
    selected = [event for event in events if inputs.partition_start_ns <= event.time_ns <= inputs.partition_end_ns]
    selected.append(ExecutionEvent(f'end:{inputs.partition_end_ns}', inputs.partition_end_ns, 'timer', None, inputs.partition_end_ns))
    return tuple(sorted(selected, key=lambda event: (event.time_ns, priority[event.kind], event.observation_phase == 'open', (signal_order if event.kind == 'signal' else source_order).get(str(event.source_row_id), 0), event.event_id)))


def with_predictions(inputs: TradeInputs, predictions: object) -> TradeInputs:
    import numpy as np

    values = np.asarray(predictions, dtype=object)
    if values.ndim != 1 or len(values) != inputs.signals.height:
        raise ValueError('Prediction count does not match causal signal timeline')
    targets = [None if value is None else finite_number(value, 'prediction') for value in cast(list[object], values.tolist())]
    signals = inputs.signals.with_columns(pl.Series('target', targets, dtype=pl.Float64))
    return replace(inputs, signals=signals)


__all__ = ['OBSERVATION_COLUMNS', 'SIGNAL_COLUMNS', 'execution_events', 'validate_observations', 'with_predictions']

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, cast

import polars as pl

from limen.backtest.trade_contract import BPS, NANOSECONDS, FundingPolicy, JsonValue, finite_number

ADAPTER_VERSION = 'limen-funding-v1'
FUNDING_KEYS = frozenset(('rate', 'rate_unit', 'rate_basis_seconds', 'mechanism', 'settlement_interval_seconds', 'settlement_phase_utc_seconds', 'cash_settlement_interval_seconds', 'cash_settlement_phase_utc_seconds', 'valuation', 'currency', 'history_interpretation', 'approximation'))
_SCHEMA = {'event_id': pl.String, 'kind': pl.String, 'time_ns': pl.Int64, 'start_ns': pl.Int64, 'end_ns': pl.Int64, 'rate_decimal': pl.Float64, 'rate_basis_seconds': pl.Float64, 'valuation_price': pl.Float64}


@dataclass(frozen=True)
class FundingContext:
    event_id: str
    start_ns: int
    end_ns: int
    kind: Literal['payment', 'accrual', 'cash_settlement']
    accrued_funding: float
    quantity: float
    valuation_price: float
    rate_decimal: float
    rate_basis_seconds: float
    mechanism: Literal['discrete', 'continuous']
    currency: str


@dataclass(frozen=True)
class FundingCashflow:
    event_id: str
    time_ns: int
    recognized_delta: float
    cash_delta: float
    currency: str


class FundingAdapter(Protocol):
    def __call__(self, context: FundingContext, *, params: Mapping[str, JsonValue]) -> FundingCashflow: ...


def funding_adapter(context: FundingContext, *, params: Mapping[str, JsonValue]) -> FundingCashflow:
    if set(params) - FUNDING_KEYS:
        raise ValueError('Unknown native funding parameters')
    if context.kind == 'cash_settlement':
        return FundingCashflow(context.event_id, context.end_ns, 0.0, context.accrued_funding, context.currency)
    for name in ('quantity', 'valuation_price', 'rate_decimal', 'rate_basis_seconds'):
        _ = finite_number(getattr(context, name), name)
    if context.end_ns < context.start_ns or context.rate_basis_seconds <= 0 or (context.quantity != 0 and context.valuation_price <= 0):
        raise ValueError('Invalid funding interval, basis or valuation')
    delta = -context.quantity * context.valuation_price * context.rate_decimal
    if context.kind == 'accrual':
        delta *= (context.end_ns - context.start_ns) / NANOSECONDS / context.rate_basis_seconds
    if not math.isfinite(delta):
        raise ValueError('Nonfinite funding cashflow')
    return FundingCashflow(context.event_id, context.end_ns, delta, delta if context.kind == 'payment' else 0.0, context.currency)


def resolve_adapter(policy: FundingPolicy) -> FundingAdapter:
    module, _, name = policy.adapter_ref.rpartition('.')
    value: object = getattr(importlib.import_module(module), name)
    if not callable(value):
        raise ValueError('Funding adapter must be callable')
    if value is funding_adapter and policy.adapter_version != ADAPTER_VERSION:
        raise ValueError('Unsupported native funding adapter version')
    return cast(FundingAdapter, value)


def scheduled_times(start: int, end: int, interval_seconds: object, phase_seconds: object) -> range:
    interval = round(finite_number(interval_seconds, 'settlement interval') * NANOSECONDS)
    phase = round(finite_number(phase_seconds, 'settlement phase') * NANOSECONDS)
    if interval <= 0:
        raise ValueError('Settlement interval must be positive at timestamp precision')
    first = phase - (-(start - phase) // interval) * interval
    return range(first, end + 1, interval)


def _history_ns(frame: pl.DataFrame, column: str) -> pl.Series:
    if column not in frame.columns:
        raise ValueError(f'Funding history requires {column}')
    series = frame[column]
    if isinstance(series.dtype, pl.Datetime):
        series = series.dt.epoch('ns')
    if series.dtype != pl.Int64 or series.null_count():
        raise ValueError(f'Funding {column} must be UTC integer nanoseconds or datetime')
    return series


def prepare_funding(policy: FundingPolicy, history: pl.DataFrame | None, start: int, end: int) -> pl.DataFrame:
    params = policy.params
    rows: list[dict[str, JsonValue]] = []
    if history is None:
        if policy.approximation != 'scenario' or policy.valuation != 'execution_proxy':
            raise ValueError('No-history funding requires an explicit scenario/execution proxy')
        rate = finite_number(params.get('rate'), 'funding rate')
        if params.get('rate_unit') == 'bps':
            rate /= BPS
        elif params.get('rate_unit') != 'decimal':
            raise ValueError('Funding rate unit must be decimal or bps')
        if policy.mechanism == 'continuous':
            rows.append({'event_id': 'scenario', 'kind': 'accrual', 'time_ns': start, 'start_ns': start, 'end_ns': end, 'rate_decimal': rate, 'rate_basis_seconds': policy.rate_basis_seconds, 'valuation_price': None})
        else:
            rows.extend({'event_id': f'payment:{time}', 'kind': 'payment', 'time_ns': time, 'start_ns': time, 'end_ns': time, 'rate_decimal': rate, 'rate_basis_seconds': policy.rate_basis_seconds, 'valuation_price': None} for time in scheduled_times(start, end, params.get('settlement_interval_seconds'), params.get('settlement_phase_utc_seconds', 0.0)))
    else:
        if 'rate' in params:
            raise ValueError('Historical funding conflicts with a constant rate')
        if params.get('rate_unit', 'decimal') != 'decimal':
            raise ValueError('Historical rate_decimal conflicts with a nondecimal rate unit')
        required = {'event_id', 'rate_decimal', 'valuation_price'}
        if required - set(history.columns) or history['event_id'].is_duplicated().any() or history.select('event_id', 'rate_decimal').null_count().sum_horizontal()[0]:
            raise ValueError('Funding history requires unique events and realized decimal rates/valuation')
        if policy.mechanism == 'discrete':
            times = _history_ns(history, 'settlement_at').to_list()
            # Historical schedules are supplied explicitly, never inferred from
            # a venue's current preset. Schedule changes can be separate rows.
            if not {'schedule_start', 'schedule_end', 'settlement_interval_seconds', 'settlement_phase_utc_seconds'} <= set(history.columns):
                raise ValueError('Historical discrete funding requires recorded schedule coverage')
            history = history.with_columns(_history_ns(history, 'schedule_start'), _history_ns(history, 'schedule_end'))
            expected: set[int] = set()
            supports: list[tuple[int, int]] = []
            schedules = history.select('schedule_start', 'schedule_end', 'settlement_interval_seconds', 'settlement_phase_utc_seconds').unique().sort('schedule_start')
            for row in schedules.iter_rows(named=True):
                left, right = int(row['schedule_start']), int(row['schedule_end'])
                supports.append((left, right))
                expected.update(scheduled_times(max(start, left), min(end, right - 1), row['settlement_interval_seconds'], row['settlement_phase_utc_seconds']))
            _coverage(supports, start, end + 1)
            slots = _history_ns(history, 'settlement_slot_ns').to_list() if 'settlement_slot_ns' in history.columns else times
            actual = {int(slot) for slot in slots if start <= int(slot) <= end}
            if actual != expected or len(times) != len(set(times)) or len(slots) != len(set(slots)):
                raise ValueError('Historical funding payment coverage is incomplete or duplicated')
            for row, time, slot in zip(history.iter_rows(named=True), times, slots, strict=True):
                if not 0 <= time - slot < finite_number(row['settlement_interval_seconds'], 'recorded settlement interval') * NANOSECONDS:
                    raise ValueError('Recorded settlement timestamp is outside its declared schedule slot')
                rows.append({'event_id': f"history:{row['event_id']}", 'kind': 'payment', 'time_ns': int(time), 'start_ns': int(time), 'end_ns': int(time), 'rate_decimal': finite_number(row['rate_decimal'], 'realized funding rate'), 'rate_basis_seconds': policy.rate_basis_seconds, 'valuation_price': _valuation(row['valuation_price'], policy)})
        else:
            begins, ends = _history_ns(history, 'start'), _history_ns(history, 'end')
            _coverage([(int(left), int(right)) for left, right in zip(begins, ends, strict=True)], start, end)
            interpretation = params.get('history_interpretation')
            if interpretation not in ('quoted', 'integrated'):
                raise ValueError('Continuous history must declare quoted or integrated interval rates')
            for row, left, right in zip(history.iter_rows(named=True), begins, ends, strict=True):
                basis = (right - left) / NANOSECONDS if interpretation == 'integrated' else finite_number(row.get('rate_basis_seconds'), 'recorded rate basis')
                if basis <= 0:
                    raise ValueError('Recorded rate basis must be positive')
                rows.append({'event_id': f"history:{row['event_id']}", 'kind': 'accrual', 'time_ns': int(left), 'start_ns': int(left), 'end_ns': int(right), 'rate_decimal': finite_number(row['rate_decimal'], 'realized funding rate'), 'rate_basis_seconds': basis, 'valuation_price': _valuation(row['valuation_price'], policy)})
    if policy.mechanism == 'continuous':
        rows.extend({'event_id': f'settlement:{time}', 'kind': 'cash_settlement', 'time_ns': time, 'start_ns': time, 'end_ns': time, 'rate_decimal': 0.0, 'rate_basis_seconds': policy.rate_basis_seconds, 'valuation_price': None} for time in scheduled_times(start, end, params.get('cash_settlement_interval_seconds'), params.get('cash_settlement_phase_utc_seconds', 0.0)))
    return pl.DataFrame(rows, schema=_SCHEMA).sort('time_ns', 'event_id')


def _valuation(value: object, policy: FundingPolicy) -> float | None:
    if value is None and policy.valuation == 'execution_proxy' and policy.approximation != 'recorded_exact':
        return None
    price = finite_number(value, 'funding valuation')
    if price <= 0:
        raise ValueError('Funding valuation must be positive')
    return price


def _coverage(intervals: list[tuple[int, int]], start: int, end: int) -> None:
    cursor = start
    for left, right in intervals:
        if right <= start or left >= end:
            continue
        if right <= left or left > cursor or (left < cursor and cursor != start):
            raise ValueError('Funding support has a gap/overlap or invalid interval')
        cursor = right
    if cursor < end:
        raise ValueError('Funding support does not cover the partition')


__all__ = ['ADAPTER_VERSION', 'FUNDING_KEYS', 'FundingAdapter', 'FundingCashflow', 'FundingContext', 'funding_adapter', 'prepare_funding', 'resolve_adapter', 'scheduled_times']

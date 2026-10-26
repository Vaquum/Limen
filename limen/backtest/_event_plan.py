from collections import OrderedDict
from dataclasses import dataclass, replace
from threading import Lock
from typing import cast

from limen.backtest.execution_events import OBSERVATION_COLUMNS, SIGNAL_COLUMNS, source_execution_events, validate_initial, validate_signal_barriers
from limen.backtest.trade_contract import ExecutionEvent, TradeInputs, TradePolicy, contract_digest, export_trade_contract, json_value, source_binding


@dataclass(frozen=True)
class EventPlan:
    events: tuple[ExecutionEvent, ...]
    times: tuple[int, ...]
    observations: dict[str, dict[str, object]]
    funding: dict[str, dict[str, object]]
    observation_order: dict[str, int]
    signal_order: dict[str, int]
    binding: str


_PLANS: OrderedDict[str, EventPlan] = OrderedDict()
_MAX_PLANS = 8
_MAX_EVENTS = 1_000_000
_PLAN_LOCK = Lock()


def _plan_key(inputs: TradeInputs, policy: TradePolicy) -> str:
    # Hash current buffers, rather than object identity: frozen TradeInputs can
    # still contain a DataFrame or funding parameter mapping changed in place.
    projected = json_value(policy)
    if isinstance(projected, dict):
        _ = projected.pop('barrier_spec', None)
    return contract_digest({
        'policy': projected, 'sources': json_value(inputs.sources),
        'initial_equity': str(inputs.initial_equity),
        'partition': [str(inputs.partition_start_ns), str(inputs.partition_end_ns)],
        'observations': source_binding(inputs.observations, '', 0, 0, 1, '').checksum,
        'signals': source_binding(inputs.signals.select('row_id', 'available_at_ns'), '', 0, 0, 1, '').checksum,
        'signal_schema': str(inputs.signals.schema),
        'funding': None if inputs.funding_events is None else source_binding(inputs.funding_events, '', 0, 0, 1, '').checksum,
    })


def event_plan(inputs: TradeInputs, policy: TradePolicy, *, cache: bool = True) -> EventPlan:
    validate_initial(inputs)
    validate_signal_barriers(inputs, policy)
    if set(OBSERVATION_COLUMNS) - set(inputs.observations.columns) or set(SIGNAL_COLUMNS) - set(inputs.signals.columns):
        _ = source_execution_events(inputs, policy, include_timers=False)
    key = _plan_key(inputs, policy) if cache else None
    if key is not None:
        with _PLAN_LOCK:
            existing = _PLANS.get(key)
            if existing is not None:
                _PLANS.move_to_end(key)
                return replace(existing, binding=contract_digest(export_trade_contract(policy, inputs))) if policy.take_profit_bps_column or policy.stop_loss_bps_column else existing
    events = source_execution_events(inputs, policy, include_timers=False)
    observation_ids = {event.source_row_id for event in events if event.kind == 'observation'}
    funding_ids = {event.source_row_id for event in events if event.kind == 'funding'}
    plan = EventPlan(
        events, tuple(event.time_ns for event in events),
        {str(row['row_id']): cast(dict[str, object], row) for row in inputs.observations.iter_rows(named=True) if str(row['row_id']) in observation_ids},
        {str(row['event_id']): cast(dict[str, object], row) for row in inputs.funding_events.iter_rows(named=True) if str(row['event_id']) in funding_ids} if inputs.funding_events is not None else {},
        {str(identity): index for index, identity in enumerate(inputs.observations['row_id']) if str(identity) in observation_ids},
        {str(identity): index for index, identity in enumerate(inputs.signals['row_id'])},
        contract_digest(export_trade_contract(policy, inputs)),
    )
    if key is not None and len(events) <= _MAX_EVENTS:
        with _PLAN_LOCK:
            _PLANS[key] = plan
            while len(_PLANS) > _MAX_PLANS or sum(len(item.events) for item in _PLANS.values()) > _MAX_EVENTS:
                _ = _PLANS.popitem(last=False)
    return plan


__all__: list[str] = []

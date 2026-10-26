from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import ClassVar, Literal

import polars as pl

from limen.backtest._event_plan import EventPlan, event_plan
from limen.backtest.funding_adapter import resolve_adapter
from limen.backtest.trade_contract import ExecutionEvent, NANOSECONDS, TradeInputs, TradePolicy, contract_digest, export_trade_contract, finite_number, source_binding
from limen.backtest.trade_execution import Account, execute_event_plan


@dataclass(frozen=True)
class TradeTargetContext:
    policy: TradePolicy
    inputs: TradeInputs
    partition_start_ns: int
    partition_end_ns: int
    contract_digest: str


@dataclass(frozen=True)
class OutcomeLabels:
    rows: pl.DataFrame
    contract_digest: str


@dataclass(frozen=True)
class _Timeline:
    prices: list[int]
    signals: list[int]
    starts: list[int]
    available: list[int]
    plan: EventPlan
    candidates: dict[int, EventPlan] = field(default_factory=dict[int, EventPlan])

    @cached_property
    def has_accrual(self) -> bool:
        return any(row['kind'] == 'accrual' for row in self.plan.funding.values())


def _candidate(inputs: TradeInputs, policy: TradePolicy, index: int, side: float, timeline: _Timeline) -> dict[str, object]:
    available = timeline.signals[index]
    if (policy.take_profit_bps_column or policy.stop_loss_bps_column) and inputs.signals['__barrier_ready__'][index] is not True:
        return {'return': None, 'available': False, 'entry_ns': None, 'exit_ns': None, 'label_available_ns': None, 'exit_reason': None, 'unavailable_reason': 'barrier_warmup'}
    end = inputs.partition_end_ns
    if policy.max_holding_seconds is not None:
        entry = bisect_left(timeline.prices, available + round(policy.execution_lag_seconds * NANOSECONDS))
        if entry < len(timeline.prices):
            deadline = timeline.prices[entry] + round(policy.max_holding_seconds * NANOSECONDS)
            decision = bisect_left(timeline.signals, deadline)
            decisions = timeline.signals[decision:decision + 1]
            if policy.timer_interval_seconds is not None:
                interval, phase = round(policy.timer_interval_seconds * NANOSECONDS), round(policy.timer_phase_utc_seconds * NANOSECONDS)
                decisions.append(phase + -(-(deadline - phase) // interval) * interval)
            if decisions:
                exit_index = bisect_left(timeline.prices, min(decisions))
                if exit_index < len(timeline.prices):
                    end = min(end, timeline.prices[exit_index])
    first = max(0, bisect_left(timeline.available, available) - 1)
    last = bisect_right(timeline.starts, end)
    adapter = resolve_adapter(policy.funding) if policy.funding is not None else None
    if (adapter is not None) != (inputs.funding_events is not None):
        raise ValueError('Funding policy and normalized events must be supplied together')
    start = max(inputs.partition_start_ns, timeline.starts[first])
    if not timeline.has_accrual:
        plan = timeline.plan
        def selected(event: ExecutionEvent) -> bool:
            if event.kind == 'observation':
                return first <= plan.observation_order[str(event.source_row_id)] < last
            return event.kind != 'signal' or plan.signal_order[str(event.source_row_id)] >= index
        account = Account(policy, inputs.initial_equity, plan.binding[:16])
        signals = {str(row['row_id']): {**row, 'target': side} for row in inputs.signals.iter_rows(named=True)} if policy.take_profit_bps_column or policy.stop_loss_bps_column else side
        execute_event_plan(account, plan, signals, start, end, adapter, selected=selected, stop_at_close=adapter is None, keep_states=False)
        episodes = account.episodes
        if not episodes or episodes[0]['closed_at_ns'] is None:
            return {'return': None, 'available': False, 'entry_ns': None if not episodes else episodes[0]['first_fill_ns'], 'exit_ns': None, 'label_available_ns': None, 'exit_reason': None}
        episode = episodes[0]
        net_return = finite_number(episode['net_pnl'], 'completed candidate PnL') / finite_number(episode['entry_equity'], 'candidate entry equity')
        return {'return': net_return, 'available': True, 'entry_ns': episode['first_fill_ns'], 'exit_ns': episode['closed_at_ns'], 'label_available_ns': episode['closed_at_ns'], 'exit_reason': episode['exit_reason']}
    signals = inputs.signals.slice(index, bisect_right(timeline.signals, end) - index).with_columns(pl.lit(side).alias('target'))
    plan = timeline.candidates.get(index)
    if plan is None:
        observations = inputs.observations.slice(first, last - first)
        funding = inputs.funding_events
        if funding is not None:
            funding = funding.filter(pl.when(pl.col('kind') == 'accrual').then((pl.col('end_ns') >= start) & (pl.col('start_ns') <= end)).otherwise(pl.col('time_ns').is_between(start, end)))
        sources = [*inputs.sources, source_binding(observations, 'candidate:prices', start, end, 1, 'recorded_candidate_execution')]
        if funding is not None:
            sources.append(source_binding(funding, 'candidate:funding', start, end, 1, 'bound_candidate_funding'))
        candidate = replace(inputs, partition_start_ns=start, partition_end_ns=end, signals=signals, observations=observations, funding_events=funding, sources=tuple(sources))
        plan = event_plan(candidate, policy, cache=False)
        # Only the immediately paired long/short labels share this source plan.
        timeline.candidates.clear()
        timeline.candidates[index] = plan
    account = Account(policy, inputs.initial_equity, plan.binding[:16])
    targets = {str(row['row_id']): row if policy.take_profit_bps_column or policy.stop_loss_bps_column else side for row in signals.iter_rows(named=True)}
    execute_event_plan(account, plan, targets, start, end, adapter, keep_states=False)
    completed = [row for row in account.episodes if row['closed_at_ns'] is not None]
    if not completed:
        return {'return': None, 'available': False, 'entry_ns': None if not account.episodes else account.episodes[0]['first_fill_ns'], 'exit_ns': None, 'label_available_ns': None, 'exit_reason': None}
    if len(completed) != 1:
        raise ValueError('A forward candidate must complete exactly one independent episode')
    episode = completed[0]
    net_return = finite_number(episode['net_pnl'], 'completed candidate PnL') / finite_number(episode['entry_equity'], 'candidate entry equity')
    return {'return': net_return, 'available': True, 'entry_ns': episode['first_fill_ns'], 'exit_ns': episode['closed_at_ns'], 'label_available_ns': episode['closed_at_ns'], 'exit_reason': episode['exit_reason']}


class TradeOutcomeTarget:
    requires_trade_context: ClassVar[bool] = True

    def __init__(self, train_data: pl.DataFrame, target_name: str, *, trade_context: TradeTargetContext, side: Literal['long', 'short'] = 'long', output: Literal['binary', 'net_return'] = 'binary', candidate_exposure: float = 1.0) -> None:
        super().__init__()
        if side not in ('long', 'short') or output not in ('binary', 'net_return'):
            raise ValueError('Trade outcome requires a long/short side and binary/net_return output')
        if not 0 < finite_number(candidate_exposure, 'candidate exposure') <= 1:
            raise ValueError('Candidate exposure must be in (0, 1]')
        if trade_context.policy.product.kind != 'linear_perpetual':
            raise ValueError('Both-side trade outcomes require a product supporting shorts')
        if trade_context.policy.max_holding_seconds is None and trade_context.policy.take_profit_bps is None and trade_context.policy.stop_loss_bps is None and trade_context.policy.take_profit_bps_column is None and trade_context.policy.stop_loss_bps_column is None:
            raise ValueError('Trade outcomes require a declared trade exit')
        self.target_name = target_name
        self.side = side
        self.output = output
        self.candidate_exposure = candidate_exposure
        self._outcomes: OutcomeLabels | None = None

    @property
    def outcomes(self) -> OutcomeLabels:
        if self._outcomes is None:
            raise ValueError('Trade outcomes have not been transformed')
        return self._outcomes

    def transform(self, data: pl.DataFrame, *, trade_context: TradeTargetContext, **transform_params: object) -> pl.DataFrame:
        if transform_params:
            raise ValueError(f'Unknown trade outcome transform params: {sorted(transform_params)}')
        inputs = trade_context.inputs
        if 'datetime' not in inputs.signals.columns:
            raise ValueError('Trade target context requires the bound model datetime identities')
        if (inputs.partition_start_ns, inputs.partition_end_ns) != (trade_context.partition_start_ns, trade_context.partition_end_ns):
            raise ValueError('Trade target context changed its true partition bounds')
        if contract_digest(export_trade_contract(trade_context.policy, inputs)) != trade_context.contract_digest:
            raise ValueError('Trade target context changed its bound execution contract')
        mapping = data.select('datetime').join(inputs.signals.select('datetime', 'row_id'), on='datetime', how='left', maintain_order='left')
        policy = replace(trade_context.policy, prediction_mode='target_exposure', notional_rate=1.0)
        plan = event_plan(inputs, policy)
        timeline = _Timeline(sorted({event.time_ns for event in plan.events if event.kind == 'observation'}), inputs.signals['available_at_ns'].to_list(), inputs.observations['start_ns'].to_list(), inputs.observations['available_at_ns'].to_list(), plan)
        positions = {identity: index for index, identity in enumerate(inputs.signals['row_id'])}
        results: list[dict[str, object]] = []
        for identity in mapping['row_id']:
            if identity is None:
                continue
            row: dict[str, object] = {'row_id': str(identity)}
            for side, signed in (('long', self.candidate_exposure), ('short', -self.candidate_exposure)):
                row.update({f'{side}_{key}': value for key, value in _candidate(inputs, policy, positions[identity], signed, timeline).items()})
            results.append(row)
        schema: dict[str, type[pl.DataType]] = {'row_id': pl.String}
        for side in ('long', 'short'):
            if policy.take_profit_bps_column or policy.stop_loss_bps_column:
                schema[f'{side}_unavailable_reason'] = pl.String
            schema.update({f'{side}_return': pl.Float64, f'{side}_available': pl.Boolean, f'{side}_entry_ns': pl.Int64, f'{side}_exit_ns': pl.Int64, f'{side}_label_available_ns': pl.Int64, f'{side}_exit_reason': pl.String})
        rows = pl.DataFrame(results, schema=schema)
        self._outcomes = OutcomeLabels(rows, trade_context.contract_digest)
        primary = mapping.join(rows.select('row_id', f'{self.side}_return'), on='row_id', how='left', maintain_order='left')[f'{self.side}_return']
        if self.output == 'binary':
            primary = (primary > 0).cast(pl.Int8)
        return data.with_columns(primary.rename(self.target_name))


__all__ = ['OutcomeLabels', 'TradeOutcomeTarget', 'TradeTargetContext']

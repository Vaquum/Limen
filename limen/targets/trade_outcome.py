from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Literal, cast

import polars as pl

from limen.backtest.trade_contract import NANOSECONDS, TradeInputs, TradePolicy, finite_number, source_binding
from limen.backtest.trade_execution import trade_execution


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


def _candidate(inputs: TradeInputs, policy: TradePolicy, row_id: str, side: float) -> dict[str, object]:
    selected = inputs.signals.filter(pl.col('row_id') == row_id)
    available = int(selected['available_at_ns'][0])
    end = inputs.partition_end_ns
    if policy.max_holding_seconds is not None:
        opens = inputs.observations.filter((pl.col('start_ns') != pl.col('end_ns')) & (pl.col('open_available_at_ns') == pl.col('start_ns')))['start_ns'].to_list()
        clocks = sorted(set([*opens, *inputs.observations['available_at_ns'].to_list()]))
        executable = [clock for clock in clocks if clock >= available + round(policy.execution_lag_seconds * NANOSECONDS)]
        if executable:
            deadline = executable[0] + round(policy.max_holding_seconds * NANOSECONDS)
            decisions = [int(clock) for clock in inputs.signals['available_at_ns'] if clock >= deadline]
            if policy.timer_interval_seconds is not None:
                interval, phase = round(policy.timer_interval_seconds * NANOSECONDS), round(policy.timer_phase_utc_seconds * NANOSECONDS)
                decisions.append(phase + -(-(deadline - phase) // interval) * interval)
            if decisions:
                exits = [clock for clock in clocks if clock >= min(decisions)]
                if exits:
                    end = min(end, exits[0])
    observations = inputs.observations.filter((pl.col('available_at_ns') >= available) & (pl.col('start_ns') <= end))
    start = max(inputs.partition_start_ns, int(observations['start_ns'][0]))
    signals = inputs.signals.filter(pl.col('available_at_ns').is_between(available, end)).with_columns(pl.lit(side).alias('target'))
    funding = inputs.funding_events
    if funding is not None:
        funding = funding.filter(pl.when(pl.col('kind') == 'accrual').then((pl.col('end_ns') >= start) & (pl.col('start_ns') <= end)).otherwise(pl.col('time_ns').is_between(start, end)))
    sources = [*inputs.sources, source_binding(observations, 'candidate:prices', start, end, 1, 'recorded_candidate_execution')]
    if funding is not None:
        sources.append(source_binding(funding, 'candidate:funding', start, end, 1, 'bound_candidate_funding'))
    candidate = replace(inputs, partition_start_ns=start, partition_end_ns=end, signals=signals, observations=observations, funding_events=funding, sources=tuple(sources))
    ledger = trade_execution(candidate, policy)
    completed = ledger.episodes.filter(pl.col('closed_at_ns').is_not_null())
    if not completed.height:
        return {'return': None, 'available': False, 'entry_ns': None if not ledger.episodes.height else ledger.episodes['first_fill_ns'][0], 'exit_ns': None, 'label_available_ns': None, 'exit_reason': None}
    if completed.height != 1:
        raise ValueError('A forward candidate must complete exactly one independent episode')
    episode = cast(dict[str, object], completed.row(0, named=True))
    net_return = finite_number(episode['net_pnl'], 'completed candidate PnL') / finite_number(episode['entry_equity'], 'candidate entry equity')
    return {'return': net_return, 'available': True, 'entry_ns': episode['first_fill_ns'], 'exit_ns': episode['closed_at_ns'], 'label_available_ns': episode['closed_at_ns'], 'exit_reason': episode['exit_reason']}


class TradeOutcomeTarget:
    requires_trade_context: ClassVar[bool] = True

    def __init__(self, train_data: pl.DataFrame, target_name: str, *, trade_context: TradeTargetContext, side: Literal['long', 'short'] = 'long', output: Literal['binary', 'net_return'] = 'binary', candidate_exposure: float = 1.0) -> None:
        if side not in ('long', 'short') or output not in ('binary', 'net_return'):
            raise ValueError('Trade outcome requires a long/short side and binary/net_return output')
        if not 0 < finite_number(candidate_exposure, 'candidate exposure') <= 1:
            raise ValueError('Candidate exposure must be in (0, 1]')
        if trade_context.policy.product.kind != 'linear_perpetual':
            raise ValueError('Both-side trade outcomes require a product supporting shorts')
        if trade_context.policy.max_holding_seconds is None and trade_context.policy.take_profit_bps is None and trade_context.policy.stop_loss_bps is None:
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
        mapping = data.select('datetime').join(inputs.signals.select('datetime', 'row_id'), on='datetime', how='left', maintain_order='left')
        policy = replace(trade_context.policy, prediction_mode='target_exposure', notional_rate=1.0)
        results: list[dict[str, object]] = []
        for identity in mapping['row_id']:
            if identity is None:
                continue
            row: dict[str, object] = {'row_id': str(identity)}
            for side, signed in (('long', self.candidate_exposure), ('short', -self.candidate_exposure)):
                row.update({f'{side}_{key}': value for key, value in _candidate(inputs, policy, str(identity), signed).items()})
            results.append(row)
        schema: dict[str, type[pl.DataType]] = {'row_id': pl.String}
        for side in ('long', 'short'):
            schema.update({f'{side}_return': pl.Float64, f'{side}_available': pl.Boolean, f'{side}_entry_ns': pl.Int64, f'{side}_exit_ns': pl.Int64, f'{side}_label_available_ns': pl.Int64, f'{side}_exit_reason': pl.String})
        rows = pl.DataFrame(results, schema=schema)
        self._outcomes = OutcomeLabels(rows, trade_context.contract_digest)
        primary = mapping.join(rows.select('row_id', f'{self.side}_return'), on='row_id', how='left', maintain_order='left')[f'{self.side}_return']
        if self.output == 'binary':
            primary = (primary > 0).cast(pl.Int8)
        return data.with_columns(primary.rename(self.target_name))


__all__ = ['OutcomeLabels', 'TradeOutcomeTarget', 'TradeTargetContext']

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import groupby
from typing import cast

import polars as pl

from limen.backtest.execution_events import execution_events
from limen.backtest.trade_contract import BPS, NANOSECONDS, ExecutionEvent, JsonValue, TradeInputs, TradeLedger, TradePolicy, contract_digest, export_trade_contract, finite_number

_STATE_SCHEMA = {'event_id': pl.String, 'time_ns': pl.Int64, 'cash': pl.Float64, 'accrued_funding': pl.Float64, 'quantity': pl.Float64, 'basis': pl.Float64, 'equity': pl.Float64, 'last_signal': pl.Float64, 'pending_intent_id': pl.String, 'episode_id': pl.String, 'forced_exit_latched': pl.Boolean, 'signal_sample': pl.Boolean}
_INTENT_SCHEMA = {'intent_id': pl.String, 'episode_id': pl.String, 'requested_at_ns': pl.Int64, 'target': pl.Float64, 'reason': pl.String, 'status': pl.String}
_FILL_SCHEMA = {'fill_id': pl.String, 'intent_id': pl.String, 'episode_id': pl.String, 'time_ns': pl.Int64, 'quantity_delta': pl.Float64, 'reference_price': pl.Float64, 'fill_price': pl.Float64, 'fee': pl.Float64, 'slippage': pl.Float64, 'evidence_row_id': pl.String}
_EPISODE_SCHEMA = {'episode_id': pl.String, 'side': pl.Int64, 'first_fill_ns': pl.Int64, 'anchor_price': pl.Float64, 'closed_at_ns': pl.Int64, 'net_pnl': pl.Float64, 'entry_equity': pl.Float64, 'exit_reason': pl.String}
_FUNDING_SCHEMA = {'event_id': pl.String, 'episode_id': pl.String, 'start_ns': pl.Int64, 'end_ns': pl.Int64, 'quantity': pl.Float64, 'rate_decimal': pl.Float64, 'valuation_price': pl.Float64, 'recognized_delta': pl.Float64, 'cash_delta': pl.Float64, 'currency': pl.String}


def _frame(rows: list[dict[str, JsonValue]], schema: dict[str, type[pl.DataType]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=schema)


def _canonical_signal(value: object, policy: TradePolicy) -> float | None:
    if value is None:
        return None
    signal = finite_number(value, 'target exposure')
    if abs(signal) > 1 or policy.prediction_mode == 'binary' and signal not in (0, 1):
        raise ValueError('Prediction does not match declared exposure mode/bounds')
    return 0.0 if abs(signal) <= policy.flat_threshold else signal


def _target_quantity(quantity: float, equity: float, price: float, fraction: float, policy: TradePolicy) -> float:
    if fraction == 0:
        return 0.0
    slip, fee = policy.slip_bps / BPS, policy.fee_bps / BPS
    candidates: list[float] = []
    for action in (1, -1):
        unit_cost = price * (slip + (1 + action * slip) * fee)
        denominator = price + fraction * unit_cost * action
        if denominator <= 0:
            raise ValueError('Post-cost sizing has no positive solution denominator')
        proposed = fraction * (equity + unit_cost * action * quantity) / denominator
        if (proposed - quantity) * action >= 0:
            candidates.append(proposed)
    if not candidates:
        raise ValueError('No affordable post-cost target quantity')
    step = policy.product.quantity_step
    units = math.floor(math.nextafter(abs(candidates[0]) / step, math.inf))
    return math.copysign(units * step, candidates[0])


@dataclass
class _Account:
    policy: TradePolicy
    cash: float
    prefix: str
    quantity: float = 0.0
    basis: float = 0.0
    accrued_funding: float = 0.0
    last_signal: float = 0.0
    latched: bool = False
    last_fill_ns: int | None = None
    episode: dict[str, JsonValue] | None = None
    pending: dict[str, JsonValue] | None = None
    ready_at_ns: int = 0
    states: list[dict[str, JsonValue]] = field(default_factory=list)
    intents: list[dict[str, JsonValue]] = field(default_factory=list)
    fills: list[dict[str, JsonValue]] = field(default_factory=list)
    episodes: list[dict[str, JsonValue]] = field(default_factory=list)
    funding: list[dict[str, JsonValue]] = field(default_factory=list)

    def equity(self, price: float) -> float:
        marked = self.quantity * price if self.policy.product.kind == 'cash_spot' else self.quantity * (price - self.basis)
        equity = self.cash + marked + self.accrued_funding
        required = self.policy.product.maintenance_margin_fraction * abs(self.quantity) * price
        if not math.isfinite(equity) or equity <= 0 or equity < required:
            raise ValueError(f'Nonpositive equity or maintenance margin breach: equity={equity}, quantity={self.quantity}, price={price}')
        return equity

    def intent(self, target: float, time_ns: int, reason: str) -> None:
        if self.pending is not None:
            self.pending['status'] = 'replaced' if reason == 'signal' else 'cancelled'
        item: dict[str, JsonValue] = {'intent_id': f'{self.prefix}:intent:{len(self.intents)}', 'episode_id': None if self.episode is None else self.episode['episode_id'], 'requested_at_ns': time_ns, 'target': target, 'reason': reason, 'status': 'pending'}
        self.intents.append(item)
        self.pending = item
        self.ready_at_ns = time_ns + (round(self.policy.execution_lag_seconds * NANOSECONDS) if reason == 'signal' else 0)

    def accept_signal(self, value: object, time_ns: int) -> None:
        signal = _canonical_signal(value, self.policy)
        if signal is None:
            return
        changed = abs(signal - self.last_signal) > self.policy.signal_change_bps / BPS
        changed = changed or (signal == 0 and self.last_signal != 0) or signal * self.last_signal < 0
        if signal == 0:
            self.latched = False
        if changed:
            self.last_signal = signal
            if not self.latched and (self.pending is None or self.pending['reason'] == 'signal'):
                fraction = self.policy.notional_rate * signal
                if abs(fraction) > self.policy.max_exposure:
                    raise ValueError('Target exceeds resolved maximum exposure')
                if fraction < 0 and self.policy.product.kind == 'cash_spot':
                    raise ValueError('Cash spot cannot open borrowed shorts')
                self.intent(fraction, time_ns, 'signal')

    def force_exit(self, time_ns: int, reason: str) -> None:
        if self.quantity != 0 and (self.pending is None or self.pending['reason'] == 'signal'):
            self.intent(0.0, time_ns, reason)
            self.latched = True

    def fill(self, new_quantity: float, price: float, event: ExecutionEvent) -> None:
        if self.pending is None:
            raise ValueError('A fill requires a pending intent')
        delta = new_quantity - self.quantity
        if delta == 0:
            self.pending['status'] = 'no_fill'
        else:
            fill_price = price * (1 + math.copysign(self.policy.slip_bps / BPS, delta))
            fee = abs(delta) * fill_price * self.policy.fee_bps / BPS
            slippage = delta * (fill_price - price)
            if new_quantity != 0 and abs(delta) * fill_price < self.policy.product.min_notional:
                self.pending['status'] = 'no_fill'
            else:
                before = self.equity(price)
                after = before - slippage - fee
                if after <= 0 or self.policy.product.initial_margin_fraction * abs(new_quantity) * price > after + math.ulp(before) * 8:
                    raise ValueError('Fill is unaffordable under collateral/cost rules')
                if self.quantity == 0:
                    episode: dict[str, JsonValue] = {'episode_id': f'{self.prefix}:episode:{len(self.episodes)}', 'side': 1 if new_quantity > 0 else -1, 'first_fill_ns': event.time_ns, 'anchor_price': fill_price, 'closed_at_ns': None, 'net_pnl': 0.0, 'entry_equity': before, 'exit_reason': None}
                    self.episode = episode
                    self.episodes.append(episode)
                if self.episode is None:
                    raise ValueError('Nonzero inventory has no episode')
                if self.policy.product.kind == 'cash_spot':
                    self.cash -= delta * fill_price + fee
                    if self.cash < -math.ulp(before) * 8:
                        raise ValueError('Cash spot fill would require borrowing')
                else:
                    reducing = self.quantity != 0 and delta * self.quantity < 0
                    realized = min(abs(delta), abs(self.quantity)) * math.copysign(1.0, self.quantity) * (fill_price - self.basis) if reducing else 0.0
                    self.cash += realized - fee
                if self.quantity == 0 or delta * self.quantity > 0:
                    self.basis = (abs(self.quantity) * self.basis + abs(delta) * fill_price) / abs(new_quantity)
                self.quantity = new_quantity
                self.last_fill_ns = event.time_ns
                self.pending['episode_id'] = self.episode['episode_id']
                self.fills.append({'fill_id': f'{self.prefix}:fill:{len(self.fills)}', 'intent_id': self.pending['intent_id'], 'episode_id': self.episode['episode_id'], 'time_ns': event.time_ns, 'quantity_delta': delta, 'reference_price': price, 'fill_price': fill_price, 'fee': fee, 'slippage': slippage, 'evidence_row_id': event.source_row_id})
                self.pending['status'] = 'filled'
                self.episode['net_pnl'] = self.equity(price) - cast(float, self.episode['entry_equity'])
                if new_quantity == 0:
                    self.episode['closed_at_ns'] = event.time_ns
                    self.episode['exit_reason'] = self.pending['reason']
                    self.episode = None
                    self.basis = 0.0

    def reconcile(self, price: float, event: ExecutionEvent) -> None:
        if self.pending is not None and event.time_ns >= self.ready_at_ns:
            target = cast(float, self.pending['target'])
            if target * self.quantity < 0:
                opposite = target
                self.fill(0.0, price, event)
                self.pending = None
                self.intent(opposite, event.time_ns, 'signal')
                self.ready_at_ns = event.time_ns
            quantity = _target_quantity(self.quantity, self.equity(price), price, cast(float, self.pending['target']), self.policy)
            self.fill(quantity, price, event)
            self.pending = None

    def record(self, time_ns: int, price: float, event_id: str, signal_sample: bool) -> None:
        equity = self.equity(price)
        if self.episode is not None:
            self.episode['net_pnl'] = equity - cast(float, self.episode['entry_equity'])
        self.states.append({'event_id': event_id, 'time_ns': time_ns, 'cash': self.cash, 'accrued_funding': self.accrued_funding, 'quantity': self.quantity, 'basis': self.basis, 'equity': equity, 'last_signal': self.last_signal, 'pending_intent_id': None if self.pending is None else self.pending['intent_id'], 'episode_id': None if self.episode is None else self.episode['episode_id'], 'forced_exit_latched': self.latched, 'signal_sample': signal_sample})


def _barrier_price(account: _Account, observation: dict[str, object], event: ExecutionEvent) -> tuple[float, str] | None:
    policy = account.policy
    if account.episode is None or policy.take_profit_bps is None and policy.stop_loss_bps is None:
        return None
    if event.observation_phase == 'close' and account.last_fill_ns is not None and account.last_fill_ns > int(cast(int, observation['start_ns'])):
        raise ValueError('Pre-entry/resize OHLC extrema require finer execution evidence')
    anchor = cast(float, account.episode['anchor_price'])
    side = math.copysign(1.0, account.quantity)
    opening = float(cast(float, observation['open']))
    high = float(cast(float, observation['high'])) if event.observation_phase == 'close' else opening
    low = float(cast(float, observation['low'])) if event.observation_phase == 'close' else opening
    stop = anchor * (1 - side * policy.stop_loss_bps / BPS) if policy.stop_loss_bps is not None else None
    take = anchor * (1 + side * policy.take_profit_bps / BPS) if policy.take_profit_bps is not None else None
    if stop is not None and side * (opening - stop) <= 0:
        return opening, 'stop_loss'
    if take is not None and side * (opening - take) >= 0:
        return take, 'take_profit'
    if stop is not None and (low <= stop if side > 0 else high >= stop):
        return stop, 'stop_loss'
    if take is not None and (high >= take if side > 0 else low <= take):
        return take, 'take_profit'
    return None


def _metrics(account: _Account, initial: float, states: pl.DataFrame) -> dict[str, float]:
    ending = float(states['equity'][-1])
    peak = initial
    drawdown = 0.0
    for equity in states['equity']:
        peak = max(peak, equity)
        drawdown = min(drawdown, equity / peak - 1)
    samples = states.filter(pl.col('signal_sample'))
    sample_exposures = [cast(float, row['absolute_exposure']) for row in account.states if row['signal_sample']]
    completed = [row for row in account.episodes if row['closed_at_ns'] is not None]
    fees = sum(cast(float, row['fee']) for row in account.fills)
    slip = sum(cast(float, row['slippage']) for row in account.fills)
    funding = sum(cast(float, row['recognized_delta']) for row in account.funding)
    return {'total_return': ending / initial - 1, 'ending_equity': ending, 'max_drawdown': drawdown, 'net_pnl': ending - initial, 'gross_pnl': ending - initial + fees + slip - funding, 'fees': fees, 'slippage': slip, 'funding_pnl': funding, 'completed_trades': float(len(completed)), 'open_trades': float(len(account.episodes) - len(completed)), 'episode_win_rate': sum(cast(float, row['net_pnl']) > 0 for row in completed) / len(completed) if completed else 0.0, 'avg_absolute_exposure': sum(sample_exposures) / samples.height if samples.height else 0.0}


def trade_execution(inputs: TradeInputs, policy: TradePolicy) -> TradeLedger:
    if policy.funding is not None:
        raise ValueError('Funding event preparation is required before funded execution')
    binding = contract_digest(export_trade_contract(policy, inputs))
    account = _Account(policy, inputs.initial_equity, binding[:16])
    observations = {str(row['row_id']): cast(dict[str, object], row) for row in inputs.observations.iter_rows(named=True)}
    signals = {str(row['row_id']): row['target'] for row in inputs.signals.iter_rows(named=True)}
    price = 0.0
    for time_ns, batch in groupby(execution_events(inputs, policy), key=lambda event: event.time_ns):
        group = list(batch)
        observed = [event for event in group if event.kind == 'observation']
        for event in observed:
            row = observations[str(event.source_row_id)]
            price = float(cast(float, row['open' if event.observation_phase == 'open' else 'close']))
            _ = account.equity(price)
            barrier = _barrier_price(account, row, event)
            if barrier is not None:
                reference, reason = barrier
                account.force_exit(time_ns, reason)
                account.reconcile(reference, event)
        declared = any(event.kind in ('signal', 'timer') and not event.event_id.startswith('end:') for event in group)
        if declared and account.episode is not None and policy.max_holding_seconds is not None and time_ns >= cast(int, account.episode['first_fill_ns']) + round(policy.max_holding_seconds * NANOSECONDS):
            account.force_exit(time_ns, 'time_stop')
        for event in group:
            if event.kind == 'signal':
                account.accept_signal(signals[str(event.source_row_id)], time_ns)
        if observed:
            account.reconcile(price, observed[-1])
        # No price at this event means the intent stays pending. Recording the
        # last causal mark does not make it an executable current price.
        account.record(time_ns, price, group[-1].event_id, any(event.kind == 'signal' for event in group))
        account.states[-1]['absolute_exposure'] = abs(account.quantity) * price / account.equity(price)
    schema = {**_STATE_SCHEMA, 'absolute_exposure': pl.Float64}
    states = _frame(account.states, schema)
    return TradeLedger(states, _frame(account.intents, _INTENT_SCHEMA), _frame(account.fills, _FILL_SCHEMA), _frame(account.episodes, _EPISODE_SCHEMA), _frame(account.funding, _FUNDING_SCHEMA), _metrics(account, inputs.initial_equity, states), binding)


__all__ = ['trade_execution']

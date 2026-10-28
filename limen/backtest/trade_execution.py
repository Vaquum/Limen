from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections.abc import Callable, Mapping
from typing import Literal, cast

import polars as pl

from limen.backtest._event_clock import clock_batches
from limen.backtest._event_plan import EventPlan, event_plan
from limen.backtest.funding_adapter import FundingAdapter, FundingContext, resolve_adapter
from limen.backtest.trade_contract import BPS, NANOSECONDS, ExecutionEvent, JsonValue, TradeInputs, TradeLedger, TradePolicy, contract_digest, export_trade_contract, finite_number

_STATE_SCHEMA = {'event_id': pl.String, 'time_ns': pl.Int64, 'cash': pl.Float64, 'accrued_funding': pl.Float64, 'quantity': pl.Float64, 'basis': pl.Float64, 'equity': pl.Float64, 'last_signal': pl.Float64, 'pending_intent_id': pl.String, 'episode_id': pl.String, 'forced_exit_latched': pl.Boolean, 'signal_sample': pl.Boolean}
_INTENT_SCHEMA = {'intent_id': pl.String, 'episode_id': pl.String, 'requested_at_ns': pl.Int64, 'target': pl.Float64, 'reason': pl.String, 'status': pl.String}
_FILL_SCHEMA = {'fill_id': pl.String, 'intent_id': pl.String, 'episode_id': pl.String, 'time_ns': pl.Int64, 'quantity_delta': pl.Float64, 'reference_price': pl.Float64, 'fill_price': pl.Float64, 'fee': pl.Float64, 'slippage': pl.Float64, 'evidence_row_id': pl.String}
_EPISODE_SCHEMA = {'episode_id': pl.String, 'side': pl.Int64, 'first_fill_ns': pl.Int64, 'anchor_price': pl.Float64, 'closed_at_ns': pl.Int64, 'net_pnl': pl.Float64, 'entry_equity': pl.Float64, 'exit_reason': pl.String}
_FUNDING_SCHEMA = {'event_id': pl.String, 'episode_id': pl.String, 'start_ns': pl.Int64, 'end_ns': pl.Int64, 'quantity': pl.Float64, 'rate_decimal': pl.Float64, 'valuation_price': pl.Float64, 'recognized_delta': pl.Float64, 'cash_delta': pl.Float64, 'currency': pl.String}


def _frame(rows: list[dict[str, JsonValue]], schema: Mapping[str, type[pl.DataType]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=schema)


def _canonical_signal(value: object, policy: TradePolicy) -> float | None:
    if value is None:
        return None
    signal = finite_number(value, 'target exposure')
    if abs(signal) > 1 or (policy.prediction_mode == 'binary' and signal not in (0, 1)):
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
        if denominator > 0:
            proposed = fraction * (equity + unit_cost * action * quantity) / denominator
            if (proposed - quantity) * action >= 0:
                candidates.append(proposed)
    if not candidates:
        raise ValueError('No affordable post-cost target quantity with a positive solution denominator')
    step = policy.product.quantity_step
    units = math.floor(math.nextafter(abs(candidates[0]) / step, math.inf))
    return math.copysign(units * step, candidates[0])


@dataclass
class Account:
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
    queued_signal: dict[str, JsonValue] | None = None
    ready_at_ns: int = 0
    states: list[dict[str, JsonValue]] = field(default_factory=list[dict[str, JsonValue]])
    state_blocks: list[pl.DataFrame] = field(default_factory=list[pl.DataFrame])
    intents: list[dict[str, JsonValue]] = field(default_factory=list[dict[str, JsonValue]])
    fills: list[dict[str, JsonValue]] = field(default_factory=list[dict[str, JsonValue]])
    episodes: list[dict[str, JsonValue]] = field(default_factory=list[dict[str, JsonValue]])
    funding: list[dict[str, JsonValue]] = field(default_factory=list[dict[str, JsonValue]])
    exit_armed: bool = True
    exit_closed_at_ns: int | None = None

    def equity(self, price: float) -> float:
        marked = self.quantity * price if self.policy.product.kind == 'cash_spot' else self.quantity * (price - self.basis)
        equity = self.cash + marked + self.accrued_funding
        required = self.policy.product.maintenance_margin_fraction * abs(self.quantity) * price
        if not math.isfinite(equity) or equity <= 0 or equity < required:
            raise ValueError(f'Nonpositive equity or maintenance margin breach: equity={equity}, quantity={self.quantity}, price={price}')
        return equity

    def intent(self, target: float, time_ns: int, reason: str, distances: Mapping[str, JsonValue] | None = None) -> None:
        if self.policy.exit_rules is not None:
            from limen.backtest._rule_exit_state import replace_pending
            if not replace_pending(self, reason):
                return
        item: dict[str, JsonValue] = {'intent_id': f'{self.prefix}:intent:{len(self.intents)}', 'episode_id': None if self.episode is None else self.episode['episode_id'], 'requested_at_ns': time_ns, 'target': target, 'reason': reason, 'status': 'pending'}
        if distances is not None:
            item.update(distances)
        self.intents.append(item)
        if reason == 'signal' and self.pending is not None and self.pending['reason'] != 'signal':
            if self.queued_signal is not None:
                self.queued_signal['status'] = 'replaced'
            self.queued_signal = item
            return
        if self.pending is not None:
            self.pending['status'] = 'replaced' if reason == 'signal' else 'cancelled'
        self.pending = item
        self.ready_at_ns = time_ns + (round(self.policy.execution_lag_seconds * NANOSECONDS) if reason in ('signal', 'rule_exit') else 0)

    def accept_signal(self, value: object, time_ns: int) -> None:
        row: Mapping[str, object] = cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}
        signal = _canonical_signal(row['target'] if isinstance(value, Mapping) else value, self.policy)
        explicit = self.policy.exit_rules is not None
        if explicit:
            from limen.backtest._rule_exit_state import accept
            if not accept(self, signal, row, time_ns):
                return
        if signal is None:
            return
        changed = explicit or abs(signal - self.last_signal) > self.policy.signal_change_bps / BPS
        changed = changed or (signal == 0 and self.last_signal != 0) or signal * self.last_signal < 0
        if signal == 0:
            self.latched = False
        if changed:
            self.last_signal = signal
            if not self.latched:
                fraction = self.policy.notional_rate * signal
                if abs(fraction) > self.policy.max_exposure:
                    raise ValueError('Target exceeds resolved maximum exposure')
                if fraction < 0 and self.policy.product.kind == 'cash_spot':
                    raise ValueError('Cash spot cannot open borrowed shorts')
                distances: dict[str, JsonValue] = {}
                if self.policy.take_profit_bps_column or self.policy.stop_loss_bps_column:
                    for name in ('take_profit_bps', 'stop_loss_bps'):
                        column = getattr(self.policy, f'{name}_column')
                        distances[name] = cast(JsonValue, row.get(column)) if column else getattr(self.policy, name)
                    entering = fraction != 0 and (self.quantity == 0 or fraction * self.quantity < 0)
                    if entering and row.get('__barrier_ready__') is not True:
                        from limen.experiment.errors import StrictModeError
                        raise StrictModeError(f'Barrier warmup at entry row={row.get("row_id")} time_ns={time_ns} columns={self.policy.take_profit_bps_column,self.policy.stop_loss_bps_column} settings={self.policy.barrier_spec}')
                self.intent(fraction, time_ns, 'signal', distances or None)

    def force_exit(self, time_ns: int, reason: str) -> None:
        if self.quantity != 0 and (self.policy.exit_rules is not None or self.pending is None or self.pending['reason'] == 'signal'):
            self.intent(0.0, time_ns, reason)
            self.latched = True
            self.exit_armed = False

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
                    if self.policy.take_profit_bps_column or self.policy.stop_loss_bps_column:
                        episode.update({name: self.pending.get(name) for name in ('take_profit_bps', 'stop_loss_bps')})
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
        if self.policy.exit_rules is not None and self.quantity == 0:
            self.exit_armed, self.exit_closed_at_ns = False, event.time_ns

    def reconcile(self, price: float, event: ExecutionEvent, *, exits_only: bool = False) -> None:
        while self.pending is not None and event.time_ns >= self.ready_at_ns:
            if exits_only and self.pending['reason'] == 'signal':
                break
            target = cast(float, self.pending['target'])
            if target * self.quantity < 0:
                opposite, distances = target, dict(self.pending)
                self.fill(0.0, price, event)
                self.pending = None
                self.intent(opposite, event.time_ns, 'signal', {name: distances[name] for name in ('take_profit_bps', 'stop_loss_bps') if name in distances})
                self.ready_at_ns = event.time_ns
            if self.pending is None:
                raise ValueError('Reconciliation lost its pending intent')
            quantity = _target_quantity(self.quantity, self.equity(price), price, cast(float, self.pending['target']), self.policy)
            self.fill(quantity, price, event)
            self.pending, self.queued_signal = self.queued_signal, None
            if self.pending is not None:
                self.ready_at_ns = cast(int, self.pending['requested_at_ns']) + round(self.policy.execution_lag_seconds * NANOSECONDS)

    def record(self, time_ns: int, price: float, event_id: str, signal_sample: bool) -> None:
        equity = self.equity(price)
        if self.episode is not None:
            self.episode['net_pnl'] = equity - cast(float, self.episode['entry_equity'])
        self.states.append({'event_id': event_id, 'time_ns': time_ns, 'cash': self.cash, 'accrued_funding': self.accrued_funding, 'quantity': self.quantity, 'basis': self.basis, 'equity': equity, 'last_signal': self.last_signal, 'pending_intent_id': None if self.pending is None else self.pending['intent_id'], 'episode_id': None if self.episode is None else self.episode['episode_id'], 'forced_exit_latched': self.latched, 'signal_sample': signal_sample})


def _barrier_price(account: Account, observation: dict[str, object], event: ExecutionEvent) -> tuple[float, str] | None:
    policy = account.policy
    take_bps = cast(float | None, account.episode.get('take_profit_bps', policy.take_profit_bps)) if account.episode else None
    stop_bps = cast(float | None, account.episode.get('stop_loss_bps', policy.stop_loss_bps)) if account.episode else None
    if account.episode is None or (take_bps is None and stop_bps is None) or (event.observation_phase == 'close' and cast(int, account.episode['first_fill_ns']) >= int(cast(int, observation['end_ns']))):
        return None
    if event.observation_phase == 'close' and account.last_fill_ns is not None and account.last_fill_ns > int(cast(int, observation['start_ns'])):
        raise ValueError('Pre-entry/resize OHLC extrema require finer execution evidence')
    anchor = cast(float, account.episode['anchor_price'])
    side = math.copysign(1.0, account.quantity)
    opening = float(cast(float, observation['open']))
    high = float(cast(float, observation['high'])) if event.observation_phase == 'close' else opening
    low = float(cast(float, observation['low'])) if event.observation_phase == 'close' else opening
    stop = anchor * (1 - side * stop_bps / BPS) if stop_bps is not None else None
    take = anchor * (1 + side * take_bps / BPS) if take_bps is not None else None
    if stop is not None and side * (opening - stop) <= 0:
        return opening, 'stop_loss'
    if take is not None and side * (opening - take) >= 0:
        return take, 'take_profit'
    if stop is not None and (low <= stop if side > 0 else high >= stop):
        return stop, 'stop_loss'
    if take is not None and (high >= take if side > 0 else low <= take):
        return take, 'take_profit'
    return None


def _metrics(account: Account, initial: float, states: pl.DataFrame) -> dict[str, float]:
    ending = float(states['equity'][-1])
    drawdown = float(states.select((pl.col('equity') / pl.max_horizontal(pl.col('equity').cum_max(), pl.lit(initial)) - 1).min()).item())
    samples = states.filter(pl.col('signal_sample'))
    sample_exposures = samples['absolute_exposure']
    completed = [row for row in account.episodes if row['closed_at_ns'] is not None]
    fees = sum(cast(float, row['fee']) for row in account.fills)
    slip = sum(cast(float, row['slippage']) for row in account.fills)
    funding = sum(cast(float, row['recognized_delta']) for row in account.funding)
    return {'total_return': ending / initial - 1, 'ending_equity': ending, 'max_drawdown': drawdown, 'net_pnl': ending - initial, 'gross_pnl': ending - initial + fees + slip - funding, 'fees': fees, 'slippage': slip, 'funding_pnl': funding, 'completed_trades': float(len(completed)), 'open_trades': float(len(account.episodes) - len(completed)), 'episode_win_rate': sum(cast(float, row['net_pnl']) > 0 for row in completed) / len(completed) if completed else 0.0, 'avg_absolute_exposure': sum(sample_exposures) / samples.height if samples.height else 0.0}


def _fund(account: Account, adapter: FundingAdapter, row: dict[str, object], start: int, end: int, price: float, kind: str) -> None:
    policy = account.policy.funding
    if policy is None:
        raise ValueError('Funding event requires a resolved policy')
    if kind == 'accrual' and policy.approximation == 'recorded_exact' and policy.params.get('history_interpretation') == 'integrated' and account.last_fill_ns is not None and cast(int, row['start_ns']) < account.last_fill_ns < cast(int, row['end_ns']):
        raise ValueError('Integrated funding with intrainterval inventory changes requires finer evidence or sampled integration')
    if row['valuation_price'] is None and kind != 'cash_settlement' and (policy.valuation != 'execution_proxy' or policy.approximation == 'recorded_exact'):
        raise ValueError('Funding requires its recorded valuation under the declared valuation policy')
    valuation = price if row['valuation_price'] is None else finite_number(row['valuation_price'], 'funding valuation')
    context = FundingContext(f'{row["event_id"]}:{start}:{end}', start, end, cast(Literal['payment', 'accrual', 'cash_settlement'], kind), account.accrued_funding, account.quantity, valuation, finite_number(row['rate_decimal'], 'funding rate'), finite_number(row['rate_basis_seconds'], 'rate basis'), policy.mechanism, policy.currency)
    flow = adapter(context, params=policy.params)
    if flow.event_id != context.event_id or flow.time_ns != end or flow.currency != policy.currency:
        raise ValueError('Funding adapter changed event/currency identity')
    recognized, cash = finite_number(flow.recognized_delta, 'funding recognition'), finite_number(flow.cash_delta, 'funding cash transfer')
    account.cash += cash
    account.accrued_funding += recognized - cash
    account.funding.append({'event_id': flow.event_id, 'episode_id': None if account.episode is None else account.episode['episode_id'], 'start_ns': start, 'end_ns': end, 'quantity': account.quantity, 'rate_decimal': context.rate_decimal, 'valuation_price': valuation, 'recognized_delta': recognized, 'cash_delta': cash, 'currency': flow.currency})


def _timer_block(account: Account, times: range, price: float, *, keep_states: bool) -> None:
    if not times:
        return
    if not keep_states:
        _ = account.equity(price)
        return
    schema = {**_STATE_SCHEMA, 'absolute_exposure': pl.Float64}
    if account.states:
        account.state_blocks.append(_frame(account.states, schema))
        account.states.clear()
    account.record(times.start, price, f'timer:{times.start}', False)
    row = account.states.pop()
    row['absolute_exposure'] = abs(account.quantity) * price / account.equity(price)
    block = pl.DataFrame({'time_ns': pl.int_range(times.start, times.stop, step=times.step, eager=True)}).with_columns(
        pl.format('timer:{}', pl.col('time_ns')).alias('event_id'),
        *(pl.lit(row[name], dtype=dtype).alias(name) for name, dtype in schema.items() if name not in ('time_ns', 'event_id')),
    ).select(*schema)
    account.state_blocks.append(block)


def _timer_span(account: Account, times: range, price: float, *, keep_states: bool) -> None:
    due = times.stop
    if account.episode is not None and account.policy.max_holding_seconds is not None:
        deadline = cast(int, account.episode['first_fill_ns']) + round(account.policy.max_holding_seconds * NANOSECONDS)
        due = max(times.start, times.start + -(-(deadline - times.start) // times.step) * times.step)
        due = min(due, times.stop)
    _timer_block(account, range(times.start, due, times.step), price, keep_states=keep_states)
    if due < times.stop:
        account.force_exit(due, 'time_stop')
        _timer_block(account, range(due, times.stop, times.step), price, keep_states=keep_states)


def execute_event_plan(
    account: Account, plan: EventPlan, signals: Mapping[str, object] | float,
    start: int, end: int, adapter: FundingAdapter | None,
    *, selected: Callable[[ExecutionEvent], bool] | None = None,
    stop_at_close: bool = False, keep_states: bool = True,
) -> None:
    policy = account.policy
    active: dict[str, object] | None = None
    previous = start
    price = 0.0
    for batch in clock_batches(plan, policy, start, end, selected):
        if isinstance(batch, range) and adapter is None:
            _timer_span(account, batch, price, keep_states=keep_states)
            previous = batch[-1]
            continue
        clocks = ((time, (ExecutionEvent(f'timer:{time}', time, 'timer', None, time),)) for time in batch) if isinstance(batch, range) else (batch,)
        for time_ns, group in clocks:
            active, price = _execute_batch(account, plan, signals, adapter, active, previous, price, time_ns, group)
            previous = time_ns
            if keep_states:
                account.record(time_ns, price, group[-1].event_id, any(event.kind == 'signal' for event in group))
                account.states[-1]['absolute_exposure'] = abs(account.quantity) * price / account.equity(price)
            else:
                _ = account.equity(price)
            if stop_at_close and account.episodes and account.episode is None:
                return


def _execute_batch(
    account: Account, plan: EventPlan, signals: Mapping[str, object] | float,
    adapter: FundingAdapter | None, active: dict[str, object] | None,
    previous: int, price: float, time_ns: int, group: tuple[ExecutionEvent, ...],
) -> tuple[dict[str, object] | None, float]:
    policy = account.policy
    if adapter is not None:
        if active is not None and time_ns > previous:
            _fund(account, adapter, active, previous, time_ns, price, 'accrual')
        for event in group:
            if event.kind == 'funding':
                row = plan.funding[str(event.source_row_id)]
                if row['kind'] == 'accrual':
                    if event.time_ns < cast(int, row['end_ns']):
                        active = row
                    elif active is row:
                        active = None
                else:
                    _fund(account, adapter, row, time_ns, time_ns, price, str(row['kind']))
    observed = [event for event in group if event.kind == 'observation']
    for event in observed:
        row = plan.observations[str(event.source_row_id)]
        price = float(cast(float, row['open' if event.observation_phase == 'open' else 'close']))
        barrier = _barrier_price(account, row, event)
        if barrier is not None:
            reference, reason = barrier
            account.force_exit(time_ns, reason)
            account.reconcile(reference, event, exits_only=True)
    declared = any(event.kind in ('signal', 'timer') and not event.event_id.startswith('end:') for event in group)
    if declared and account.episode is not None and policy.max_holding_seconds is not None and time_ns >= cast(int, account.episode['first_fill_ns']) + round(policy.max_holding_seconds * NANOSECONDS):
        account.force_exit(time_ns, 'time_stop')
    for event in group:
        if event.kind == 'signal':
            account.accept_signal(signals[str(event.source_row_id)] if isinstance(signals, Mapping) else signals, time_ns)
    if observed:
        account.reconcile(price, observed[-1])
    return active, price


def trade_execution(inputs: TradeInputs, policy: TradePolicy) -> TradeLedger:
    adapter = resolve_adapter(policy.funding) if policy.funding is not None else None
    if (adapter is not None) != (inputs.funding_events is not None):
        raise ValueError('Funding policy and normalized events must be supplied together')
    plan = event_plan(inputs, policy)
    binding = contract_digest(export_trade_contract(policy, inputs)) if policy.exit_rules is not None else plan.binding
    account = Account(policy, inputs.initial_equity, binding[:16])
    signals = {str(row['row_id']): row if policy.take_profit_bps_column or policy.stop_loss_bps_column or policy.exit_rules is not None else row['target'] for row in inputs.signals.iter_rows(named=True)}
    execute_event_plan(account, plan, signals, inputs.partition_start_ns, inputs.partition_end_ns, adapter)
    schema = {**_STATE_SCHEMA, 'absolute_exposure': pl.Float64}
    states = pl.concat([*account.state_blocks, _frame(account.states, schema)]) if account.state_blocks else _frame(account.states, schema)
    prior = pl.col('equity').shift(1).fill_null(inputs.initial_equity)
    states = states.with_columns((pl.col('equity') / prior - 1).alias('net_return'))
    extras = {'take_profit_bps': pl.Float64, 'stop_loss_bps': pl.Float64} if policy.take_profit_bps_column or policy.stop_loss_bps_column else {}
    return TradeLedger(states, _frame(account.intents, {**_INTENT_SCHEMA, **extras}), _frame(account.fills, _FILL_SCHEMA), _frame(account.episodes, {**_EPISODE_SCHEMA, **extras}), _frame(account.funding, _FUNDING_SCHEMA), _metrics(account, inputs.initial_equity, states), binding)


__all__ = ['trade_execution']

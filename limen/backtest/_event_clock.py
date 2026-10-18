from bisect import bisect_left, bisect_right
from collections.abc import Callable, Iterator
from itertools import groupby

from limen.backtest._event_plan import EventPlan
from limen.backtest.execution_events import timer_clock
from limen.backtest.trade_contract import ExecutionEvent, TradePolicy


def clock_batches(
    plan: EventPlan, policy: TradePolicy, start: int, end: int,
    selected: Callable[[ExecutionEvent], bool] | None = None,
) -> Iterator[tuple[int, tuple[ExecutionEvent, ...]] | range]:
    interval, phase = timer_clock(policy)
    next_timer = phase + -(-(start - phase) // interval) * interval if interval else end + 1
    first, last = bisect_left(plan.times, start), bisect_right(plan.times, end)
    events = (plan.events[index] for index in range(first, last))
    events = (event for event in events if not event.event_id.startswith('end:') and (selected is None or selected(event)))
    terminal = ExecutionEvent(f'end:{end}', end, 'timer', None, end)
    for time_ns, batch in groupby(_with_terminal(events, terminal), key=lambda event: event.time_ns):
        if interval and next_timer < time_ns:
            stop = next_timer + -(-(time_ns - next_timer) // interval) * interval
            yield range(next_timer, stop, interval)
            next_timer = stop
        group = tuple(batch)
        if interval and next_timer == time_ns:
            timer = ExecutionEvent(f'timer:{time_ns}', time_ns, 'timer', None, time_ns)
            # Source clocks already have the canonical funding/price/timer/
            # signal order; the terminal ID sorts before a scheduled timer ID.
            split = next((index for index, event in enumerate(group) if event.kind == 'signal'), len(group))
            group = (*group[:split], timer, *group[split:])
            next_timer += interval
        yield time_ns, group


def _with_terminal(events: Iterator[ExecutionEvent], terminal: ExecutionEvent) -> Iterator[ExecutionEvent]:
    for event in events:
        if event.time_ns == terminal.time_ns and event.kind == 'signal':
            yield terminal
            yield event
            yield from events
            return
        yield event
    yield terminal


__all__: list[str] = []

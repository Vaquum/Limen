from collections.abc import Mapping
from typing import Protocol

from limen.backtest.trade_contract import JsonValue


class ExitAccount(Protocol):
    quantity: float
    last_signal: float
    latched: bool
    exit_armed: bool
    exit_closed_at_ns: int | None
    pending: dict[str, JsonValue] | None
    queued_signal: dict[str, JsonValue] | None

    def intent(self, target: float, time_ns: int, reason: str, distances: Mapping[str, JsonValue] | None = None) -> None: ...


def accept(account: ExitAccount, signal: float | None, row: Mapping[str, object], moment: int) -> bool:
    if signal is None:
        raise ValueError('Explicit exits require nonmissing binary entry predictions')
    account.last_signal = signal
    exiting = row.get('exit')
    if type(exiting) is not bool:
        raise ValueError('Explicit exits require nonmissing boolean exit values')
    if account.pending is not None:
        if account.quantity == 0 and account.pending['reason'] == 'signal' and exiting:
            account.pending['status'] = 'cancelled'
            account.pending, account.queued_signal = None, None
            account.exit_armed, account.latched, account.exit_closed_at_ns = False, True, moment
        elif account.quantity != 0 and exiting:
            account.intent(0.0, moment, 'rule_exit')
            account.exit_armed, account.latched = False, True
        return False
    if account.quantity != 0:
        if exiting:
            account.intent(0.0, moment, 'rule_exit')
            account.exit_armed, account.latched = False, True
        return False
    if signal == 0 and (account.exit_closed_at_ns is None or moment > account.exit_closed_at_ns):
        account.exit_armed, account.latched = True, False
    if account.exit_armed and signal == 1 and not exiting:
        account.exit_armed, account.latched = False, False
        return True
    return False


_PRIORITY = {'signal': 0, 'rule_exit': 1, 'time_stop': 2, 'stop_loss': 3, 'take_profit': 3}


def replace_pending(account: ExitAccount, reason: str) -> bool:
    if account.pending is not None:
        previous = str(account.pending['reason'])
        if _PRIORITY[reason] <= _PRIORITY[previous]:
            return False
        account.pending['status'] = 'cancelled' if previous == 'signal' else 'replaced'
        account.pending, account.queued_signal = None, None
    return True


__all__: list[str] = []

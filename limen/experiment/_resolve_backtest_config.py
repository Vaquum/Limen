import math
import numbers
from collections.abc import Mapping
from typing import Protocol

from limen.backtest._long_flat_tp_sl import _validate_barrier

BACKTEST_KEYS = ('fee_bps', 'slip_bps', 'notional_rate', 'take_profit_bps', 'stop_loss_bps')


class _BacktestConfig(Protocol):
    fee_bps: float | str
    slip_bps: float | str
    notional_rate: float | str
    take_profit_bps: float | str | None
    stop_loss_bps: float | str | None


def _configured_barriers(config: _BacktestConfig | None) -> bool:
    return config is not None and (config.take_profit_bps is not None or config.stop_loss_bps is not None)


def _resolve_backtest_config(config: _BacktestConfig | None, params: Mapping[str, object]) -> dict[str, float | None]:
    if config is None:
        return {}
    resolved: dict[str, float | None] = {}
    raw = {'fee_bps': config.fee_bps, 'slip_bps': config.slip_bps,
           'notional_rate': config.notional_rate, 'take_profit_bps': config.take_profit_bps,
           'stop_loss_bps': config.stop_loss_bps}
    for name in BACKTEST_KEYS:
        value: object = raw[name]
        if isinstance(value, str):
            ref = value[1:-1] if value.startswith('{') and value.endswith('}') else value
            if ref not in params:
                raise ValueError(f"Manifest backtest {name} references unknown search-param '{ref}'; add it to params() or pass a number")
            value = params[ref]
        resolved[name] = _validate_backtest_value(name, value)
    return resolved


def _validate_backtest_value(name: str, value: object) -> float | None:
    if name in ('take_profit_bps', 'stop_loss_bps'):
        return _validate_barrier(name, value)
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or (not 0 < value <= 1 if name == 'notional_rate' else value < 0):
        bound = 'in (0, 1]' if name == 'notional_rate' else 'a non-negative finite number'
        raise ValueError(f'Manifest backtest {name} must be {bound}, got {value!r}')
    return float(value)

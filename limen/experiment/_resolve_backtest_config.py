import math
import numbers
from collections.abc import Mapping
from typing import Protocol

from limen.backtest._long_flat_tp_sl import validate_barrier as _validate_barrier

BACKTEST_KEYS = ('fee_bps', 'slip_bps', 'notional_rate', 'take_profit_bps', 'stop_loss_bps')


class BacktestConfig(Protocol):
    fee_bps: float | str
    slip_bps: float | str
    notional_rate: float | str
    take_profit_bps: float | str | None
    stop_loss_bps: float | str | None


def configured_barriers(config: BacktestConfig | None) -> bool:
    return config is not None and (config.take_profit_bps is not None or config.stop_loss_bps is not None)


def resolve_backtest_config(config: BacktestConfig | None, params: Mapping[str, object]) -> dict[str, float | None]:
    if config is None:
        return {}
    resolved: dict[str, float | None] = {}
    raw = {'fee_bps': config.fee_bps, 'slip_bps': config.slip_bps,
           'notional_rate': config.notional_rate, 'take_profit_bps': config.take_profit_bps,
           'stop_loss_bps': config.stop_loss_bps}
    for name in BACKTEST_KEYS:
        value: object = raw[name]
        if isinstance(value, str):
            value = value.strip()
            ref = value[1:-1] if value.startswith('{') and value.endswith('}') else value
            if ref not in params:
                raise ValueError(f"Manifest backtest {name} references unknown search-param '{ref}'; add it to params() or pass a number")
            value = params[ref]
        resolved[name] = validate_backtest_value(name, value)
    return resolved


def validate_backtest_value(name: str, value: object) -> float | None:
    if name in ('take_profit_bps', 'stop_loss_bps'):
        return _validate_barrier(name, value)
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or (not 0 < float(value) <= 1 if name == 'notional_rate' else float(value) < 0):
        bound = 'in (0, 1]' if name == 'notional_rate' else 'a non-negative finite number'
        raise ValueError(f'Manifest backtest {name} must be {bound}, got {value!r}')
    return float(value)


__all__ = ['BACKTEST_KEYS', 'BacktestConfig', 'configured_barriers', 'resolve_backtest_config', 'validate_backtest_value']

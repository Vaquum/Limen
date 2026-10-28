import numbers

import numpy as np
import numpy.typing as npt

from limen.backtest._long_flat_tp_sl import _level
from limen.backtest.long_flat_strategy import ExecutionResult, long_flat_strategy

__all__ = ['explicit_exit_strategy']


def explicit_exit_strategy(
    predictions: npt.ArrayLike, exits: npt.ArrayLike,
    open_px: npt.NDArray[np.float64], close_px: npt.NDArray[np.float64],
    price_change: npt.NDArray[np.float64], *,
    high_px: npt.NDArray[np.float64] | None = None,
    low_px: npt.NDArray[np.float64] | None = None,
    take_profit_bps: float | None = None, stop_loss_bps: float | None = None,
    execution_lag_bars: int = 1, fee_bps: float = 5.0, slip_bps: float = 5.0,
) -> ExecutionResult:
    if isinstance(execution_lag_bars, bool) or not isinstance(execution_lag_bars, numbers.Integral) or execution_lag_bars <= 0:
        raise ValueError('Explicit exits require a positive integer execution_lag_bars')
    pred, exit_values = np.asarray(predictions, dtype=float), np.asarray(exits)
    if pred.ndim != 1 or pred.size != close_px.size or not np.isfinite(pred).all() or not np.isin(pred, (0, 1)).all():
        raise ValueError('Explicit exits require equal-length binary predictions')
    if exit_values.ndim != 1 or exit_values.size != pred.size or exit_values.dtype.kind != 'b':
        raise ValueError('Explicit exits require equal-length nonmissing boolean exit values')
    held = np.zeros(pred.size)
    effective_close = close_px.copy()
    armed, active, closed_at = True, False, -1
    entry_ready: int | None = None
    rule_close: int | None = None
    take: float | None = None
    stop: float | None = None
    for row in range(pred.size):
        if entry_ready == row:
            active, entry_ready = True, None
            take = _level(float(close_px[row - 1]), take_profit_bps, 'take_profit_bps')
            stop = _level(float(close_px[row - 1]), stop_loss_bps, 'stop_loss_bps')
        if active:
            held[row] = 1
            price = _barrier_price(float(open_px[row]), high_px, low_px, row, take, stop)
            if price is not None:
                effective_close[row] = price
                active, armed, closed_at, rule_close = False, False, row, None
        if entry_ready is not None:
            if exit_values[row]:
                entry_ready, armed, closed_at = None, False, row
        elif active:
            if exit_values[row] and rule_close is None:
                rule_close = row + execution_lag_bars - 1
        else:
            if pred[row] == 0 and row > closed_at:
                armed = True
            if armed and pred[row] == 1 and not exit_values[row]:
                entry_ready, armed = row + execution_lag_bars, False
        if active and rule_close == row:
            active, armed, closed_at, rule_close = False, False, row, None
    effective_pred = np.zeros(pred.size)
    if execution_lag_bars < pred.size:
        effective_pred[:-execution_lag_bars] = held[execution_lag_bars:]
    return long_flat_strategy(
        effective_pred, open_px, effective_close, price_change + effective_close - close_px,
        execution_lag_bars=execution_lag_bars, fee_bps=fee_bps, slip_bps=slip_bps,
    )


def _barrier_price(
    opening: float, high: npt.NDArray[np.float64] | None,
    low: npt.NDArray[np.float64] | None, row: int,
    take: float | None, stop: float | None,
) -> float | None:
    if stop is not None and opening <= stop:
        return opening
    if take is not None and opening >= take:
        return take
    if stop is not None and low is not None and low[row] <= stop:
        return stop
    if take is not None and high is not None and high[row] >= take:
        return take
    return None

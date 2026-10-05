import math
import numbers

import numpy as np
import numpy.typing as npt

from limen.backtest.long_flat_strategy import ExecutionResult, long_flat_strategy

BPS_PER_UNIT = 10_000.0


def validate_barrier(name: str, value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or float(value) <= 0 or (name == 'stop_loss_bps' and float(value) >= BPS_PER_UNIT):
        raise ValueError(f'{name} must be finite and positive' + (' and below 10000' if name == 'stop_loss_bps' else '') + f', got {value!r}')
    return float(value)


def _level(entry: float, bps: float | None, name: str) -> float | None:
    if bps is None:
        return None
    level = entry * (1 + bps / BPS_PER_UNIT) if name == 'take_profit_bps' else entry * (1 - bps / BPS_PER_UNIT)
    if not math.isfinite(level) or level <= 0:
        raise ValueError(f'{name} computed level must be finite and positive')
    return level


def long_flat_tp_sl(
    predictions: npt.ArrayLike, open_px: npt.NDArray[np.float64],
    close_px: npt.NDArray[np.float64], price_change: npt.NDArray[np.float64], *,
    high_px: npt.NDArray[np.float64], low_px: npt.NDArray[np.float64],
    take_profit_bps: float | None, stop_loss_bps: float | None,
    execution_lag_bars: int = 1, fee_bps: float = 5.0, slip_bps: float = 5.0,
) -> ExecutionResult:
    pred = np.asarray(predictions, dtype=float)
    if pred.ndim != 1 or pred.size != close_px.size or not np.isfinite(pred).all() or not np.isin(pred, (0, 1)).all():
        raise ValueError('long_flat_strategy predictions must contain only 0 or 1 in an equal-length 1D array')
    lagged = np.zeros(pred.size)
    if execution_lag_bars < pred.size:
        lagged[execution_lag_bars:] = pred[:-execution_lag_bars]
    held = np.zeros(pred.size)
    effective_close = close_px.copy()
    armed, active = True, False
    tp = sl = None
    for row in range(execution_lag_bars, pred.size):
        if lagged[row] == 0:
            armed, active = True, False
        elif armed:
            if not active:
                entry = float(close_px[row - 1])
                tp = _level(entry, take_profit_bps, 'take_profit_bps')
                sl = _level(entry, stop_loss_bps, 'stop_loss_bps')
                active = True
            held[row] = 1
            exit_price = None
            if sl is not None and open_px[row] <= sl:
                exit_price = float(open_px[row])
            elif tp is not None and open_px[row] >= tp:
                exit_price = tp
            elif sl is not None and low_px[row] <= sl:
                exit_price = sl
            elif tp is not None and high_px[row] >= tp:
                exit_price = tp
            if exit_price is not None:
                effective_close[row] = exit_price
                armed, active = False, False
    effective_pred = np.zeros(pred.size)
    if execution_lag_bars < pred.size:
        effective_pred[:-execution_lag_bars] = held[execution_lag_bars:]
    return long_flat_strategy(
        effective_pred, open_px, effective_close,
        price_change + effective_close - close_px,
        execution_lag_bars=execution_lag_bars, fee_bps=fee_bps, slip_bps=slip_bps,
    )


__all__ = ['long_flat_tp_sl', 'validate_barrier']

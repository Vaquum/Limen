from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

BPS_PER_UNIT = 10_000.0

__all__ = ['ExecutionResult', 'long_flat_strategy']


class ExecutionResult(NamedTuple):

    '''
    Per-bar execution arrays consumed by the snapshot ledger.

    Attributes:
        pos (npt.NDArray[np.float64]): Position held per bar (0 or 1 under the
            all-in model; a strategy may return a deployed fraction).
        gross (npt.NDArray[np.float64]): Per-bar gross return before costs.
        net (npt.NDArray[np.float64]): Per-bar net return after entry and exit costs.
    '''

    pos: npt.NDArray[np.float64]
    gross: npt.NDArray[np.float64]
    net: npt.NDArray[np.float64]


def _shift(arr: npt.NDArray[Any], periods: int, fill: Any) -> npt.NDArray[Any]:

    '''Positional shift matching pandas Series.shift(periods, fill_value=fill).'''

    n = arr.shape[0]
    out = np.empty(n, dtype=arr.dtype)
    out[:] = fill
    if periods > 0:
        if periods < n:
            out[periods:] = arr[:n - periods]
    elif periods < 0:
        gap = -periods
        if gap < n:
            out[:n - gap] = arr[gap:]
    else:
        out[:] = arr
    return out


def long_flat_strategy(predictions: Any,
                       open_px: Any,
                       close_px: Any,
                       price_change: Any,
                       *,
                       execution_lag_bars: int = 1,
                       fee_bps: float = 5.0,
                       slip_bps: float = 5.0) -> ExecutionResult:

    '''
    Long-only, hold-while-1 execution over pre-aligned intrabar returns.

    Interprets a binary 0/1 signal as an all-in long position: enter at the open
    of the first signalled bar, ride close-to-close while the signal persists, and
    exit at the close of the last signalled bar. Fee and slippage are applied
    multiplicatively on the entry and exit fills.

    Predictions are shifted forward by execution_lag_bars onto the execution rows.
    The entry-bar gross return is price_change / open; the continuation-bar gross
    return is close_t / close_{t-1} - 1; a flat bar is a real 0.

    Args:
        predictions (Any): Per-bar signal (array-like); must contain only 0 or 1
        open_px (Any): Bar open price (array-like)
        close_px (Any): Bar close price (array-like)
        price_change (Any): Bar close minus open (array-like)
        execution_lag_bars (int): Bars between a signal row and its execution row
        fee_bps (float): Per-fill fee in basis points
        slip_bps (float): Per-fill slippage in basis points

    Returns:
        ExecutionResult: Per-bar pos, gross, and net return arrays
    '''

    if execution_lag_bars < 0:
        raise ValueError('long_flat_strategy execution_lag_bars must be >= 0')

    try:
        pred = np.asarray(predictions).astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError('long_flat_strategy predictions must contain only 0 or 1') from exc

    if np.isnan(pred).any() or not np.isin(pred, (0.0, 1.0)).all():
        raise ValueError('long_flat_strategy predictions must contain only 0 or 1')

    pred = pred.astype(int)
    total_bars = pred.shape[0]

    open_a = np.asarray(open_px, dtype=float)
    close_a = np.asarray(close_px, dtype=float)
    dpx = np.asarray(price_change, dtype=float)

    tradable = ~np.isnan(open_a) & ~np.isnan(close_a) & ~np.isnan(dpx) & (open_a != 0)
    execution_rows = np.zeros(total_bars, dtype=bool)
    if execution_lag_bars < total_bars:
        execution_rows[execution_lag_bars:] = True

    pred = _shift(pred, execution_lag_bars, 0)
    eval_mask = execution_rows & tradable
    pos = (pred == 1) & eval_mask

    entry_mask = pos & ~_shift(pos, 1, False)
    cont_mask = pos & _shift(pos, 1, False)

    with np.errstate(divide='ignore', invalid='ignore'):
        r_entry = dpx / open_a
        r_cont = (close_a / _shift(close_a, 1, np.nan)) - 1.0

    gross = np.where(entry_mask, r_entry, 0.0) + np.where(cont_mask, r_cont, 0.0)
    gross = np.where(np.isnan(gross), 0.0, gross)

    fee = fee_bps / BPS_PER_UNIT
    slip = slip_bps / BPS_PER_UNIT
    entry_mult = (1.0 - fee) / (1.0 + slip)
    exit_mult = (1.0 - fee) * (1.0 - slip)

    exit_mask = pos & ~_shift(pos, -1, False)
    cost_mult = np.ones(total_bars, dtype=float)
    cost_mult[entry_mask] *= entry_mult
    cost_mult[exit_mask] *= exit_mult

    net = ((1.0 + gross) * cost_mult) - 1.0
    net = np.where(np.isnan(net), 0.0, net)

    return ExecutionResult(pos=pos.astype(float), gross=gross, net=net)

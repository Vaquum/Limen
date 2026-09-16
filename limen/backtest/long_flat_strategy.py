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

    out = np.full(arr.shape[0], fill, dtype=arr.dtype)
    if periods > 0:
        out[periods:] = arr[:-periods]
    elif periods < 0:
        out[:periods] = arr[-periods:]
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
    Long-only, hold-while-1 execution over pre-aligned close-to-close returns.

    A binary 0/1 signal, shifted forward by execution_lag_bars, holds an all-in long
    position from the prior close to the close of the last signalled row, earning
    close_t / close_{t-1} - 1 per held bar and a real 0 when flat; a row whose prior
    close is missing or zero is non-tradable. Slippage adjusts the fill prices; fee_bps
    of the entry notional is paid from cash at entry and fee_bps of the exit proceeds
    at exit, so each bar's net is the return on equity, the position less the entry fee.

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

    arrays = (pred, open_a, close_a, dpx)
    if any(arr.ndim != 1 for arr in arrays) or len({arr.shape[0] for arr in arrays}) != 1:
        raise ValueError('long_flat_strategy inputs must be equal-length 1D arrays')

    prev_close = _shift(close_a, 1, np.nan)
    tradable = (
        ~np.isnan(open_a) & ~np.isnan(close_a) & ~np.isnan(dpx)
        & ~np.isnan(prev_close) & (prev_close != 0)
    )
    execution_rows = np.zeros(total_bars, dtype=bool)
    if execution_lag_bars < total_bars:
        execution_rows[execution_lag_bars:] = True

    pred = _shift(pred, execution_lag_bars, 0)
    eval_mask = execution_rows & tradable
    pos = (pred == 1) & eval_mask

    prev_pos = _shift(pos, 1, False)
    entry_mask = pos & ~prev_pos

    with np.errstate(divide='ignore', invalid='ignore'):
        r_cont = (close_a / prev_close) - 1.0

    gross = np.where(pos, r_cont, 0.0)
    gross = np.where(np.isnan(gross), 0.0, gross)

    fee = fee_bps / BPS_PER_UNIT
    slip = slip_bps / BPS_PER_UNIT
    exit_mask = pos & ~_shift(pos, -1, False)

    factor = np.where(pos, 1.0 + gross, 1.0)
    factor[entry_mask] /= 1.0 + slip
    cumulative = np.cumprod(factor)
    segment_start = np.maximum.accumulate(np.where(entry_mask, np.arange(total_bars), 0))
    position = cumulative / _shift(cumulative, 1, 1.0)[segment_start]
    equity = np.where(pos, position - fee, 1.0)
    equity[exit_mask] = position[exit_mask] * (1.0 - fee) * (1.0 - slip) - fee
    previous_equity = np.where(entry_mask, 1.0, _shift(equity, 1, 1.0))

    net = np.where(pos, equity / previous_equity - 1.0, 0.0)

    return ExecutionResult(pos=pos.astype(float), gross=gross, net=net)

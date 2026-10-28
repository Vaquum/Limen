import numbers

import numpy as np
import numpy.typing as npt

from limen.backtest.long_flat_strategy import ExecutionResult

__all__ = ['short_flat_strategy']

_BPS = 10_000.0


def short_flat_strategy(
    predictions: npt.ArrayLike,
    open_px: npt.ArrayLike,
    close_px: npt.ArrayLike,
    price_change: npt.ArrayLike,
    *,
    execution_lag_bars: int = 1,
    fee_bps: float = 5.0,
    slip_bps: float = 5.0,
) -> ExecutionResult:
    '''Evaluate binary short/flat signals with constant-notional -1x returns.

    Entries use the execution-row open; continuations use the preceding close.
    Entry-sell and exit-cover costs debit positive fill notionals.
    Missing prices flatten the position; a later valid signal starts a new trade.
    This approximation excludes borrowing, margin, liquidation and funding.
    '''
    if isinstance(execution_lag_bars, bool) or not isinstance(execution_lag_bars, numbers.Integral) or execution_lag_bars < 0:
        raise ValueError('short_flat_strategy execution_lag_bars must be >= 0 and an integer')
    for name, value in (('fee_bps', fee_bps), ('slip_bps', slip_bps)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value) or not 0 <= value < _BPS:
            raise ValueError(f'short_flat_strategy {name} must be finite and in [0, 10000)')
    try:
        pred = np.asarray(predictions, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('short_flat_strategy predictions must contain only 0 or 1') from exc
    if not np.isin(pred, (0.0, 1.0)).all():
        raise ValueError('short_flat_strategy predictions must contain only 0 or 1')
    try:
        prices = tuple(np.asarray(value, dtype=float) for value in (open_px, close_px, price_change))
    except (TypeError, ValueError) as exc:
        raise ValueError('short_flat_strategy prices must be numeric') from exc
    if any(arr.ndim != 1 for arr in (pred, *prices)) or len({arr.size for arr in (pred, *prices)}) != 1:
        raise ValueError('short_flat_strategy inputs must be equal-length 1D arrays')
    if not pred.size:
        raise ValueError('short_flat_strategy requires at least one row')
    open_a, close_a, dpx = prices
    if any(np.isinf(arr).any() for arr in prices) or any((arr[np.isfinite(arr)] <= 0).any() for arr in (open_a, close_a)):
        raise ValueError('short_flat_strategy prices must be positive or missing, never infinite')
    tradable = np.isfinite(open_a) & np.isfinite(close_a) & np.isfinite(dpx)
    if not np.isclose(dpx[tradable], close_a[tradable] - open_a[tradable], rtol=1e-9, atol=1e-12).all():
        raise ValueError('short_flat_strategy price_change must equal close - open')
    signal = np.zeros(pred.size, dtype=float)
    if execution_lag_bars < pred.size:
        signal[execution_lag_bars:] = pred[:pred.size - execution_lag_bars]
    pos = (signal == 1) & tradable
    entry = pos & ~np.concatenate(([False], pos[:-1]))
    continuation = pos & ~entry
    exit_mask = pos & ~np.concatenate((pos[1:], [False]))
    gross = np.zeros(pred.size, dtype=float)
    gross[entry] = -dpx[entry] / open_a[entry]
    previous_close = np.concatenate(([np.nan], close_a[:-1]))
    gross[continuation] = 1.0 - close_a[continuation] / previous_close[continuation]
    net = gross.copy()
    entry_sell = (1.0 - fee_bps / _BPS) * (1.0 - slip_bps / _BPS)
    exit_cover = (1.0 - fee_bps / _BPS) / (1.0 + slip_bps / _BPS)
    net[entry] -= 1.0 - entry_sell
    cover_notional = (1.0 - gross[exit_mask]) * np.where(entry[exit_mask], entry_sell, 1.0)
    net[exit_mask] -= cover_notional * (1.0 - exit_cover)
    if not np.isfinite(gross).all() or not np.isfinite(net).all():
        raise ValueError('short_flat_strategy returns must be finite')
    return ExecutionResult(pos.astype(float), gross, net)

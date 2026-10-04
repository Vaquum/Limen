from limen.backtest._snapshot_ledger import snapshot_ledger as _snapshot_ledger
import numbers
from collections.abc import Callable, Mapping, Sized
from typing import cast

import numpy as np
import numpy.typing as npt

from limen.backtest.long_flat_strategy import ExecutionResult, long_flat_strategy
from limen.backtest._long_flat_tp_sl import long_flat_tp_sl as _long_flat_tp_sl, validate_barrier as _validate_barrier

PRICE_CHANGE_RTOL = 1e-09
PRICE_CHANGE_ATOL = 1e-12


def _validate_execution_result(result: object, expected_len: int) -> ExecutionResult:
    if not isinstance(result, ExecutionResult):
        raise ValueError('backtest_snapshot strategy must return ExecutionResult(pos, gross, net)')

    normalized: dict[str, npt.NDArray[np.float64]] = {}
    for field in ('pos', 'gross', 'net'):
        try:
            arr = np.asarray(getattr(result, field), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'backtest_snapshot strategy {field} must be numeric') from exc
        if arr.ndim != 1 or arr.shape[0] != expected_len:
            raise ValueError(
                f'backtest_snapshot strategy {field} must be a full-window array matching the input length'
            )
        if not np.isfinite(arr).all():
            raise ValueError(f'backtest_snapshot strategy {field} must be finite')
        normalized[field] = arr

    return ExecutionResult(**normalized)


def snapshot_execution(
    columns: Mapping[str, object], *, pred_col: str, open_col: str,
    close_col: str, price_change_col: str,
    strategy: Callable[..., ExecutionResult], execution_lag_bars: object,
    fee_bps: float, slip_bps: float, notional_rate: float,
    take_profit_bps: float | None, stop_loss_bps: float | None,
    high_col: str, low_col: str,
) -> ExecutionResult:
    enabled = take_profit_bps is not None or stop_loss_bps is not None
    take_profit_bps = _validate_barrier('take_profit_bps', take_profit_bps)
    stop_loss_bps = _validate_barrier('stop_loss_bps', stop_loss_bps)
    required_cols = (pred_col, open_col, close_col, price_change_col)
    if enabled:
        required_cols += (high_col, low_col)
        if strategy is not long_flat_strategy:
            raise ValueError('backtest_snapshot custom strategy cannot use TP/SL')
        if isinstance(execution_lag_bars, bool) or not isinstance(execution_lag_bars, int) or execution_lag_bars <= 0:
            raise ValueError('backtest_snapshot execution_lag_bars must be a positive integer for TP/SL')
    missing = [col for col in required_cols if col not in columns]
    if missing:
        raise ValueError(f"backtest_snapshot columns mapping is missing required keys: {', '.join(missing)}")

    try:
        if any(not isinstance(columns[col], Sized) for col in required_cols):
            raise TypeError('unsized column')
        lengths = {len(cast(Sized, columns[col])) for col in required_cols}
    except TypeError as exc:
        raise ValueError('backtest_snapshot columns must be sized array-likes') from exc
    if lengths == {0}:
        raise ValueError('backtest_snapshot requires at least one row')
    if len(lengths) != 1:
        raise ValueError('backtest_snapshot columns must have equal lengths')

    if (
        isinstance(notional_rate, bool)
        or not isinstance(notional_rate, numbers.Real)
        or not 0 < notional_rate <= 1
    ):
        raise ValueError('backtest_snapshot notional_rate must be in (0, 1]')

    try:
        open_px = np.asarray(cast(npt.ArrayLike, columns[open_col]), dtype=float)
        close_px = np.asarray(cast(npt.ArrayLike, columns[close_col]), dtype=float)
        dpx = np.asarray(cast(npt.ArrayLike, columns[price_change_col]), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('backtest_snapshot open, close, and price_change must be numeric') from exc

    high = low = None
    if enabled:
        try:
            high = np.asarray(cast(npt.ArrayLike, columns[high_col]), dtype=float)
            low = np.asarray(cast(npt.ArrayLike, columns[low_col]), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError('backtest_snapshot high and low must be numeric') from exc
        arrays = (open_px, close_px, dpx, high, low)
        if any(arr.ndim != 1 for arr in arrays):
            raise ValueError('backtest_snapshot OHLC must be equal-length 1D arrays')
        if any(not np.isfinite(arr).all() for arr in arrays) or any((arr <= 0).any() for arr in (open_px, close_px, high, low)):
            raise ValueError('backtest_snapshot OHLC must be finite and positive')
        if (low > np.minimum(open_px, close_px)).any() or (high < np.maximum(open_px, close_px)).any():
            raise ValueError('backtest_snapshot OHLC high/low bounds are invalid')

    price_check_mask = ~np.isnan(open_px) & ~np.isnan(close_px) & ~np.isnan(dpx)
    expected_dpx = close_px - open_px

    if price_check_mask.any() and not np.isclose(
        dpx[price_check_mask],
        expected_dpx[price_check_mask],
        rtol=PRICE_CHANGE_RTOL,
        atol=PRICE_CHANGE_ATOL,
    ).all():
        raise ValueError('backtest_snapshot price_change must equal close - open')

    total_bars = open_px.shape[0]
    if high is not None and low is not None:
        result = _long_flat_tp_sl(
            cast(npt.ArrayLike, columns[pred_col]), open_px, close_px, dpx,
            high_px=high, low_px=low, take_profit_bps=take_profit_bps,
            stop_loss_bps=stop_loss_bps, execution_lag_bars=cast(int, execution_lag_bars),
            fee_bps=fee_bps, slip_bps=slip_bps,
        )
    else:
        result = strategy(
            columns[pred_col], open_px, close_px, dpx,
            execution_lag_bars=cast(int, execution_lag_bars), fee_bps=fee_bps, slip_bps=slip_bps,
        )
    result = _validate_execution_result(result, total_bars)

    return result


def snapshot_with_execution(
    columns: Mapping[str, object], *, pred_col: str = 'predictions',
    open_col: str = 'open', close_col: str = 'close',
    price_change_col: str = 'price_change',
    strategy: Callable[..., ExecutionResult] = long_flat_strategy,
    execution_lag_bars: int = 1, fee_bps: float = 5.0,
    slip_bps: float = 5.0, notional_rate: float = 1.0,
    take_profit_bps: float | None = None, stop_loss_bps: float | None = None,
    high_col: str = 'high', low_col: str = 'low',
) -> tuple[dict[str, float], ExecutionResult]:
    result = snapshot_execution(
        columns, pred_col=pred_col, open_col=open_col, close_col=close_col,
        price_change_col=price_change_col, strategy=strategy,
        execution_lag_bars=execution_lag_bars, fee_bps=fee_bps,
        slip_bps=slip_bps, notional_rate=notional_rate,
        take_profit_bps=take_profit_bps, stop_loss_bps=stop_loss_bps,
        high_col=high_col, low_col=low_col,
    )
    metrics = _snapshot_ledger(result, notional_rate)
    return metrics, result


__all__ = ['snapshot_execution', 'snapshot_with_execution']

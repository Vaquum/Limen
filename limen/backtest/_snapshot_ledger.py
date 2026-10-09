import numpy as np
import numpy.typing as npt

from limen.backtest.long_flat_strategy import ExecutionResult

BPS_PER_UNIT = 10_000.0
CVAR_TAIL_FRACTION = 0.05
CVAR_MIN_BARS = 20


def _finite_values(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _quantiles(values: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    arr = _finite_values(values)
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    p05, p50, p95 = (float(np.quantile(arr, q)) for q in (0.05, 0.50, 0.95))
    return (p05, p50, p95)


def _mean_bps(values: npt.NDArray[np.float64]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan
    return float(arr.mean()) * BPS_PER_UNIT


def _cvar_tail_bps(returns: npt.NDArray[np.float64]) -> float:
    arr = np.asarray(returns, dtype=float)
    if arr.size < CVAR_MIN_BARS:
        return np.nan
    tail_count = int(np.floor(CVAR_TAIL_FRACTION * arr.size))
    return float(np.sort(arr)[:tail_count].mean()) * BPS_PER_UNIT


def snapshot_ledger(result: ExecutionResult, notional_rate: float) -> dict[str, float]:
    gross = result.gross * notional_rate
    net = result.net * notional_rate
    pos = result.pos * notional_rate

    eq_net = np.cumprod(1.0 + net)
    drawdown = (eq_net / np.clip(np.maximum.accumulate(eq_net), 1.0, None)) - 1.0
    cost = gross - net
    wins = net > 0
    in_market = pos > 0
    entry_mask = in_market & ~np.concatenate(([False], in_market[:-1]))

    data: dict[str, float] = {}
    for prefix, values in [
        ('edge_bps', gross * BPS_PER_UNIT),
        ('pnl_bps', net * BPS_PER_UNIT),
        ('cost_bps', cost * BPS_PER_UNIT),
        ('drawdown_bps', drawdown * BPS_PER_UNIT),
    ]:
        p5, p50, p95 = _quantiles(values)
        data[f'{prefix}_p5'] = p5
        data[f'{prefix}_p50'] = p50
        data[f'{prefix}_p95'] = p95

    data['wins_per_bar'] = float(wins.mean())
    data['pnl_per_bar_bps'] = _mean_bps(net)
    data['avg_win_bps'] = _mean_bps(net[wins])
    data['avg_loss_bps'] = _mean_bps(net[net < 0])
    data['cvar_95_pnl_bps'] = _cvar_tail_bps(net)
    data['trades_per_bar'] = float(entry_mask.sum()) / result.pos.size
    data['inventory_per_bar'] = float(pos.mean())
    data['cost_per_bar_bps'] = _mean_bps(cost)

    return data


__all__ = ['snapshot_ledger']

import numbers
from collections.abc import Callable

import numpy as np
import pandas as pd

from limen.backtest.long_flat_strategy import ExecutionResult
from limen.backtest.long_flat_strategy import long_flat_strategy

PRICE_CHANGE_RTOL = 1e-09
PRICE_CHANGE_ATOL = 1e-12
BPS_PER_UNIT = 10_000.0
CVAR_TAIL_FRACTION = 0.05
CVAR_MIN_BARS = 20
BPS_DECIMALS = 1
FRACTION_DECIMALS = 4
RATE_DECIMALS = 5
BACKTEST_SNAPSHOT_COLUMNS = [
    'edge_bps_p5',
    'edge_bps_p50',
    'edge_bps_p95',
    'pnl_bps_p5',
    'pnl_bps_p50',
    'pnl_bps_p95',
    'cost_bps_p5',
    'cost_bps_p50',
    'cost_bps_p95',
    'drawdown_bps_p5',
    'drawdown_bps_p50',
    'drawdown_bps_p95',
    'wins_per_bar',
    'pnl_per_bar_bps',
    'avg_win_bps',
    'avg_loss_bps',
    'cvar_95_pnl_bps',
    'trades_per_bar',
    'inventory_per_bar',
    'cost_per_bar_bps',
]


def _finite_values(values: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _quantiles(values: pd.Series | np.ndarray | list[float], decimals: int = BPS_DECIMALS) -> tuple[float, float, float]:
    arr = _finite_values(values)
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    return tuple(round(float(np.quantile(arr, q)), decimals) for q in (0.05, 0.50, 0.95))


def _mean_bps(values: pd.Series) -> float:
    return round(float(values.mean()) * BPS_PER_UNIT, BPS_DECIMALS)


def _cvar_tail_bps(returns: pd.Series) -> float:
    arr = returns.to_numpy(dtype=float)
    if arr.size < CVAR_MIN_BARS:
        return np.nan
    tail_count = int(np.floor(CVAR_TAIL_FRACTION * arr.size))
    return round(float(np.sort(arr)[:tail_count].mean()) * BPS_PER_UNIT, BPS_DECIMALS)


def _validate_execution_result(result: ExecutionResult, expected_index: pd.Index) -> ExecutionResult:
    if not isinstance(result, ExecutionResult):
        raise ValueError('backtest_snapshot strategy must return ExecutionResult(pos, gross, net)')

    normalized: dict[str, pd.Series] = {}
    for field in ('pos', 'gross', 'net'):
        values = getattr(result, field)
        if not isinstance(values, pd.Series):
            raise ValueError(f'backtest_snapshot strategy {field} must be a pd.Series')
        if len(values) != len(expected_index) or not values.index.equals(expected_index):
            raise ValueError(
                f'backtest_snapshot strategy {field} must be a full-window series with the input index'
            )
        try:
            numeric = pd.to_numeric(values, errors='raise')
        except (TypeError, ValueError) as exc:
            raise ValueError(f'backtest_snapshot strategy {field} must be numeric') from exc
        arr = numeric.to_numpy(dtype=float)
        if np.isnan(arr).any() or not np.isfinite(arr).all():
            raise ValueError(f'backtest_snapshot strategy {field} must be finite')
        normalized[field] = pd.Series(arr, index=expected_index, name=values.name)

    return ExecutionResult(**normalized)


def backtest_snapshot(df: pd.DataFrame,
                      *,
                      pred_col: str = 'predictions',
                      open_col: str = 'open',
                      close_col: str = 'close',
                      price_change_col: str = 'price_change',
                      strategy: Callable[..., ExecutionResult] = long_flat_strategy,
                      execution_lag_bars: int = 1,
                      fee_bps: float = 5.0,
                      slip_bps: float = 5.0,
                      notional_rate: float = 1.0) -> pd.DataFrame:

    '''
    Bar-based metric ledger over a strategy's per-bar returns.

    Validates the price columns, delegates execution to `strategy` (default
    long_flat_strategy), and summarizes the returned per-bar series into a one-row,
    purely bar-based ledger: one unit (the bar), one population (every bar in the
    window, with flat bars counted as a real 0), and every column intensive (a rate,
    ratio, or per-bar quantity). No wall-clock time.

    Takes in output of log.permutation_prediction_performance and returns the
    one-row backtest ledger.

    The strategy receives the prediction column and the validated open, close, and
    price_change series plus execution_lag_bars, fee_bps, and slip_bps, and returns an
    ExecutionResult of per-bar pos, gross, and net return series. Every column flows
    from that triple. notional_rate (the deployed fraction of capital) is then applied
    here as a uniform scale on that triple — it commutes with the fill mechanics, so
    strategies never handle it — scaling edge, pnl, and cost and making
    inventory_per_bar the average deployed notional; 1.0 is all-in.

    Columns (all computed over every bar)
    - Distributions (p5/p50/p95): edge_bps (gross return), pnl_bps (net return),
      cost_bps (gross minus net), drawdown_bps (net equity against its running peak).
    - Scalars: wins_per_bar, pnl_per_bar_bps, avg_win_bps, avg_loss_bps, cvar_95_pnl_bps,
      trades_per_bar, inventory_per_bar, cost_per_bar_bps.

    wins_per_bar is the share of all bars with a positive net return (a flat bar is not
    a win), so it cannot exceed inventory_per_bar, the average position held per bar.
    avg_win_bps and avg_loss_bps are NaN when there are no winning or no losing bars,
    and cvar_95_pnl_bps is NaN when there are fewer than CVAR_MIN_BARS bars.

    Args:
        df (pd.DataFrame): Per-round table with the prediction and price columns.
        pred_col (str): Prediction column name.
        open_col (str): Open price column name.
        close_col (str): Close price column name.
        price_change_col (str): Price-change column name (close minus open).
        strategy (Callable[..., ExecutionResult]): Execution model mapping the signal
            and prices to per-bar pos, gross, and net series.
        execution_lag_bars (int): Bars between a signal row and its execution row.
        fee_bps (float): Per-fill fee in basis points.
        slip_bps (float): Per-fill slippage in basis points.
        notional_rate (float): Fraction of capital deployed while in position, in (0, 1];
            applied as a uniform scale on the strategy's returned pos, gross, and net.

    Returns:
        pd.DataFrame: One-row ledger with columns BACKTEST_SNAPSHOT_COLUMNS.
    '''

    if df.empty:
        raise ValueError('backtest_snapshot requires at least one row')

    if (
        isinstance(notional_rate, bool)
        or not isinstance(notional_rate, numbers.Real)
        or not 0 < notional_rate <= 1
    ):
        raise ValueError('backtest_snapshot notional_rate must be in (0, 1]')

    try:
        open_px = pd.to_numeric(df[open_col], errors='raise')
        close_px = pd.to_numeric(df[close_col], errors='raise')
        dpx = pd.to_numeric(df[price_change_col], errors='raise')
    except (TypeError, ValueError) as exc:
        raise ValueError('backtest_snapshot open, close, and price_change must be numeric') from exc

    price_check_mask = open_px.notna() & close_px.notna() & dpx.notna()
    expected_dpx = close_px - open_px

    if price_check_mask.any() and not np.isclose(
        dpx[price_check_mask],
        expected_dpx[price_check_mask],
        rtol=PRICE_CHANGE_RTOL,
        atol=PRICE_CHANGE_ATOL,
    ).all():
        raise ValueError('backtest_snapshot price_change must equal close - open')

    result = strategy(
        df[pred_col],
        open_px,
        close_px,
        dpx,
        execution_lag_bars=execution_lag_bars,
        fee_bps=fee_bps,
        slip_bps=slip_bps,
    )
    result = _validate_execution_result(result, df.index)

    gross = result.gross * notional_rate
    net = result.net * notional_rate
    pos = result.pos * notional_rate
    total_bars = len(df)

    eq_net = (1.0 + net).cumprod()
    drawdown = (eq_net / eq_net.cummax().clip(lower=1.0)) - 1.0
    in_market = pos > 0
    entry_mask = in_market & (~in_market.shift(1, fill_value=False))

    data: dict[str, float] = {}
    for prefix, values in [
        ('edge_bps', gross * BPS_PER_UNIT),
        ('pnl_bps', net * BPS_PER_UNIT),
        ('cost_bps', (gross - net) * BPS_PER_UNIT),
        ('drawdown_bps', drawdown * BPS_PER_UNIT),
    ]:
        p5, p50, p95 = _quantiles(values, BPS_DECIMALS)
        data[f'{prefix}_p5'] = p5
        data[f'{prefix}_p50'] = p50
        data[f'{prefix}_p95'] = p95

    data['wins_per_bar'] = round(float((net > 0).mean()), FRACTION_DECIMALS)
    data['pnl_per_bar_bps'] = _mean_bps(net)
    data['avg_win_bps'] = _mean_bps(net[net > 0])
    data['avg_loss_bps'] = _mean_bps(net[net < 0])
    data['cvar_95_pnl_bps'] = _cvar_tail_bps(net)
    data['trades_per_bar'] = round(float(entry_mask.sum()) / total_bars, RATE_DECIMALS)
    data['inventory_per_bar'] = round(float(pos.mean()), FRACTION_DECIMALS)
    data['cost_per_bar_bps'] = _mean_bps(gross - net)

    data = {col: data[col] for col in BACKTEST_SNAPSHOT_COLUMNS}

    return pd.DataFrame.from_records([data])

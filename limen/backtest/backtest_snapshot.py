import numpy as np
import pandas as pd

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
    'win_rate',
    'pnl_per_bar_bps',
    'avg_win_bps',
    'avg_loss_bps',
    'cvar_95_pnl_bps',
    'trades_per_bar',
    'in_market_per_bar',
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


def backtest_snapshot(df: pd.DataFrame,
                     *,
                     pred_col: str = 'predictions',
                     open_col: str = 'open',
                     close_col: str = 'close',
                     price_change_col: str = 'price_change',
                     execution_lag_bars: int = 1,
                     fee_bps: float = 5.0,
                     slip_bps: float = 5.0) -> pd.DataFrame:

    '''
    Long-only, hold-while-1 evaluation using pre-aligned intrabar returns.

    Emits a purely bar-based metric ledger: one unit (the bar), one population
    (every bar in the window, with flat bars counted as a real 0), and every
    column intensive (a rate, ratio, or per-bar quantity). No wall-clock time.

    Takes in output of log.permutation_prediction_performance and returns the
    one-row backtest ledger.

    Atoms (built once; every column flows from these)
    - Predictions are shifted forward by `execution_lag_bars` onto the execution rows.
    - pos = lagged predictions == 1 on a tradable execution row (long-only, all-in).
    - Entry bar gross return r_entry = price_change / open; continuation bar gross
      return r_cont = close_t / close_{t-1} - 1.
    - R_gross is r_entry on entry bars, r_cont on continuation bars, 0 on flat bars;
      fee/slippage are applied multiplicatively on the entry and exit fills to give
      R_net; eq_net compounds R_net.

    Columns (all computed over every bar)
    - Distributions (p5/p50/p95): edge_bps (gross return), pnl_bps (net return),
      cost_bps (gross minus net), drawdown_bps (net equity against its running peak).
    - Scalars: win_rate, pnl_per_bar_bps, avg_win_bps, avg_loss_bps, cvar_95_pnl_bps,
      trades_per_bar, in_market_per_bar, inventory_per_bar, cost_per_bar_bps.

    avg_win_bps and avg_loss_bps are NaN when there are no winning or no losing bars,
    and cvar_95_pnl_bps is NaN when there are fewer than CVAR_MIN_BARS bars.

    Returns a one-row DataFrame with columns (in order):
      BACKTEST_SNAPSHOT_COLUMNS
    '''

    df = df.copy()

    if df.empty:
        raise ValueError('backtest_snapshot requires at least one row')

    if execution_lag_bars < 0:
        raise ValueError('backtest_snapshot execution_lag_bars must be >= 0')

    try:
        pred = pd.to_numeric(df[pred_col], errors='raise')
    except (TypeError, ValueError) as exc:
        raise ValueError('backtest_snapshot predictions must contain only 0 or 1') from exc

    if pred.isna().any() or (~pred.isin([0, 1])).any():
        raise ValueError('backtest_snapshot predictions must contain only 0 or 1')

    pred = pred.astype(int)
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

    tradable = open_px.notna() & close_px.notna() & dpx.notna() & (open_px != 0)
    execution_rows = pd.Series(False, index=df.index)
    if execution_lag_bars < len(df):
        execution_rows.iloc[execution_lag_bars:] = True

    pred = pred.shift(execution_lag_bars, fill_value=0)
    eval_mask = execution_rows & tradable
    pos = (pred == 1) & eval_mask

    entry_mask = pos & (~pos.shift(1, fill_value=False))
    cont_mask = pos & (pos.shift(1, fill_value=False))
    exit_mask = pos & (~pos.shift(-1, fill_value=False))

    r_entry = dpx / open_px
    r_cont = (close_px / close_px.shift(1)) - 1.0

    R_gross = np.where(entry_mask, r_entry, 0.0) + np.where(cont_mask, r_cont, 0.0)
    R_gross = pd.Series(R_gross, index=df.index).fillna(0.0)

    fee = fee_bps / BPS_PER_UNIT
    slip = slip_bps / BPS_PER_UNIT
    entry_mult = (1.0 - fee) / (1.0 + slip)
    exit_mult = (1.0 - fee) * (1.0 - slip)

    cost_mult = pd.Series(1.0, index=df.index)
    cost_mult.loc[entry_mask] *= entry_mult
    cost_mult.loc[exit_mask] *= exit_mult

    R_net = (((1.0 + R_gross) * cost_mult) - 1.0).fillna(0.0)
    eq_net = (1.0 + R_net).cumprod()

    total_bars = len(df)
    capital_fraction = pos.astype(float)
    drawdown = (eq_net / eq_net.cummax().clip(lower=1.0)) - 1.0

    data: dict[str, float] = {}
    for prefix, values in [
        ('edge_bps', R_gross * BPS_PER_UNIT),
        ('pnl_bps', R_net * BPS_PER_UNIT),
        ('cost_bps', (R_gross - R_net) * BPS_PER_UNIT),
        ('drawdown_bps', drawdown * BPS_PER_UNIT),
    ]:
        p5, p50, p95 = _quantiles(values, BPS_DECIMALS)
        data[f'{prefix}_p5'] = p5
        data[f'{prefix}_p50'] = p50
        data[f'{prefix}_p95'] = p95

    data['win_rate'] = round(float((R_net > 0).mean()), FRACTION_DECIMALS)
    data['pnl_per_bar_bps'] = _mean_bps(R_net)
    data['avg_win_bps'] = _mean_bps(R_net[R_net > 0])
    data['avg_loss_bps'] = _mean_bps(R_net[R_net < 0])
    data['cvar_95_pnl_bps'] = _cvar_tail_bps(R_net)
    data['trades_per_bar'] = round(float(entry_mask.sum()) / total_bars, RATE_DECIMALS)
    data['in_market_per_bar'] = round(float((pos != 0).sum()) / total_bars, FRACTION_DECIMALS)
    data['inventory_per_bar'] = round(float(capital_fraction.mean()), FRACTION_DECIMALS)
    data['cost_per_bar_bps'] = _mean_bps(R_gross - R_net)

    data = {col: data[col] for col in BACKTEST_SNAPSHOT_COLUMNS}

    return pd.DataFrame.from_records([data])

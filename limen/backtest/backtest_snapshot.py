import numpy as np
import pandas as pd

PRICE_CHANGE_RTOL = 1e-09
PRICE_CHANGE_ATOL = 1e-12
BPS_PER_UNIT = 10_000.0
BACKTEST_SNAPSHOT_COLUMNS = [
    'edge_per_signal_bps_p5',
    'edge_per_signal_bps_p50',
    'edge_per_signal_bps_p95',
    'trade_pnl_net_bps_p5',
    'trade_pnl_net_bps_p50',
    'trade_pnl_net_bps_p95',
    'cost_drag_bps_p5',
    'cost_drag_bps_p50',
    'cost_drag_bps_p95',
    'rolling_return_net_bps_p5',
    'rolling_return_net_bps_p50',
    'rolling_return_net_bps_p95',
    'return_on_exposure_p5',
    'return_on_exposure_p50',
    'return_on_exposure_p95',
    'drawdown_depth_bps_p5',
    'drawdown_depth_bps_p50',
    'drawdown_depth_bps_p95',
    'drawdown_duration_days_p5',
    'drawdown_duration_days_p50',
    'drawdown_duration_days_p95',
    'cvar_95_return_bps',
]


def _finite_values(values: pd.Series | np.ndarray | list[float]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _quantiles(values: pd.Series | np.ndarray | list[float], decimals: int = 1) -> tuple[float, float, float]:
    arr = _finite_values(values)
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    return tuple(round(float(np.quantile(arr, q)), decimals) for q in (0.05, 0.50, 0.95))


def _clock_window_returns(
        df: pd.DataFrame,
        eval_mask: pd.Series,
        pos: pd.Series,
        R_net: pd.Series,
        datetime_col: str,
        clock_window: str) -> tuple[pd.Series, pd.Series]:

    if datetime_col not in df:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    dt = pd.to_datetime(df[datetime_col], errors='coerce')
    mask = eval_mask & dt.notna()
    if not mask.any():
        return pd.Series(dtype=float), pd.Series(dtype=float)

    windows = dt[mask].dt.floor(clock_window)
    window_returns = (1.0 + R_net[mask]).groupby(windows).prod() - 1.0
    exposure = pos[mask].astype(float).groupby(windows).mean()
    return_on_exposure = (window_returns / exposure).where(exposure > 0) * BPS_PER_UNIT

    return window_returns * BPS_PER_UNIT, return_on_exposure


def _drawdown_episode_metrics(
        eq_net: pd.Series,
        eval_mask: pd.Series,
        df: pd.DataFrame,
        datetime_col: str) -> tuple[list[float], list[float]]:

    eq = eq_net[eval_mask]
    if eq.empty:
        return [], []

    drawdown = (eq / eq.cummax().clip(lower=1.0)) - 1.0
    timestamps = (
        pd.to_datetime(df.loc[eq.index, datetime_col], errors='coerce')
        if datetime_col in df
        else pd.Series(pd.NaT, index=eq.index)
    )

    depths_bps: list[float] = []
    durations_days: list[float] = []
    in_drawdown = False
    start_time = pd.NaT
    trough = 0.0

    for idx, dd in drawdown.items():
        ts = timestamps.loc[idx]
        if dd < 0 and not in_drawdown:
            in_drawdown = True
            start_time = ts
            trough = float(dd)
        elif dd < 0:
            trough = min(trough, float(dd))
        elif in_drawdown:
            depths_bps.append(trough * BPS_PER_UNIT)
            if pd.notna(start_time) and pd.notna(ts):
                durations_days.append(float((ts - start_time) / pd.Timedelta(days=1)))
            in_drawdown = False
            start_time = pd.NaT
            trough = 0.0

    if in_drawdown:
        depths_bps.append(trough * BPS_PER_UNIT)

    return depths_bps, durations_days

def backtest_snapshot(df: pd.DataFrame,
                     *,
                     pred_col: str = 'predictions',
                     open_col: str = 'open',
                     close_col: str = 'close',
                     price_change_col: str = 'price_change',
                     datetime_col: str = 'datetime',
                     execution_lag_bars: int = 1,
                     clock_window: str = '1D',
                     fee_bps: float = 5.0,
                     slip_bps: float = 5.0) -> pd.DataFrame:

    '''
    Long-only, HOLD-WHILE-1 evaluation using pre-aligned intrabar returns.
    Emits the decoder-level metric ledger used before market replay.
    Return and ratio outputs are basis-point scaled.

    Takes in output of log.permutation_prediction_performance and returns backtest results.

    Logic
    - Predictions are shifted forward by `execution_lag_bars` onto the execution bar sequence.
    - Position pos = 1 wherever the lagged predictions==1 on a tradable execution row.
    - Price columns must be numeric; missing price rows are treated as non-tradable gaps.
    - Entry bar gross return: r_entry = price_change / open  (≈ close/open - 1).
    - Continuation bar gross return: r_cont = close_t / close_{t-1} - 1  (holding across bars).
    - Fee/slippage costs are applied multiplicatively on entry and exit fills.
    - Trade metrics are computed from compounded consecutive 1-run returns.
    - Equity compounds over R_net; drawdown is computed from net equity.

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
    cont_mask  = pos & ( pos.shift(1, fill_value=False))

    r_entry = dpx / open_px
    r_cont  = (close_px / close_px.shift(1)) - 1.0

    R_gross = np.where(entry_mask, r_entry, 0.0) + np.where(cont_mask, r_cont, 0.0)
    R_gross = pd.Series(R_gross, index=df.index).fillna(0.0)

    fee = fee_bps / BPS_PER_UNIT
    slip = slip_bps / BPS_PER_UNIT
    entry_mult = (1.0 - fee) / (1.0 + slip)
    exit_mult = (1.0 - fee) * (1.0 - slip)

    exit_mask = pos & (~pos.shift(-1, fill_value=False))
    cost_mult = pd.Series(1.0, index=df.index)
    cost_mult.loc[entry_mask] *= entry_mult
    cost_mult.loc[exit_mask] *= exit_mult

    R_net = (((1.0 + R_gross) * cost_mult) - 1.0).fillna(0.0)
    eq_net = (1.0 + R_net).cumprod()

    run_ids = entry_mask.cumsum()
    trade_pnl_net = (
        (1.0 + R_net[pos]).groupby(run_ids[pos]).prod() - 1.0
    ) if entry_mask.any() else pd.Series(dtype=float)
    trade_pnl_gross = (
        (1.0 + R_gross[pos]).groupby(run_ids[pos]).prod() - 1.0
    ) if entry_mask.any() else pd.Series(dtype=float)

    edge_per_signal = R_gross[pos] * BPS_PER_UNIT
    trade_pnl_net_bps = trade_pnl_net * BPS_PER_UNIT
    cost_drag_bps = (trade_pnl_gross - trade_pnl_net) * BPS_PER_UNIT
    rolling_return_net_bps, return_on_exposure = _clock_window_returns(
        df, eval_mask, pos, R_net, datetime_col, clock_window
    )
    drawdown_depth_bps, drawdown_duration_days = _drawdown_episode_metrics(
        eq_net, eval_mask, df, datetime_col
    )

    cvar_values = _finite_values(rolling_return_net_bps)
    if cvar_values.size:
        cvar_cutoff = np.quantile(cvar_values, 0.05)
        cvar_95_return_bps = round(float(cvar_values[cvar_values <= cvar_cutoff].mean()), 1)
    else:
        cvar_95_return_bps = np.nan

    data: dict[str, float] = {}
    for prefix, values, decimals in [
        ('edge_per_signal_bps', edge_per_signal, 1),
        ('trade_pnl_net_bps', trade_pnl_net_bps, 1),
        ('cost_drag_bps', cost_drag_bps, 1),
        ('rolling_return_net_bps', rolling_return_net_bps, 1),
        ('return_on_exposure', return_on_exposure, 1),
        ('drawdown_depth_bps', drawdown_depth_bps, 1),
        ('drawdown_duration_days', drawdown_duration_days, 3),
    ]:
        p5, p50, p95 = _quantiles(values, decimals)
        data[f'{prefix}_p5'] = p5
        data[f'{prefix}_p50'] = p50
        data[f'{prefix}_p95'] = p95

    data['cvar_95_return_bps'] = cvar_95_return_bps
    data = {col: data[col] for col in BACKTEST_SNAPSHOT_COLUMNS}

    return pd.DataFrame.from_records([data])

import numpy as np
import pandas as pd

PRICE_CHANGE_RTOL = 1e-09
PRICE_CHANGE_ATOL = 1e-12
BPS_PER_UNIT = 10_000.0

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
    Long-only, HOLD-WHILE-1 evaluation using pre-aligned intrabar returns.
    All ratio fields use basis points, including bounded proportions:
    0.5 is reported as 5000.0 bps. Sharpe is per bar (unitless).

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
      [
        'trade_win_rate_bps',
        'trade_expectancy_bps',
        'max_drawdown_bps',
        'total_return_gross_bps',
        'total_return_net_bps',
        'trade_return_mean_win_bps',
        'trade_return_mean_loss_bps',
        'mean_kelly_bps',
        'bars_total',
        'sharpe_per_bar',
        'bars_in_market_bps',
        'trades_count',
        'cost_round_trip_bps',
      ]
    '''

    df = df.copy()

    if df.empty:
        raise ValueError('backtest_snapshot requires at least one row')

    if execution_lag_bars < 0:
        raise ValueError('execution_lag_bars must be >= 0')

    try:
        pred = pd.to_numeric(df[pred_col], errors='raise')
    except (TypeError, ValueError) as exc:
        raise ValueError('predictions must contain only 0 or 1') from exc

    if pred.isna().any() or (~pred.isin([0, 1])).any():
        raise ValueError('predictions must contain only 0 or 1')

    pred = pred.astype(int)
    try:
        open_px = pd.to_numeric(df[open_col], errors='raise')
        close_px = pd.to_numeric(df[close_col], errors='raise')
        dpx = pd.to_numeric(df[price_change_col], errors='raise')
    except (TypeError, ValueError) as exc:
        raise ValueError('open, close, and price_change must be numeric') from exc

    price_check_mask = open_px.notna() & close_px.notna() & dpx.notna()
    expected_dpx = close_px - open_px

    if price_check_mask.any() and not np.isclose(
        dpx[price_check_mask],
        expected_dpx[price_check_mask],
        rtol=PRICE_CHANGE_RTOL,
        atol=PRICE_CHANGE_ATOL,
    ).all():
        raise ValueError('price_change must equal close - open')

    tradable = open_px.notna() & close_px.notna() & dpx.notna() & (open_px != 0)
    execution_rows = pd.Series(False, index=df.index)
    if execution_lag_bars < len(df):
        execution_rows.iloc[execution_lag_bars:] = True

    pred = pred.shift(execution_lag_bars, fill_value=0)
    eval_mask = execution_rows & tradable
    pos = (pred == 1) & eval_mask

    bars_total = int(eval_mask.sum())
    bars_in_market_bps = float((pos.sum() / bars_total) * BPS_PER_UNIT) if bars_total else np.nan

    entry_mask = pos & (~pos.shift(1, fill_value=False))
    cont_mask  = pos & ( pos.shift(1, fill_value=False))

    trades_count = int(entry_mask.sum())

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
    eq_gross = (1.0 + R_gross).cumprod()
    eq_net = (1.0 + R_net).cumprod()

    peak = eq_net.cummax().clip(lower=1.0)
    max_drawdown_bps = float((eq_net / peak - 1.0).min() * BPS_PER_UNIT)

    total_return_gross_bps = float((eq_gross.iloc[-1] - 1.0) * BPS_PER_UNIT)
    total_return_net_bps = float((eq_net.iloc[-1]   - 1.0) * BPS_PER_UNIT)

    run_ids = entry_mask.cumsum()
    trade_returns = (
        (1.0 + R_net[pos]).groupby(run_ids[pos]).prod() - 1.0
    ) if entry_mask.any() else pd.Series(dtype=float)

    if trade_returns.size:
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]
        trade_win_rate_bps = float((wins.size / trade_returns.size) * BPS_PER_UNIT)
        trade_expectancy_bps = float(trade_returns.mean() * BPS_PER_UNIT)
        trade_return_mean_win_bps = float(wins.mean() * BPS_PER_UNIT) if wins.size else np.nan
        trade_return_mean_loss_bps = float(losses.mean() * BPS_PER_UNIT) if losses.size else np.nan
    else:
        trade_win_rate_bps = trade_expectancy_bps = np.nan
        trade_return_mean_win_bps = trade_return_mean_loss_bps = np.nan

    kelly_returns = trade_returns

    kelly_wins = kelly_returns[kelly_returns > 0]
    kelly_losses = kelly_returns[kelly_returns < 0]
    if kelly_wins.size and kelly_losses.size:
        win_rate = float(kelly_wins.size / kelly_returns.size)
        loss_rate = float(kelly_losses.size / kelly_returns.size)
        avg_win = float(kelly_wins.mean())
        avg_loss = abs(float(kelly_losses.mean()))
        payout_ratio = avg_win / avg_loss if avg_loss > 0 else np.nan
        mean_kelly_bps = float((win_rate - (loss_rate / payout_ratio)) * BPS_PER_UNIT) if payout_ratio > 0 else np.nan
    else:
        mean_kelly_bps = np.nan

    eval_returns = R_net[eval_mask]
    mu = float(eval_returns.mean()) if eval_returns.size else np.nan
    sd = float(eval_returns.std(ddof=1)) if eval_returns.size > 1 else np.nan

    sharpe_per_bar = float(mu / sd) if sd > 0 else np.nan

    data = pd.DataFrame.from_records([{
        'trade_win_rate_bps': round(trade_win_rate_bps, 1),
        'trade_expectancy_bps': round(trade_expectancy_bps, 1),
        'max_drawdown_bps': round(max_drawdown_bps, 1),
        'total_return_gross_bps': round(total_return_gross_bps, 1),
        'total_return_net_bps': round(total_return_net_bps, 1),
        'trade_return_mean_win_bps': round(trade_return_mean_win_bps, 1),
        'trade_return_mean_loss_bps': round(trade_return_mean_loss_bps, 1),
        'mean_kelly_bps': round(mean_kelly_bps, 1),
        'bars_total': int(bars_total),
        'sharpe_per_bar': round(sharpe_per_bar, 2),
        'bars_in_market_bps': round(bars_in_market_bps, 1),
        'trades_count': int(trades_count),
        'cost_round_trip_bps': round((1.0 - (entry_mult * exit_mult)) * BPS_PER_UNIT),
    }])

    return data

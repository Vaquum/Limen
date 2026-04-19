import numpy as np
import pandas as pd

def backtest_snapshot(df: pd.DataFrame,
                     *,
                     pred_col: str = 'predictions',
                     open_col: str = 'open',
                     close_col: str = 'close',
                     price_change_col: str = 'price_change',
                     execution_lag_bars: int = 1,
                     fee_bps: float = 5.0,
                     slip_bps: float = 5.0,
                     trades_count_mode: str = 'bars') -> pd.DataFrame:

    '''
    Long-only, HOLD-WHILE-1 evaluation using pre-aligned intrabar returns.
    All percentage fields are in % units (not fractions). Sharpe is per bar (unitless).

    Takes in output of log.permutation_prediction_performance and returns backtest results.

    Logic
    - Predictions are shifted forward by `execution_lag_bars` onto the execution bar sequence.
    - Position pos = 1 wherever the lagged predictions==1 on a tradable execution row.
    - Entry bar gross return: r_entry = price_change / open  (≈ close/open - 1).
    - Continuation bar gross return: r_cont = close_t / close_{t-1} - 1  (holding across bars).
    - One round-trip cost per *consecutive 1-run*, charged on the run's exit bar.
    - Net per-bar return: R_net = (pos * r_gross) + cost_at_exit_bar.
    - Equity compounds over R_net; drawdown is computed from net equity.

    Returns a one-row DataFrame with columns (in order):
      [
        'trade_win_rate_pct',
        'trade_expectancy_pct',
        'max_drawdown_pct',
        'total_return_gross_pct',
        'total_return_net_pct',
        'trade_return_mean_win_pct',
        'trade_return_mean_loss_pct',
        'mean_kelly_pct',
        'bars_total',
        'sharpe_per_bar',
        'bars_in_market_pct',
        'trades_count',
        'cost_round_trip_bps',
      ]
    '''

    df = df.copy()

    if execution_lag_bars < 0:
        raise ValueError('execution_lag_bars must be >= 0')

    pred = pd.to_numeric(df[pred_col], errors='coerce').fillna(0).astype(int).clip(0, 1)
    open_px = pd.to_numeric(df[open_col], errors='coerce')
    close_px = pd.to_numeric(df[close_col], errors='coerce')
    dpx = pd.to_numeric(df[price_change_col], errors='coerce')  # close - open

    tradable = open_px.notna() & close_px.notna() & dpx.notna() & (open_px != 0)
    execution_rows = pd.Series(False, index=df.index)
    if execution_lag_bars < len(df):
        execution_rows.iloc[execution_lag_bars:] = True

    pred = pred.shift(execution_lag_bars, fill_value=0)
    eval_mask = execution_rows & tradable
    pos = (pred == 1) & eval_mask

    bars_total = int(eval_mask.sum())
    bars_in_market_pct = float((pos.sum() / bars_total) * 100.0) if bars_total else np.nan

    if trades_count_mode == 'runs':
        entries = pos & (~pos.shift(1, fill_value=False))
        trades_count = int(entries.sum())
    else:
        trades_count = int(pos.sum())

    entry_mask = pos & (~pos.shift(1, fill_value=False))
    cont_mask  = pos & ( pos.shift(1, fill_value=False))

    r_entry = dpx / open_px
    r_cont  = (close_px / close_px.shift(1)) - 1.0

    R_gross = np.where(entry_mask, r_entry, 0.0) + np.where(cont_mask, r_cont, 0.0)
    R_gross = pd.Series(R_gross, index=df.index).fillna(0.0)

    rt_cost = 2.0 * (fee_bps + slip_bps) / 10_000.0
    exit_mask = pos & (~pos.shift(-1, fill_value=False))
    cost_bar = pd.Series(np.where(exit_mask, -rt_cost, 0.0), index=df.index)

    R_net = (R_gross + cost_bar).fillna(0.0)
    eq_gross = (1.0 + R_gross).cumprod()
    eq_net = (1.0 + R_net).cumprod()

    peak = eq_net.cummax()
    max_drawdown_pct = float((eq_net / peak - 1.0).min() * 100.0)

    total_return_gross_pct = float((eq_gross.iloc[-1] - 1.0) * 100.0)
    total_return_net_pct = float((eq_net.iloc[-1]   - 1.0) * 100.0)

    tr = R_net[pos]
    if tr.size:
        wins = tr[tr > 0]
        losses = tr[tr < 0]
        trade_win_rate_pct = float((wins.size / tr.size) * 100.0)
        trade_expectancy_pct = float(tr.mean() * 100.0)
        trade_return_mean_win_pct = float(wins.mean() * 100.0) if wins.size else np.nan
        trade_return_mean_loss_pct = float(losses.mean() * 100.0) if losses.size else np.nan
    else:
        trade_win_rate_pct = trade_expectancy_pct = _trade_return_mean_pct = np.nan
        trade_return_mean_win_pct = trade_return_mean_loss_pct = np.nan

    if trades_count_mode == 'runs':
        entries = pos & (~pos.shift(1, fill_value=False))
        run_ids = entries.cumsum()
        kelly_returns = ((1.0 + R_net[pos]).groupby(run_ids[pos]).prod() - 1.0) if int(entries.sum()) else pd.Series(dtype=float)
    else:
        kelly_returns = R_net[pos]

    kelly_wins = kelly_returns[kelly_returns > 0]
    kelly_losses = kelly_returns[kelly_returns < 0]
    if kelly_wins.size and kelly_losses.size:
        win_rate = float(kelly_wins.size / kelly_returns.size)
        loss_rate = float(kelly_losses.size / kelly_returns.size)
        avg_win = float(kelly_wins.mean())
        avg_loss = abs(float(kelly_losses.mean()))
        payout_ratio = avg_win / avg_loss if avg_loss > 0 else np.nan
        mean_kelly_pct = float((win_rate - (loss_rate / payout_ratio)) * 100.0) if payout_ratio > 0 else np.nan
    else:
        mean_kelly_pct = np.nan

    eval_returns = R_net[eval_mask]
    mu = float(eval_returns.mean()) if eval_returns.size else np.nan
    sd = float(eval_returns.std(ddof=1)) if eval_returns.size > 1 else np.nan

    sharpe_per_bar = float(mu / sd) if sd > 0 else np.nan

    data = pd.DataFrame.from_records([{
        'trade_win_rate_pct': round(trade_win_rate_pct, 1),
        'trade_expectancy_pct': round(trade_expectancy_pct, 3),
        'max_drawdown_pct': round(max_drawdown_pct, 1),
        'total_return_gross_pct': round(total_return_gross_pct, 1),
        'total_return_net_pct': round(total_return_net_pct, 1),
        'trade_return_mean_win_pct': round(trade_return_mean_win_pct, 1),
        'trade_return_mean_loss_pct': round(trade_return_mean_loss_pct, 1),
        'mean_kelly_pct': round(mean_kelly_pct, 3),
        'bars_total': int(bars_total),
        'sharpe_per_bar': round(sharpe_per_bar, 2),
        'bars_in_market_pct': round(bars_in_market_pct, 1),
        'trades_count': int(trades_count),
        'cost_round_trip_bps': round(2 * (fee_bps + slip_bps)),
    }])

    return data

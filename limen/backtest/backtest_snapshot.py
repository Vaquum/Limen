import numpy as np
import pandas as pd

def backtest_snapshot(df: pd.DataFrame,
                     *,
                     pred_col: str = 'predictions',
                     open_col: str = 'open',
                     close_col: str = 'close',
                     price_change_col: str = 'price_change',
                     fee_bps: float = 5.0,
                     slip_bps: float = 5.0,
                     trades_count_mode: str = 'bars') -> pd.DataFrame:
    
    '''
    Long-only, HOLD-WHILE-1 evaluation using pre-aligned intrabar returns.
    All percentage fields are in % units (not fractions). Sharpe is per bar (unitless).

    Takes in output of log.permutation_prediction_performance and returns backtest results.

    Logic
    - Position pos = 1 wherever predictions==1 (no shifting).
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
        'bars_total',
        'sharpe_per_bar',
        'bars_in_market_pct',
        'trades_count',
        'cost_round_trip_bps',
      ]
    '''
    
    df = df.copy()

    pred = pd.to_numeric(df[pred_col], errors='coerce').fillna(0).astype(int).clip(0, 1)
    open_px = pd.to_numeric(df[open_col], errors='coerce')
    close_px = pd.to_numeric(df[close_col], errors='coerce')
    dpx = pd.to_numeric(df[price_change_col], errors='coerce')  # close - open

    pos = (pred == 1) & open_px.notna() & close_px.notna() & dpx.notna() & (open_px != 0)

    bars_total = int(len(df))
    bars_in_market_pct = float(pos.mean() * 100.0)

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

    mu = float(R_net.mean())
    sd = float(R_net.std(ddof=1))
    
    sharpe_per_bar = float(mu / sd) if sd > 0 else np.nan

    data = pd.DataFrame.from_records([{
        'trade_win_rate_pct': round(trade_win_rate_pct, 1),
        'trade_expectancy_pct': round(trade_expectancy_pct, 3),
        'max_drawdown_pct': round(max_drawdown_pct, 1),
        'total_return_gross_pct': round(total_return_gross_pct, 1),
        'total_return_net_pct': round(total_return_net_pct, 1),
        'trade_return_mean_win_pct': round(trade_return_mean_win_pct, 1),
        'trade_return_mean_loss_pct': round(trade_return_mean_loss_pct, 1),
        'bars_total': int(bars_total),
        'sharpe_per_bar': round(sharpe_per_bar, 2),
        'bars_in_market_pct': round(bars_in_market_pct, 1),
        'trades_count': int(trades_count),
        'cost_round_trip_bps': int(round(2 * (fee_bps + slip_bps))),
    }])

    return data

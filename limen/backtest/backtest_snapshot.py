import numpy as np
import pandas as pd


def backtest_snapshot(df: pd.DataFrame,
                     *,
                     pred_col: str = 'predictions',
                     actual_col: str = 'actuals',
                     open_col: str = 'open',
                     close_col: str = 'close',
                     price_change_col: str = 'price_change',
                     fee_bps: float = 5.0,
                     slip_bps: float = 5.0,
                     trades_count_mode: str = 'runs',
                     execution_lag_bars: int = 1) -> pd.DataFrame:

    '''
    Long-only, HOLD-WHILE-1 evaluation using aligned tradable returns.
    All percentage fields are in % units (not fractions). Sharpe is per bar (unitless).

    Takes in output of log.permutation_prediction_performance and returns backtest results.

    Logic
    - By default, a prediction made on row t is executed on row t+1.
    - Set `execution_lag_bars=0` to reproduce the legacy same-row execution.
    - Position pos = 1 wherever predictions==1 and lagged price data exists.
    - `bars_total`, `bars_in_market_pct`, and `sharpe_per_bar` are computed on the
      tradable window only, so the trailing `execution_lag_bars` rows are excluded.
    - Entry bar gross return: r_entry = price_change / open  (≈ close/open - 1).
    - Continuation bar gross return: r_cont = close_t / close_{t-1} - 1  (holding across bars).
    - One round-trip cost per *consecutive 1-run*, charged on the run's exit bar.
    - Net per-bar return: R_net = (pos * r_gross) + cost_at_exit_bar.
    - Equity compounds over R_net; drawdown is computed from net equity.
    - `trade_*` metrics are computed from completed 1-runs by default.
    - Set `trades_count_mode='bars'` to reproduce the legacy bar-level `trade_*` fields.
    - `bar_*` metrics are always computed from in-market bars.
    - `mean_kelly_pct` reports the full-Kelly fraction for the active return
      distribution, keeping zero-return observations in the sample denominator.

    Returns a one-row DataFrame with columns (in order):
      [
        'trade_win_rate_pct',
        'trade_expectancy_pct',
        'max_drawdown_pct',
        'total_return_gross_pct',
        'total_return_net_pct',
        'trade_return_mean_win_pct',
        'trade_return_mean_loss_pct',
        'bar_win_rate_pct',
        'bar_expectancy_pct',
        'bar_return_mean_win_pct',
        'bar_return_mean_loss_pct',
        'tp_mean_return_pct',
        'fp_mean_return_pct',
        'tn_mean_return_pct',
        'fn_mean_return_pct',
        'mean_kelly_pct',
        'bars_total',
        'sharpe_per_bar',
        'bars_in_market_pct',
        'bars_in_market_count',
        'trades_count',
        'trade_runs_count',
        'cost_round_trip_bps',
        'execution_lag_bars',
      ]
    '''

    if trades_count_mode not in {'runs', 'bars'}:
        raise ValueError('trades_count_mode must be either \'runs\' or \'bars\'')
    if execution_lag_bars < 0:
        raise ValueError('execution_lag_bars must be >= 0')

    df = df.copy()

    pred = pd.to_numeric(df[pred_col], errors='coerce').fillna(0).astype(int).clip(0, 1)
    open_px = pd.to_numeric(df[open_col], errors='coerce')
    close_px = pd.to_numeric(df[close_col], errors='coerce')
    dpx = pd.to_numeric(df[price_change_col], errors='coerce')  # close - open

    if execution_lag_bars > 0:
        open_px = open_px.shift(-execution_lag_bars)
        close_px = close_px.shift(-execution_lag_bars)
        dpx = dpx.shift(-execution_lag_bars)

    tradable = open_px.notna() & close_px.notna() & dpx.notna() & (open_px != 0)
    pos = (pred == 1) & tradable

    bars_total = int(tradable.sum())
    bars_in_market_count = int(pos.sum())
    bars_in_market_pct = float((bars_in_market_count / bars_total) * 100.0) if bars_total else np.nan

    entries = pos & (~pos.shift(1, fill_value=False))
    trade_runs_count = int(entries.sum())
    if trades_count_mode == 'bars':
        trades_count = bars_in_market_count
    else:
        trades_count = trade_runs_count

    entry_mask = entries
    cont_mask = pos & (pos.shift(1, fill_value=False))

    r_entry = dpx / open_px
    r_cont = (close_px / close_px.shift(1)) - 1.0

    R_gross = np.where(entry_mask, r_entry, 0.0) + np.where(cont_mask, r_cont, 0.0)
    R_gross = pd.Series(R_gross, index=df.index).fillna(0.0)

    rt_cost = 2.0 * (fee_bps + slip_bps) / 10_000.0
    exit_mask = pos & (~pos.shift(-1, fill_value=False))
    cost_bar = pd.Series(np.where(exit_mask, -rt_cost, 0.0), index=df.index)

    R_net = (R_gross + cost_bar).fillna(0.0)
    tradable_R_gross = R_gross[tradable]
    tradable_R_net = R_net[tradable]
    if tradable_R_net.empty:
        max_drawdown_pct = np.nan
        total_return_gross_pct = np.nan
        total_return_net_pct = np.nan
        sharpe_per_bar = np.nan
    else:
        eq_gross = (1.0 + tradable_R_gross).cumprod()
        eq_net = (1.0 + tradable_R_net).cumprod()
        peak = eq_net.cummax()
        max_drawdown_pct = float((eq_net / peak - 1.0).min() * 100.0)
        total_return_gross_pct = float((eq_gross.iloc[-1] - 1.0) * 100.0)
        total_return_net_pct = float((eq_net.iloc[-1] - 1.0) * 100.0)
        mu = float(tradable_R_net.mean())
        sd = float(tradable_R_net.std(ddof=1))
        sharpe_per_bar = float(mu / sd) if sd > 0 else np.nan

    bar_returns = R_net[pos]
    if bar_returns.size:
        bar_wins = bar_returns[bar_returns > 0]
        bar_losses = bar_returns[bar_returns < 0]
        bar_win_rate_pct = float((bar_wins.size / bar_returns.size) * 100.0)
        bar_expectancy_pct = float(bar_returns.mean() * 100.0)
        bar_return_mean_win_pct = float(bar_wins.mean() * 100.0) if bar_wins.size else np.nan
        bar_return_mean_loss_pct = float(bar_losses.mean() * 100.0) if bar_losses.size else np.nan
    else:
        bar_win_rate_pct = np.nan
        bar_expectancy_pct = np.nan
        bar_return_mean_win_pct = np.nan
        bar_return_mean_loss_pct = np.nan

    actual = pd.to_numeric(df[actual_col], errors='coerce') if actual_col in df else None
    if actual is not None:
        aligned_bar_return = (dpx / open_px).replace([np.inf, -np.inf], np.nan)
        valid_actual = tradable & actual.isin([0, 1]) & aligned_bar_return.notna()
        m_tp = valid_actual & (pred == 1) & (actual == 1)
        m_fp = valid_actual & (pred == 1) & (actual == 0)
        m_tn = valid_actual & (pred == 0) & (actual == 0)
        m_fn = valid_actual & (pred == 0) & (actual == 1)
        tp_mean_return_pct = float(aligned_bar_return.loc[m_tp].mean() * 100.0) if m_tp.any() else np.nan
        fp_mean_return_pct = float(aligned_bar_return.loc[m_fp].mean() * 100.0) if m_fp.any() else np.nan
        tn_mean_return_pct = float(aligned_bar_return.loc[m_tn].mean() * 100.0) if m_tn.any() else np.nan
        fn_mean_return_pct = float(aligned_bar_return.loc[m_fn].mean() * 100.0) if m_fn.any() else np.nan
    else:
        tp_mean_return_pct = np.nan
        fp_mean_return_pct = np.nan
        tn_mean_return_pct = np.nan
        fn_mean_return_pct = np.nan

    if trade_runs_count:
        run_ids = entries.cumsum()
        trade_returns = ((1.0 + R_net[pos]).groupby(run_ids[pos]).prod() - 1.0)
        trade_wins = trade_returns[trade_returns > 0]
        trade_losses = trade_returns[trade_returns < 0]
        trade_win_rate_pct = float((trade_wins.size / trade_returns.size) * 100.0)
        trade_expectancy_pct = float(trade_returns.mean() * 100.0)
        trade_return_mean_win_pct = float(trade_wins.mean() * 100.0) if trade_wins.size else np.nan
        trade_return_mean_loss_pct = float(trade_losses.mean() * 100.0) if trade_losses.size else np.nan
    else:
        trade_win_rate_pct = np.nan
        trade_expectancy_pct = np.nan
        trade_return_mean_win_pct = np.nan
        trade_return_mean_loss_pct = np.nan

    if trades_count_mode == 'bars':
        trade_win_rate_pct = bar_win_rate_pct
        trade_expectancy_pct = bar_expectancy_pct
        trade_return_mean_win_pct = bar_return_mean_win_pct
        trade_return_mean_loss_pct = bar_return_mean_loss_pct
        kelly_returns = bar_returns
    else:
        kelly_returns = trade_returns if trade_runs_count else pd.Series(dtype=float)

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

    data = pd.DataFrame.from_records([{
        'trade_win_rate_pct': round(trade_win_rate_pct, 1),
        'trade_expectancy_pct': round(trade_expectancy_pct, 3),
        'max_drawdown_pct': round(max_drawdown_pct, 1),
        'total_return_gross_pct': round(total_return_gross_pct, 1),
        'total_return_net_pct': round(total_return_net_pct, 1),
        'trade_return_mean_win_pct': round(trade_return_mean_win_pct, 1),
        'trade_return_mean_loss_pct': round(trade_return_mean_loss_pct, 1),
        'bar_win_rate_pct': round(bar_win_rate_pct, 1),
        'bar_expectancy_pct': round(bar_expectancy_pct, 3),
        'bar_return_mean_win_pct': round(bar_return_mean_win_pct, 1),
        'bar_return_mean_loss_pct': round(bar_return_mean_loss_pct, 1),
        'tp_mean_return_pct': round(tp_mean_return_pct, 3),
        'fp_mean_return_pct': round(fp_mean_return_pct, 3),
        'tn_mean_return_pct': round(tn_mean_return_pct, 3),
        'fn_mean_return_pct': round(fn_mean_return_pct, 3),
        'mean_kelly_pct': round(mean_kelly_pct, 3),
        'bars_total': int(bars_total),
        'sharpe_per_bar': round(sharpe_per_bar, 2),
        'bars_in_market_pct': round(bars_in_market_pct, 1),
        'bars_in_market_count': int(bars_in_market_count),
        'trades_count': int(trades_count),
        'trade_runs_count': int(trade_runs_count),
        'cost_round_trip_bps': round(2 * (fee_bps + slip_bps)),
        'execution_lag_bars': int(execution_lag_bars),
    }])

    return data

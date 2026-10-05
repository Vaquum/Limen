from collections.abc import Callable
from collections.abc import Mapping
from typing import Any


from limen.backtest.long_flat_strategy import ExecutionResult
from limen.backtest.long_flat_strategy import long_flat_strategy
from limen.backtest._snapshot_execution import snapshot_with_execution as _snapshot_with_execution

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


def backtest_snapshot(columns: Mapping[str, Any],
                      *,
                      pred_col: str = 'predictions',
                      open_col: str = 'open',
                      close_col: str = 'close',
                      price_change_col: str = 'price_change',
                      strategy: Callable[..., ExecutionResult] = long_flat_strategy,
                      execution_lag_bars: int = 1,
                      fee_bps: float = 5.0,
                      slip_bps: float = 5.0,
                      notional_rate: float = 1.0,
                      take_profit_bps: float | None = None,
                      stop_loss_bps: float | None = None,
                      high_col: str = 'high', low_col: str = 'low') -> dict[str, float]:

    '''
    Bar-based metric ledger over a strategy's per-bar returns.

    Validates the price columns, delegates execution to `strategy` (default
    long_flat_strategy), and summarizes the returned per-bar arrays into a one-row,
    purely bar-based ledger: one unit (the bar), one population (every bar in the
    window, with flat bars counted as a real 0), and every column intensive (a rate,
    ratio, or per-bar quantity). No wall-clock time.

    Takes the columns of log.permutation_prediction_performance as a mapping of
    equal-length array-like columns keyed by name (a dict of numpy arrays) and
    returns the one-row backtest ledger as a dict.

    The strategy receives the prediction column and the validated open, close, and
    price_change arrays plus execution_lag_bars, fee_bps, and slip_bps, and returns an
    ExecutionResult of per-bar pos, gross, and net return arrays. Every column flows
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
        columns (Mapping[str, Any]): Per-round columns with the prediction and price
            arrays, keyed by name
        pred_col (str): Prediction column name
        open_col (str): Open price column name
        close_col (str): Close price column name
        price_change_col (str): Price-change column name (close minus open)
        strategy (Callable[..., ExecutionResult]): Execution model mapping the signal
            and prices to per-bar pos, gross, and net arrays
        execution_lag_bars (int): Bars between a signal row and its execution row
        fee_bps (float): Per-fill fee in basis points
        slip_bps (float): Per-fill slippage in basis points
        notional_rate (float): Fraction of capital deployed while in position, in (0, 1];
            applied as a uniform scale on the strategy's returned pos, gross, and net

        take_profit_bps (float | None): Fixed gross entry-relative profit distance; None disables.
        stop_loss_bps (float | None): Fixed gross entry-relative loss distance; None disables.
        high_col (str): High price column required for enabled barriers.
        low_col (str): Low price column required for enabled barriers.

    Returns:
        dict[str, float]: One-row ledger keyed by BACKTEST_SNAPSHOT_COLUMNS
    '''

    return _snapshot_with_execution(
        columns, pred_col=pred_col, open_col=open_col, close_col=close_col,
        price_change_col=price_change_col, strategy=strategy,
        execution_lag_bars=execution_lag_bars, fee_bps=fee_bps,
        slip_bps=slip_bps, notional_rate=notional_rate,
        take_profit_bps=take_profit_bps, stop_loss_bps=stop_loss_bps,
        high_col=high_col, low_col=low_col,
    )[0]




__all__ = ['BACKTEST_SNAPSHOT_COLUMNS', 'backtest_snapshot']

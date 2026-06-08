from typing import NamedTuple

import numpy as np
import pandas as pd

BPS_PER_UNIT = 10_000.0

__all__ = ['ExecutionResult', 'long_flat_strategy']


class ExecutionResult(NamedTuple):

    '''
    Per-bar execution series consumed by the snapshot ledger.

    Attributes:
        pos (pd.Series): Position held per bar (0 when flat, else the deployed
            fraction notional_rate; 1 under the all-in default).
        gross (pd.Series): Per-bar gross return before costs.
        net (pd.Series): Per-bar net return after entry and exit costs.
    '''

    pos: pd.Series
    gross: pd.Series
    net: pd.Series


def long_flat_strategy(predictions: pd.Series,
                       open_px: pd.Series,
                       close_px: pd.Series,
                       price_change: pd.Series,
                       *,
                       execution_lag_bars: int = 1,
                       fee_bps: float = 5.0,
                       slip_bps: float = 5.0,
                       notional_rate: float = 1.0) -> ExecutionResult:

    '''
    Long-only, hold-while-1 execution over pre-aligned intrabar returns.

    Interprets a binary 0/1 signal as a long position sized at notional_rate
    (all-in by default): enter at the open of the first signalled bar, ride
    close-to-close while the signal persists, and exit at the close of the last
    signalled bar. Fee and slippage are applied multiplicatively on the entry and
    exit fills.

    Predictions are shifted forward by execution_lag_bars onto the execution rows.
    The entry-bar gross return is price_change / open; the continuation-bar gross
    return is close_t / close_{t-1} - 1; a flat bar is a real 0.

    notional_rate is the fraction of capital deployed while in position; the rest
    sits in cash at 0, so the per-bar pos, gross, and net all scale by it. Every
    ledger column then reflects the account at that bet size, and notional_rate of
    1.0 reproduces the all-in profile exactly.

    Args:
        predictions (pd.Series): Per-bar signal; must contain only 0 or 1.
        open_px (pd.Series): Bar open price.
        close_px (pd.Series): Bar close price.
        price_change (pd.Series): Bar close minus open.
        execution_lag_bars (int): Bars between a signal row and its execution row.
        fee_bps (float): Per-fill fee in basis points.
        slip_bps (float): Per-fill slippage in basis points.
        notional_rate (float): Fraction of capital deployed while in position, in
            (0, 1]; 1.0 is all-in.

    Returns:
        ExecutionResult: Per-bar pos, gross, and net return series.
    '''

    if execution_lag_bars < 0:
        raise ValueError('long_flat_strategy execution_lag_bars must be >= 0')

    if not (0 < notional_rate <= 1):
        raise ValueError('long_flat_strategy notional_rate must be in (0, 1]')

    try:
        pred = pd.to_numeric(predictions, errors='raise')
    except (TypeError, ValueError) as exc:
        raise ValueError('long_flat_strategy predictions must contain only 0 or 1') from exc

    if pred.isna().any() or (~pred.isin([0, 1])).any():
        raise ValueError('long_flat_strategy predictions must contain only 0 or 1')

    pred = pred.astype(int)
    index = predictions.index
    total_bars = len(index)

    tradable = open_px.notna() & close_px.notna() & price_change.notna() & (open_px != 0)
    execution_rows = pd.Series(False, index=index)
    if execution_lag_bars < total_bars:
        execution_rows.iloc[execution_lag_bars:] = True

    pred = pred.shift(execution_lag_bars, fill_value=0)
    eval_mask = execution_rows & tradable
    pos = (pred == 1) & eval_mask

    entry_mask = pos & (~pos.shift(1, fill_value=False))
    cont_mask = pos & (pos.shift(1, fill_value=False))
    exit_mask = pos & (~pos.shift(-1, fill_value=False))

    r_entry = price_change / open_px
    r_cont = (close_px / close_px.shift(1)) - 1.0

    gross = np.where(entry_mask, r_entry, 0.0) + np.where(cont_mask, r_cont, 0.0)
    gross = pd.Series(gross, index=index).fillna(0.0)

    fee = fee_bps / BPS_PER_UNIT
    slip = slip_bps / BPS_PER_UNIT
    entry_mult = (1.0 - fee) / (1.0 + slip)
    exit_mult = (1.0 - fee) * (1.0 - slip)

    cost_mult = pd.Series(1.0, index=index)
    cost_mult.loc[entry_mask] *= entry_mult
    cost_mult.loc[exit_mask] *= exit_mult

    net = (((1.0 + gross) * cost_mult) - 1.0).fillna(0.0)

    return ExecutionResult(
        pos=pos.astype(float) * notional_rate,
        gross=gross * notional_rate,
        net=net * notional_rate,
    )

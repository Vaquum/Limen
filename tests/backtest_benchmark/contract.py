"""Common outputs and an independent cash replay; no framework imports."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INITIAL_CASH = 1_000_000.0
EQUITY_RTOL = 1e-10
EQUITY_ATOL = 1e-6
FRACTIONAL_UNIT = 1e-12


@dataclass
class Result:
    equity: np.ndarray
    held: np.ndarray
    fill_bars: np.ndarray
    fill_sides: np.ndarray
    fill_prices: np.ndarray


def targets(signals):
    values = np.asarray(signals)
    if values.ndim != 1 or len(values) < 3 or not np.isin(values, (0, 1)).all():
        raise ValueError('Signals require two evaluation states and one real settlement row')
    desired = values.astype(np.int8).copy()
    # The last evaluated close liquidates; the last source row settles orders.
    desired[-2:] = 0
    return desired


def transitions(desired):
    previous = np.r_[0, desired[:-1]]
    bars = np.flatnonzero(desired != previous)
    sides = np.where(desired[bars] == 1, 1, -1).astype(np.int8)
    return bars, sides


def reference_result(close, desired):
    """Replay all-in cash and units independently of return compounding."""
    count = len(close) - 1
    equity = np.empty(count, dtype=float)
    held = np.empty(count, dtype=np.int8)
    cash, quantity = INITIAL_CASH, 0.0
    for bar in range(count):
        held[bar] = quantity > 0
        price = float(close[bar])
        if desired[bar] and not quantity:
            quantity, cash = cash / price, 0.0
        elif not desired[bar] and quantity:
            cash, quantity = quantity * price, 0.0
        equity[bar] = cash + quantity * price
    bars, sides = transitions(desired)
    return Result(equity, held, bars, sides, close[bars].copy())


def assert_parity(actual, expected):
    for name in ('equity', 'held', 'fill_bars', 'fill_sides', 'fill_prices'):
        value = getattr(actual, name)
        if not np.isfinite(value).all():
            raise AssertionError(f'{name} contains nonfinite values')
    np.testing.assert_array_equal(actual.held, expected.held, err_msg='held bar mismatch')
    np.testing.assert_array_equal(actual.fill_bars, expected.fill_bars,
                                  err_msg='fill bar mismatch')
    np.testing.assert_array_equal(actual.fill_sides, expected.fill_sides,
                                  err_msg='fill side mismatch')
    np.testing.assert_allclose(actual.fill_prices, expected.fill_prices,
                               rtol=1e-12, atol=1e-8, err_msg='fill price mismatch')
    np.testing.assert_allclose(actual.equity, expected.equity,
                               rtol=EQUITY_RTOL, atol=EQUITY_ATOL,
                               err_msg='full equity path mismatch')


def result_summary(result):
    peak = np.maximum.accumulate(np.r_[INITIAL_CASH, result.equity])[1:]
    returns = result.equity / np.r_[INITIAL_CASH, result.equity[:-1]] - 1
    return {
        'final_equity': float(result.equity[-1]),
        'total_return': float(result.equity[-1] / INITIAL_CASH - 1),
        'max_drawdown': float(np.min(result.equity / peak - 1)),
        'mean_bar_return': float(np.mean(returns)),
        'held_fraction': float(np.mean(result.held)),
        'entries': int(np.sum(result.fill_sides == 1)),
        'exits': int(np.sum(result.fill_sides == -1)),
    }


def save_result(path, result):
    np.savez(path, equity=result.equity, held=result.held,
             fill_bars=result.fill_bars, fill_sides=result.fill_sides,
             fill_prices=result.fill_prices)


def load_result(path):
    with np.load(path, allow_pickle=False) as saved:
        return Result(**{name: saved[name] for name in Result.__dataclass_fields__})

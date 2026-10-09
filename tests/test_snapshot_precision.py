from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from limen.backtest import backtest_snapshot
from limen.backtest.long_flat_strategy import long_flat_strategy
from limen.cohort.sfc.top_n import select


@pytest.fixture(scope='module')
def recorded_prices() -> pd.DataFrame:
    return pd.read_parquet(
        Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'
    )


def _columns(prices: pd.DataFrame, signal_bar: int | None) -> dict[str, np.ndarray]:
    predictions = np.zeros(len(prices))
    if signal_bar is not None:
        predictions[signal_bar] = 1
    open_px = prices['open'].to_numpy()
    close_px = prices['close'].to_numpy()
    return {
        'predictions': predictions,
        'open': open_px,
        'close': close_px,
        'price_change': close_px - open_px,
    }


@pytest.mark.parametrize('signal_bar', [0, 1])
@pytest.mark.parametrize('notional_rate', [1.0, 0.1])
@pytest.mark.parametrize(('fee_bps', 'slip_bps'), [(0.0, 0.0), (5.0, 5.0)])
def test_sparse_recorded_rounds_preserve_precision(
    recorded_prices: pd.DataFrame, signal_bar: int, notional_rate: float,
    fee_bps: float, slip_bps: float,
) -> None:
    columns = _columns(recorded_prices, signal_bar)
    ledger = backtest_snapshot(
        columns, fee_bps=fee_bps, slip_bps=slip_bps, notional_rate=notional_rate,
    )
    execution = long_flat_strategy(
        columns['predictions'], columns['open'], columns['close'], columns['price_change'],
        fee_bps=fee_bps, slip_bps=slip_bps,
    )
    net = execution.net * notional_rate
    gross = execution.gross * notional_rate
    assert 0 < abs(ledger['pnl_per_bar_bps']) < 0.05
    assert ledger['pnl_per_bar_bps'] == float(net.mean()) * 10000
    assert ledger['cost_per_bar_bps'] == float((gross - net).mean()) * 10000
    assert ledger['inventory_per_bar'] == float((execution.pos * notional_rate).mean())
    assert ledger['trades_per_bar'] == 1 / len(recorded_prices)
    assert ledger['wins_per_bar'] == np.count_nonzero(net > 0) / len(recorded_prices)
    if fee_bps > 0:
        assert 0 < ledger['cost_per_bar_bps'] < 0.05
    assert np.sign(ledger['pnl_per_bar_bps']) == (1 if signal_bar == 0 else -1)


def test_recorded_distributions_and_tail_preserve_precision(recorded_prices: pd.DataFrame) -> None:
    columns = _columns(recorded_prices, None)
    columns['predictions'][:] = 1
    ledger = backtest_snapshot(columns, fee_bps=5.0, slip_bps=5.0)
    execution = long_flat_strategy(
        columns['predictions'], columns['open'], columns['close'], columns['price_change'],
        fee_bps=5.0, slip_bps=5.0,
    )
    equity = np.cumprod(1 + execution.net)
    drawdown = equity / np.maximum(1, np.maximum.accumulate(equity)) - 1
    for prefix, values in (
        ('edge_bps', execution.gross), ('pnl_bps', execution.net),
        ('cost_bps', execution.gross - execution.net), ('drawdown_bps', drawdown),
    ):
        for suffix, quantile in (('p5', 0.05), ('p50', 0.5), ('p95', 0.95)):
            assert ledger[f'{prefix}_{suffix}'] == float(np.quantile(values * 10000, quantile))
    assert ledger['avg_win_bps'] == float(execution.net[execution.net > 0].mean()) * 10000
    assert ledger['avg_loss_bps'] == float(execution.net[execution.net < 0].mean()) * 10000
    tail = np.sort(execution.net)[:int(len(execution.net) * 0.05)]
    assert ledger['cvar_95_pnl_bps'] == float(tail.mean()) * 10000


def test_sparse_recorded_rounds_rank_after_csv(
    recorded_prices: pd.DataFrame, tmp_path: Path,
) -> None:
    rows = []
    for round_id, signal_bar in enumerate((1, None, 0)):
        ledger = backtest_snapshot(_columns(recorded_prices, signal_bar), fee_bps=0.0, slip_bps=0.0)
        rows.append({'id': round_id, **{f'backtest_{name}': value for name, value in ledger.items()}})
    path = tmp_path / 'results.csv'
    pl.DataFrame(rows).write_csv(path)
    results = pl.read_csv(path)
    assert results['backtest_pnl_per_bar_bps'][0] < 0
    assert results['backtest_pnl_per_bar_bps'][1] == 0
    assert results['backtest_pnl_per_bar_bps'][2] > 0
    assert select({'results': results}, column='backtest_pnl_per_bar_bps', n=3) == [2, 1, 0]

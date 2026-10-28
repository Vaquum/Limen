from pathlib import Path

import numpy as np
import polars as pl
import pytest

from limen.backtest._snapshot_execution import snapshot_with_execution
from limen.backtest.backtest_snapshot import backtest_snapshot
from limen.backtest.long_flat_strategy import long_flat_strategy


@pytest.fixture(scope='module')
def market():
    return pl.read_parquet(Path(__file__).parent / 'fixtures/spot_1h_20240101_20240116.parquet').head(12)


def _execute(market, entries, exits, **options):
    columns = {**market.head(len(entries)).select('open', 'high', 'low', 'close').to_dict(as_series=False),
               'predictions': entries, 'exit': exits}
    columns['price_change'] = np.asarray(columns['close']) - np.asarray(columns['open'])
    return snapshot_with_execution(columns, exit_col='exit', fee_bps=0.0, slip_bps=0.0, **options)


def test_snapshot_entry_holds_until_independent_exit(market):
    _, result = _execute(market, [1, 0, 0, 0, 0, 0], [False, False, False, True, False, False])
    assert result.pos.tolist() == [0, 1, 1, 1, 0, 0]
    assert result.gross[1:4] == pytest.approx(market['close'].to_numpy()[1:4] / market['close'].to_numpy()[:3] - 1)


def test_snapshot_exit_requires_a_post_closure_false_before_reentry(market):
    _, result = _execute(market, [1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
                         [False, False, True, False, False, True, False, False, False, False])
    assert result.pos.tolist() == [0, 1, 1, 0, 0, 1, 0, 0, 0, 1]


def test_snapshot_pending_entry_cancellation_and_rule_lag(market):
    _, result = _execute(market, [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                         [False, True, False, False, False, False, False, True, False, False, False, False],
                         execution_lag_bars=3)
    assert result.pos.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]


def test_snapshot_flat_exit_veto_preserves_initial_arming(market):
    _, result = _execute(market, [1, 1, 1, 0], [True, False, False, False])
    assert result.pos.tolist() == [0, 0, 1, 1]


@pytest.mark.parametrize('lag', [0, -1, True, 1.5])
def test_snapshot_explicit_exit_lag_refusals(market, lag):
    with pytest.raises(ValueError, match='positive integer'):
        _execute(market, [1, 0], [False, False], execution_lag_bars=lag)


def test_snapshot_explicit_exit_evidence_and_price_refusals(market):
    for values in ([0, 1], [False, None], [False]):
        with pytest.raises(ValueError, match='boolean|length'):
            _execute(market, [1, 0], values)
    with pytest.raises(ValueError, match='finite positive prices'):
        _execute(market.with_columns(pl.when(pl.int_range(pl.len()) == 1).then(None).otherwise(pl.col('close')).alias('close')), [1, 0], [False, False])
    columns = {**market.head(2).to_dict(as_series=False), 'predictions': [1, 0], 'exit': [False, False]}
    columns['price_change'] = np.asarray(columns['close']) - np.asarray(columns['open'])
    with pytest.raises(ValueError, match='default long-flat strategy'):
        backtest_snapshot(columns, exit_col='exit', strategy=lambda *args, **kwargs: long_flat_strategy(*args, **kwargs))

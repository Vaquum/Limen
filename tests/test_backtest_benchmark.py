"""Accounting and input-contract checks use recorded prices, never generated bars."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from limen.sfd.rule_based.common_strategies import STRATEGY_NAMES, strategy_signals
from tests.backtest_benchmark.adapters import limen_adapter
from tests.backtest_benchmark.contract import (
    INITIAL_CASH, BatchResult, assert_parity, load_result, reference_result, save_result, targets,
)
from tests.backtest_benchmark.prepare import prepare_scenario, read_bars

FIXTURE = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'


@pytest.fixture(scope='module')
def recorded():
    return read_bars(FIXTURE, rows=600)


def _inputs(recorded, strategy):
    evaluation = recorded.head(recorded.height - 1)
    signal = strategy_signals(evaluation, strategy)
    return {'open': recorded['open'].to_numpy(), 'close': recorded['close'].to_numpy(),
            'signal': targets(np.r_[signal, 0])}


@pytest.mark.parametrize('strategy', STRATEGY_NAMES)
def test_native_execution_matches_independent_cash_replay(recorded, strategy):
    data = _inputs(recorded, strategy)
    adapter = limen_adapter(data)
    actual = adapter.normalize(adapter.execute())
    expected = reference_result(data['close'], data['signal'])
    assert_parity(actual, expected)


def test_buy_hold_enters_first_close_and_liquidates_final_evaluated_close(recorded):
    data = _inputs(recorded, 'buy_and_hold')
    adapter = limen_adapter(data)
    result = adapter.normalize(adapter.execute())
    np.testing.assert_array_equal(result.fill_bars, (0, recorded.height - 2))
    np.testing.assert_array_equal(result.fill_sides, (1, -1))
    np.testing.assert_allclose(result.equity,
                               INITIAL_CASH * data['close'][:-1] / data['close'][0])
    assert result.equity[0] == INITIAL_CASH


def test_parity_checks_the_entire_path_even_when_terminal_equity_matches(recorded):
    data = _inputs(recorded, 'buy_and_hold')
    expected = reference_result(data['close'], data['signal'])
    wrong_path = expected.equity.copy()
    wrong_path[20] = expected.equity[21]
    with pytest.raises(AssertionError, match='full equity path mismatch'):
        assert_parity(replace(expected, equity=wrong_path), expected)


def test_parity_checks_fill_execution_timestamps(recorded):
    data = _inputs(recorded, 'buy_and_hold')
    expected = reference_result(data['close'], data['signal'])
    wrong_bars = expected.fill_bars.copy()
    wrong_bars[-1] -= 1
    with pytest.raises(AssertionError, match='fill bar mismatch'):
        assert_parity(replace(expected, fill_bars=wrong_bars), expected)


def test_result_archives_preserve_full_execution(recorded, tmp_path):
    data = _inputs(recorded, 'sma_cross')
    expected = reference_result(data['close'], data['signal'])
    path = tmp_path / 'execution.npz'
    save_result(path, expected)
    assert_parity(load_result(path), expected)


@pytest.mark.parametrize('malformation', ('reverse', 'duplicate', 'nonfinite', 'null'))
def test_recorded_bar_validation_rejects_invalid_inputs(recorded, tmp_path, malformation):
    if malformation == 'reverse':
        invalid = recorded.reverse()
    elif malformation == 'duplicate':
        invalid = pl.concat((recorded.head(1), recorded))
    elif malformation == 'nonfinite':
        invalid = recorded.with_columns(pl.when(pl.int_range(pl.len()) == 0)
                                        .then(float('inf')).otherwise(pl.col('close')).alias('close'))
    else:
        invalid = recorded.with_columns(pl.when(pl.int_range(pl.len()) == 0)
                                        .then(None).otherwise(pl.col('close')).alias('close'))
    path = tmp_path / f'{malformation}.parquet'
    invalid.write_parquet(path)
    with pytest.raises(ValueError, match=r'strictly increasing|finite|null'):
        read_bars(path)


def test_short_real_input_is_not_tiled_or_padded(recorded, tmp_path):
    path = tmp_path / 'recorded-prefix.parquet'
    recorded.write_parquet(path)
    with pytest.raises(ValueError, match='one real settlement bar'):
        read_bars(path, rows=recorded.height)


def test_fractional_signal_is_not_silently_cast_to_binary(recorded):
    signal = strategy_signals(recorded, 'buy_and_hold').astype(float)
    signal[0] = 0.5
    with pytest.raises(ValueError, match='evaluation states'):
        targets(signal)


def _batch_inputs(recorded):
    inputs = tuple(_inputs(recorded, name) for name in STRATEGY_NAMES)
    return {'open': inputs[0]['open'], 'close': inputs[0]['close'],
            'signal': np.column_stack([item['signal'] for item in inputs]),
            'strategy_names': np.asarray(STRATEGY_NAMES)}


def test_native_batch_has_independent_cash_and_per_column_execution(recorded):
    data = _batch_inputs(recorded)
    adapter = limen_adapter(data)
    actual = adapter.normalize(adapter.execute())
    expected = BatchResult({name: reference_result(data['close'], data['signal'][:, index])
                            for index, name in enumerate(STRATEGY_NAMES)})
    assert_parity(actual, expected)
    assert actual.equity.shape == (recorded.height - 1, len(STRATEGY_NAMES))
    np.testing.assert_array_equal(actual.equity[0], INITIAL_CASH)
    assert adapter.snapshot is None


def test_batch_archives_preserve_strategy_identity(recorded, tmp_path):
    data = _batch_inputs(recorded)
    expected = BatchResult({name: reference_result(data['close'], data['signal'][:, index])
                            for index, name in enumerate(STRATEGY_NAMES)})
    path = tmp_path / 'batch.npz'
    save_result(path, expected)
    actual = load_result(path)
    assert_parity(actual, expected)
    renamed = BatchResult(dict(reversed(tuple(actual.columns.items()))))
    with pytest.raises(AssertionError, match='strategy columns differ'):
        assert_parity(renamed, expected)


def test_prepared_batch_matches_the_same_independent_strategy_signals(recorded, tmp_path):
    path = tmp_path / 'prepared.npz'
    metadata = prepare_scenario(recorded, STRATEGY_NAMES, path, repeats=3)
    expected = _batch_inputs(recorded)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['signal'], expected['signal'])
        np.testing.assert_array_equal(saved['strategy_names'], STRATEGY_NAMES)
        np.testing.assert_array_equal(saved['close'], recorded['close'].to_numpy())
    assert metadata['portfolio_count'] == len(STRATEGY_NAMES)
    assert metadata['evaluation_rows'] == recorded.height - 1

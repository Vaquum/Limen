from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from limen.data.utils import split_sequential
from limen.features.common_strategy_positions import POSITION_COLUMN, common_strategy_positions
from limen.indicators.bollinger_bands import bollinger_bands
from limen.indicators.wilder_rsi import wilder_rsi
from limen.sfd.foundational_sfd import common_strategies as sfd
from limen.sfd.rule_based.common_strategies import STRATEGY_NAMES, STRATEGY_PARAMETERS, strategy_signals
from limen.yaml.compiler import CompiledSFD, build_search_strategy
from limen.yaml.parser import parse
from limen.yaml.validator import validate


_WARMUP_ROWS = {
    'buy_and_hold': 0,
    'sma_cross': 199,
    'ema_cross': 25,
    'macd': 33,
    'rsi_mean_reversion': 14,
    'bollinger_mean_reversion': 19,
    'donchian_breakout': 20,
    'time_series_momentum': 20,
}


@pytest.fixture(scope='module')
def recorded_bars() -> pl.DataFrame:
    return pl.read_parquet(
        Path(__file__).parent / 'fixtures' / 'spot_1h_20240101_20241231.parquet',
        n_rows=3000,
    ).select('datetime', 'open', 'high', 'low', 'close')


@pytest.mark.parametrize('strategy', STRATEGY_NAMES)
def test_signals_are_causal_and_keep_warmup_rows(recorded_bars, strategy):
    original = recorded_bars.clone()
    signals = strategy_signals(recorded_bars, strategy)
    warmup = _WARMUP_ROWS[strategy]

    assert signals.dtype == np.int8
    assert len(signals) == recorded_bars.height
    assert set(np.unique(signals)) <= {0, 1}
    assert not signals[:warmup].any()
    for count in sorted({0, 1, warmup, warmup + 1, 401, 1001}):
        np.testing.assert_array_equal(
            strategy_signals(recorded_bars.head(count), strategy), signals[:count],
        )
    assert_frame_equal(recorded_bars, original)


def test_passive_and_momentum_rules_use_completed_prices(recorded_bars):
    np.testing.assert_array_equal(
        strategy_signals(recorded_bars, 'buy_and_hold'),
        np.ones(recorded_bars.height, dtype=np.int8),
    )
    prices = recorded_bars['close'].to_numpy()
    expected = np.zeros(len(prices), dtype=np.int8)
    expected[20:] = prices[20:] > prices[:-20]
    np.testing.assert_array_equal(strategy_signals(recorded_bars, 'time_series_momentum'), expected)


def test_sma_cross_matches_independent_rolling_means(recorded_bars):
    prices = recorded_bars['close'].to_numpy()
    expected = np.zeros(len(prices), dtype=np.int8)
    for row in range(199, len(prices)):
        expected[row] = prices[row - 49:row + 1].mean() > prices[row - 199:row + 1].mean()
    np.testing.assert_array_equal(strategy_signals(recorded_bars, 'sma_cross'), expected)


def test_ema_cross_matches_independent_seeded_recurrence(recorded_bars):
    prices = recorded_bars['close'].to_numpy()
    expected = np.zeros(len(prices), dtype=np.int8)
    fast = float(prices[:12].mean())
    slow = float(prices[:26].mean())
    for row in range(12, len(prices)):
        fast = fast * (1.0 - 2.0 / 13.0) + prices[row] * (2.0 / 13.0)
        if row >= 26:
            slow = slow * (1.0 - 2.0 / 27.0) + prices[row] * (2.0 / 27.0)
        if row >= 25:
            expected[row] = fast > slow
    np.testing.assert_array_equal(strategy_signals(recorded_bars, 'ema_cross'), expected)


@pytest.mark.parametrize('strategy', ['rsi_mean_reversion', 'bollinger_mean_reversion'])
def test_mean_reversion_holds_between_entry_and_exit_thresholds(recorded_bars, strategy):
    if strategy == 'rsi_mean_reversion':
        rsi = wilder_rsi(recorded_bars, period=14)['wilder_rsi_14'].to_numpy()
        entries, exits = rsi <= 30.0, rsi >= 70.0
    else:
        bands = bollinger_bands(recorded_bars, window=20, num_std=2.0)
        close = bands['close'].to_numpy()
        entries = close <= bands['bb_lower'].to_numpy()
        exits = close >= bands['bb_middle'].to_numpy()
    positions = strategy_signals(recorded_bars, strategy)
    previous = np.concatenate(([0], positions[:-1]))
    neutral = ~entries & ~exits

    assert np.any(entries)
    assert np.any(exits)
    assert np.any(neutral & (positions == 1)), 'The recorded sample must exercise multi-bar holding'
    np.testing.assert_array_equal(positions[neutral], previous[neutral])
    assert np.all(positions[entries & ~exits] == 1)
    assert np.all(positions[exits] == 0)


def test_donchian_uses_prior_channels_and_holds_until_exit(recorded_bars):
    prices = recorded_bars['close'].to_numpy()
    highs = recorded_bars['high'].to_numpy()
    lows = recorded_bars['low'].to_numpy()
    expected = np.zeros(len(prices), dtype=np.int8)
    for row in range(20, len(prices)):
        if prices[row] < min(lows[row - 10:row]):
            expected[row] = 0
        elif prices[row] > max(highs[row - 20:row]):
            expected[row] = 1
        else:
            expected[row] = expected[row - 1]
    assert expected.any(), 'The recorded sample must include breakouts'
    np.testing.assert_array_equal(strategy_signals(recorded_bars, 'donchian_breakout'), expected)


@pytest.mark.parametrize('strategy', ['rsi_mean_reversion', 'donchian_breakout'])
def test_lazy_feature_keeps_history_before_filters_slices_and_projection(recorded_bars, strategy):
    expected = common_strategy_positions(recorded_bars, strategy)
    lazy = common_strategy_positions(recorded_bars.lazy(), strategy)
    assert isinstance(expected, pl.DataFrame)
    assert isinstance(lazy, pl.LazyFrame)
    assert expected.columns == [*recorded_bars.columns, POSITION_COLUMN]
    assert expected[POSITION_COLUMN].dtype == pl.Int8
    assert_frame_equal(lazy.collect(), expected)
    assert_frame_equal(lazy.slice(1500, 300).collect(), expected.slice(1500, 300))
    cutoff = recorded_bars['datetime'][1500]
    assert_frame_equal(
        lazy.filter(pl.col('datetime') >= cutoff).select(POSITION_COLUMN).collect(),
        expected.filter(pl.col('datetime') >= cutoff).select(POSITION_COLUMN),
    )


def test_native_manifest_evaluates_all_strategies_with_independent_split_history(recorded_bars):
    manifest = sfd.manifest()
    raw_splits = split_sequential(recorded_bars, (8, 1, 2))
    assert sfd.params() == {'strategy': list(STRATEGY_NAMES)}
    assert set(STRATEGY_PARAMETERS) == set(_WARMUP_ROWS)
    assert manifest.data_source_config is not None
    assert manifest.test_data_source_config is not None
    assert manifest.data_source_config.params == {
        'kline_size': 3600,
        'start_date_limit': '2024-01-01',
        'end_date_limit': '2024-12-31',
    }
    assert manifest.test_data_source_config.params == manifest.data_source_config.params

    for strategy in STRATEGY_NAMES:
        prepared = manifest.prepare_data(recorded_bars, {'strategy': strategy})
        for split_name, raw in zip(('train', 'val', 'test'), raw_splits, strict=True):
            np.testing.assert_array_equal(
                prepared[split_name][POSITION_COLUMN].to_numpy(), strategy_signals(raw, strategy),
            )
        result = manifest.run_model(prepared, {'strategy': strategy})
        np.testing.assert_array_equal(result['_preds'], strategy_signals(raw_splits[2], strategy))
        assert 'pnl_per_trade_bps_test' in result
        assert result['num_executed_trades_test'] >= 0
        assert prepared['backtest_fee_bps'] == 10.0
        assert prepared['backtest_slip_bps'] == 5.0


def test_signal_contract_rejects_unknown_missing_nonnumeric_and_unsorted_input(recorded_bars):
    with pytest.raises(ValueError, match='Unknown strategy'):
        strategy_signals(recorded_bars, 'unknown')
    with pytest.raises(ValueError, match='missing required price columns'):
        strategy_signals(recorded_bars.drop('high'), 'donchian_breakout')
    with pytest.raises(TypeError, match='close must be numeric'):
        strategy_signals(recorded_bars.with_columns(pl.col('close').cast(pl.String)), 'buy_and_hold')
    with pytest.raises(ValueError, match='strictly increasing'):
        strategy_signals(recorded_bars.reverse(), 'buy_and_hold')
    with pytest.raises(ValueError, match='low <= close <= high'):
        strategy_signals(recorded_bars.rename({'high': 'low', 'low': 'high'}), 'donchian_breakout')


def test_feature_rejects_output_collision(recorded_bars):
    prepared = common_strategy_positions(recorded_bars, 'buy_and_hold')
    with pytest.raises(ValueError, match='output column already exists'):
        common_strategy_positions(prepared, 'buy_and_hold')


def test_yaml_template_compiles_the_complete_fixed_catalogue():
    template = Path(__file__).resolve().parents[1] / 'limen/yaml/templates/common_strategies.yaml'
    config, errors = parse(template.read_text(encoding='utf-8'))
    assert not errors
    validation = validate(config)
    assert validation.valid, [error.message for error in validation.errors]
    compiled = CompiledSFD(config)
    assert compiled.params() == sfd.params()
    compiled_manifest = compiled.manifest()
    assert compiled_manifest.required_bar_columns == ['datetime', 'open', 'high', 'low', 'close']
    assert compiled_manifest.resolve_backtest_config({})['fee_bps'] == 10.0
    candidates = list(build_search_strategy(config))
    assert [candidate['strategy'] for candidate in candidates] == list(STRATEGY_NAMES)
    assert len({candidate['_id'] for candidate in candidates}) == len(STRATEGY_NAMES)

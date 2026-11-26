from importlib import import_module
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from limen import features
from limen.features import _filang_engine
from limen.features.filang_pattern import filang_pattern

FIXTURE = Path(__file__).parent / 'fixtures/spot_15m_20250101_20250531.parquet'
SHAPE = 'g1:o:0,1,3,4,u;2,3,5,6,u'
COUNTERS = ['liquidity_sum', 'maker_liquidity', 'no_of_trades']


@pytest.fixture(scope='module')
def bars() -> pl.DataFrame:
    return pl.read_parquet(FIXTURE)


def _events(frame: pl.DataFrame, definition: str, **arguments: object) -> np.ndarray:
    return filang_pattern(frame, definition, **arguments)['filang_event'].to_numpy()


# Filang publishes these counts for this recording in its own verification: the bars of Filang's pinned study
# recording are this fixture's bars. They are the engine's answers, not numbers produced by this feature.
@pytest.mark.parametrize('definition,count', [
    (f'chained_g1("{SHAPE}")', 1463),
    (f'exact_g1("{SHAPE}")', 772),
    ('exact_g1("g1:o:0,1,2,2,u")', 324),
    ('wick_rejection(body_max: 35%, wick_min: 45%)', 2077),
])
def test_counts_match_the_answers_filang_publishes_for_these_bars(bars, definition, count):
    result = filang_pattern(bars, definition)
    assert result.columns == [*bars.columns, 'filang_event', 'filang_unknown', 'filang_position']
    assert result.height == bars.height
    assert int(result['filang_event'].sum()) == count
    assert result.select(pl.col('^filang_.*$').null_count().sum()).row(0) == (0, 0, 0)


def test_readings_of_the_counters_equal_plain_arithmetic_on_the_columns(bars):
    traded = bars['liquidity_sum'].to_numpy()
    taker_buy = traded - bars['maker_liquidity'].to_numpy()
    trades = bars['no_of_trades'].to_numpy().astype(float)
    delta = 2 * taker_buy - traded
    rising = np.r_[False, traded[1:] > traded[:-1]]
    np.testing.assert_array_equal(_events(bars, 'candle_relation(delta(), is: above, right: 0)') == 1, delta > 0)
    np.testing.assert_array_equal(
        _events(bars, 'candle_relation(volume(), is: above, right: volume(bars_ago: 1))') == 1, rising)
    size = traded / trades
    np.testing.assert_array_equal(
        _events(bars, 'candle_relation(size(), is: above, right: size(bars_ago: 1))') == 1, np.r_[False, size[1:] > size[:-1]])
    refined = sum(int(_events(bars, f'chained_g1("{SHAPE}|volume:{part}")').sum()) for part in ('0,1', '1,0', '0,0'))
    assert refined == 1463


def test_a_condition_narrows_the_event_and_unknown_is_kept_apart_from_false(bars):
    event = filang_pattern(bars, 'channel_breakout(over: 1d)')
    narrowed = filang_pattern(bars, 'channel_breakout(over: 1d) where candle_relation(delta(), is: above, right: 0)')
    buying = _events(bars, 'candle_relation(delta(), is: above, right: 0)') == 1
    breakout = event['filang_event'].to_numpy() == 1
    np.testing.assert_array_equal(narrowed['filang_event'].to_numpy() == 1, breakout & buying)
    # A one-day lookback is incomplete on the first day of the recording. Filang answers unknown there only where
    # the bars it has cannot settle the question, never a match, and it is never unknown once the day is complete.
    assert event['filang_unknown'].head(96).sum() > 0
    assert event['filang_event'].head(96).sum() == 0
    assert event['filang_unknown'].tail(bars.height - 96).sum() == 0
    assert int((event['filang_event'] * event['filang_unknown']).sum()) == 0
    # Where the event is false the narrowed pattern is false, even at closes the condition could not decide.
    assert int(narrowed['filang_unknown'].sum()) <= int(event['filang_unknown'].sum())


def test_position_holds_from_each_match_and_a_later_match_extends_it(bars):
    result = filang_pattern(bars, 'hammer()', hold_bars=4)
    event = result['filang_event'].to_numpy()
    expected = np.zeros(bars.height, dtype=np.int8)
    for start in np.flatnonzero(event == 1):
        expected[start:start + 4] = 1
    np.testing.assert_array_equal(result['filang_position'].to_numpy(), expected)
    assert int(expected.sum()) > int(event.sum()) > 0
    np.testing.assert_array_equal(filang_pattern(bars, 'hammer()')['filang_position'].to_numpy(), event)


def test_lazy_frames_and_several_patterns_side_by_side(bars):
    eager = filang_pattern(filang_pattern(bars, 'hammer()', prefix='entry'), 'bearish_engulfing()', prefix='exit')
    lazy = filang_pattern(filang_pattern(bars.lazy(), 'hammer()', prefix='entry'), 'bearish_engulfing()', prefix='exit')
    assert isinstance(lazy, pl.LazyFrame)
    assert_frame_equal(lazy.collect(), eager)
    assert eager.columns[-6:] == ['entry_event', 'entry_unknown', 'entry_position', 'exit_event', 'exit_unknown', 'exit_position']
    assert features.filang_pattern is filang_pattern
    empty = filang_pattern(bars.head(0), 'hammer()')
    assert empty.height == 0 and empty.columns[-3:] == ['filang_event', 'filang_unknown', 'filang_position']


def test_other_bar_sizes_and_frames_without_counters(bars):
    hourly = bars.group_by_dynamic('datetime', every='1h').agg(
        pl.col('open').first(), pl.col('high').max(), pl.col('low').min(), pl.col('close').last(), pl.col('volume').sum())
    result = filang_pattern(hourly, 'hammer()', interval_seconds=3600)
    assert result.height == hourly.height and 0 < int(result['filang_event'].sum()) < hourly.height
    prices = bars.drop(COUNTERS)
    np.testing.assert_array_equal(_events(prices, 'hammer()'), _events(bars, 'hammer()'))
    with pytest.raises(ValueError, match='no counters'):
        filang_pattern(prices, 'candle_relation(delta(), is: above, right: 0)')


@pytest.mark.parametrize('column', ['open', 'high', 'low', 'close'])
def test_prices_that_would_be_rounded_are_rejected(bars, column):
    prices = bars.head(2).with_columns((pl.col(column) + 0.001).alias(column))
    with pytest.raises(ValueError, match='OHLC prices must be exactly representable with two decimal places'):
        filang_pattern(prices, 'hammer()')


def test_answer_cache_is_fifo_bounded_and_evicted_parts_are_recomputed(bars):
    recording = _filang_engine._Recording(bars.head(4), 900)
    part = 'candle_relation(volume(), is: above, right: volume(bars_ago: 1))'
    try:
        first = recording.ask(part, condition=False)
        np.testing.assert_array_equal(recording.ask(part, condition=True), first)
        assert len(recording._answers) == 2
        assert recording.ask(part, condition=False) is first
        for lag in range(2, _filang_engine._KEEP_ANSWERS + 1):
            recording.ask(f'candle_relation(volume(), is: above, right: volume(bars_ago: {lag}))', condition=bool(lag % 2))
            assert len(recording._answers) <= _filang_engine._KEEP_ANSWERS
        assert len(recording._answers) == _filang_engine._KEEP_ANSWERS
        assert f'find {part}' not in recording._answers
        assert f'where {part}' in recording._answers
        recomputed = recording.ask(part, condition=False)
        assert recomputed is not first
        np.testing.assert_array_equal(recomputed, first)
        assert len(recording._answers) == _filang_engine._KEEP_ANSWERS
        assert f'where {part}' not in recording._answers
    finally:
        recording.engine.close()


@pytest.mark.parametrize('definition,message', [
    ('not_a_filang_event()', 'not_a_filang_event'),
    ('hammer(', 'definition is not a pattern Filang can answer'),
    ('wick_rejection(body_max: {25% | 35%}, wick_min: 45%)', 'must be one pattern'),
])
def test_patterns_filang_cannot_answer_are_refused_with_its_reason(bars, definition, message):
    with pytest.raises(ValueError, match=message):
        filang_pattern(bars, definition)


def test_invalid_data_and_arguments_fail_before_the_engine_is_asked(bars, monkeypatch):
    def unavailable(*_: object) -> object:
        raise AssertionError('the engine was asked')
    monkeypatch.setattr(import_module('limen.features.filang_pattern'), 'pattern_answers', unavailable)
    with pytest.raises(ValueError, match="missing columns: \\['close'\\]"):
        filang_pattern(bars.drop('close'), 'hammer()')
    with pytest.raises(ValueError, match='already exist'):
        filang_pattern(bars.with_columns(pl.lit(0).alias('filang_event')), 'hammer()')
    with pytest.raises(ValueError, match='interval_seconds must be a whole number of minutes'):
        filang_pattern(bars, 'hammer()', interval_seconds=90)
    with pytest.raises(ValueError, match='definition'):
        filang_pattern(bars, '   ')


def test_a_missing_engine_says_how_to_install_it(bars, monkeypatch):
    def missing(module: str, package: str, extra: str) -> object:
        raise ImportError(f'{package} is required for this Limen surface. Install it with `pip install vaquum-limen[{extra}]`.')
    monkeypatch.setattr(_filang_engine, 'require_optional', missing)
    monkeypatch.setattr(_filang_engine, '_RECORDINGS', {})
    with pytest.raises(ImportError, match=r'filang is required.*vaquum-limen\[filang\]'):
        filang_pattern(bars, 'hammer()')

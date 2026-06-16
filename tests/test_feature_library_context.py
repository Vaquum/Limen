import numpy as np
import polars as pl
import pytest

from limen.features.active_lines import active_lines
from limen.features.active_quantile_count import active_quantile_count
from limen.features.hours_since_big_move import hours_since_big_move
from limen.features.hours_since_quantile_line import hours_since_quantile_line
from limen.features.market_regime import market_regime
from limen.features.price_lines import price_lines
from limen.features.quantile_line_density import quantile_line_density
from limen.features.quantile_price_lines import quantile_price_lines
from limen.utils.filter_lines_by_quantile import filter_lines_by_quantile
from limen.utils.find_price_lines import find_price_lines


def test_active_lines_counts_overlapping_long_and_short_spans() -> None:
    data = pl.DataFrame({
        'datetime': list(range(6)),
        'close': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    })

    result = active_lines(
        data,
        long_lines=[{'start_idx': 1, 'end_idx': 3}],
        short_lines=[{'start_idx': 2, 'end_idx': 4}],
    )

    assert result['active_lines'].to_list() == [0, 1, 2, 2, 1, 0]


def test_active_lines_returns_zero_when_no_lines_or_rows_exist() -> None:
    data = pl.DataFrame({'datetime': [1, 2, 3], 'close': [10.0, 11.0, 12.0]})
    empty = pl.DataFrame({
        'datetime': pl.Series('datetime', [], dtype=pl.Int64),
        'close': pl.Series('close', [], dtype=pl.Float64),
    })

    no_lines = active_lines(data, long_lines=[], short_lines=[])
    empty_result = active_lines(empty, long_lines=[], short_lines=[])

    assert no_lines['active_lines'].to_list() == [0, 0, 0]
    assert empty_result.columns == ['datetime', 'close', 'active_lines']
    assert empty_result.height == 0


def test_active_quantile_count_tracks_quantile_filtered_spans() -> None:
    data = pl.DataFrame({
        'datetime': list(range(6)),
        'close': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    })

    result = active_quantile_count(
        data,
        long_lines_q=[{'start_idx': 1, 'end_idx': 3}],
        short_lines_q=[{'start_idx': 2, 'end_idx': 4}],
    )

    assert result['active_quantile_count'].to_list() == [0, 1, 2, 2, 1, 0]


def test_quantile_line_density_counts_recent_line_endings_within_lookback() -> None:
    data = pl.DataFrame({
        'datetime': list(range(6)),
        'close': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    })

    result = quantile_line_density(
        data,
        long_lines_q=[{'end_idx': 1}, {'end_idx': 4}],
        short_lines_q=[{'end_idx': 2}],
        lookback_hours=2,
    )

    density_columns = [
        column_name for column_name in result.columns if column_name.startswith('quantile_line_density_')
    ]

    assert density_columns == ['quantile_line_density_2h']
    assert result[density_columns[0]].to_list() == [0, 1, 2, 2, 2, 1]


def test_hours_since_big_move_resets_at_line_end_and_caps_at_lookback() -> None:
    data = pl.DataFrame({
        'datetime': list(range(6)),
        'close': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    })

    result = hours_since_big_move(
        data,
        long_lines=[{'start_idx': 0, 'end_idx': 1}],
        short_lines=[],
        lookback_hours=2,
    )

    assert result['hours_since_big_move'].to_list() == [2.0, 0.0, 1.0, 2.0, 2.0, 2.0]


def test_hours_since_big_move_returns_lookback_without_lines_or_rows() -> None:
    data = pl.DataFrame({'datetime': [1, 2, 3], 'close': [10.0, 11.0, 12.0]})
    empty = pl.DataFrame({
        'datetime': pl.Series('datetime', [], dtype=pl.Int64),
        'close': pl.Series('close', [], dtype=pl.Float64),
    })

    no_lines = hours_since_big_move(data, long_lines=[], short_lines=[], lookback_hours=5)
    empty_result = hours_since_big_move(empty, long_lines=[], short_lines=[], lookback_hours=5)

    assert no_lines['hours_since_big_move'].to_list() == [5.0, 5.0, 5.0]
    assert empty_result.columns == ['datetime', 'close', 'hours_since_big_move']
    assert empty_result.height == 0


def test_hours_since_quantile_line_resets_at_each_end_and_caps_at_lookback() -> None:
    data = pl.DataFrame({
        'datetime': list(range(7)),
        'close': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
    })

    result = hours_since_quantile_line(
        data,
        long_lines_q=[{'end_idx': 1}],
        short_lines_q=[{'end_idx': 3}],
        lookback_hours=2,
    )

    assert result['hours_since_quantile_line'].to_list() == [2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 2.0]


def test_hours_since_quantile_line_returns_lookback_without_lines_or_rows() -> None:
    data = pl.DataFrame({'datetime': [1, 2, 3], 'close': [10.0, 11.0, 12.0]})
    empty = pl.DataFrame({
        'datetime': pl.Series('datetime', [], dtype=pl.Int64),
        'close': pl.Series('close', [], dtype=pl.Float64),
    })

    no_lines = hours_since_quantile_line(data, long_lines_q=[], short_lines_q=[], lookback_hours=4)
    empty_result = hours_since_quantile_line(empty, long_lines_q=[], short_lines_q=[], lookback_hours=4)

    assert no_lines['hours_since_quantile_line'].to_list() == [4.0, 4.0, 4.0]
    assert empty_result.columns == ['datetime', 'close', 'hours_since_quantile_line']
    assert empty_result.height == 0


def test_market_regime_keeps_parameterized_smas_and_adds_normalized_aliases() -> None:
    data = pl.DataFrame({
        'close': [100.0 + idx for idx in range(80)],
        'volume': ([100.0] * 40) + ([200.0] * 40),
    })

    result = market_regime(
        data,
        lookback=12,
        short_sma=3,
        long_sma=5,
    )

    assert 'returns_temp' not in result.columns
    assert {
        'sma_3',
        'sma_5',
        'sma_12',
        'sma_20',
        'sma_50',
        'trend_strength',
        'volatility_ratio',
        'volume_sma',
        'volume_regime',
        'market_favorable',
    }.issubset(result.columns)
    trend_tail = result['trend_strength'].drop_nulls().tail(5).to_list()
    assert all(value > 0 for value in trend_tail)
    assert trend_tail == sorted(trend_tail, reverse=True)
    assert result['market_favorable'].drop_nulls().tail(5).to_list() == pytest.approx(
        [1.0, 1.0, 1.0, 1.0, 1.0]
    )


def _spike_close() -> list[float]:
    closes = [100.0] * 60
    closes[10] = 103.0
    return closes


def test_find_price_lines_matches_archive_definition() -> None:
    rng = np.random.default_rng(7)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, 120))

    long_lines, short_lines = find_price_lines(close, max_duration_hours=6, min_height_pct=0.01)

    expected_long = []
    expected_short = []
    for start in range(len(close)):
        for end in range(start + 1, min(start + 6, len(close))):
            height = (close[end] - close[start]) / close[start]
            if abs(height) >= 0.01:
                (expected_long if height > 0 else expected_short).append(
                    (start, end, pytest.approx(height))
                )

    got_long = sorted((line['start_idx'], line['end_idx'], line['height_pct']) for line in long_lines)
    got_short = sorted((line['start_idx'], line['end_idx'], line['height_pct']) for line in short_lines)
    assert got_long == sorted(expected_long, key=lambda t: (t[0], t[1]))
    assert got_short == sorted(expected_short, key=lambda t: (t[0], t[1]))
    assert all(1 <= line['duration_hours'] < 6 for line in long_lines + short_lines)


def test_find_price_lines_rejects_invalid_params_and_handles_short_input() -> None:
    with pytest.raises(ValueError, match='find_price_lines max_duration_hours'):
        find_price_lines(np.array([100.0, 101.0]), max_duration_hours=1, min_height_pct=0.01)

    with pytest.raises(ValueError, match='find_price_lines min_height_pct'):
        find_price_lines(np.array([100.0, 101.0]), max_duration_hours=48, min_height_pct=0.0)

    assert find_price_lines(np.array([100.0]), max_duration_hours=48, min_height_pct=0.01) == ([], [])


def test_filter_lines_by_quantile_keeps_top_heights() -> None:
    lines = [{'height_pct': h} for h in (0.01, -0.02, 0.03, -0.04)]

    kept = filter_lines_by_quantile(lines, 0.75)

    assert [line['height_pct'] for line in kept] == [-0.04]
    assert filter_lines_by_quantile([], 0.75) == []

    with pytest.raises(ValueError, match='filter_lines_by_quantile quantile'):
        filter_lines_by_quantile(lines, 1.5)


def test_price_lines_adds_five_line_columns_from_scalar_params() -> None:
    data = pl.DataFrame({'close': _spike_close()})

    result = price_lines(data, max_duration_hours=4, min_height_pct=0.02)

    assert {
        'active_lines',
        'hours_since_big_move',
        'line_momentum_6h',
        'trending_score',
        'reversal_potential',
    }.issubset(result.columns)

    momentum = result['line_momentum_6h'].to_list()
    assert momentum[10:15] == [0.0, 3.0, 2.0, 1.0, 0.0]
    assert result['trending_score'][11] == pytest.approx(1.0)
    assert result['reversal_potential'][11] == pytest.approx(0.0)

    lazy_result = price_lines(data.lazy(), max_duration_hours=4, min_height_pct=0.02).collect()
    assert result.equals(lazy_result)

    live_safe = price_lines(
        data, max_duration_hours=4, min_height_pct=0.02, include_research_only=False
    )
    assert 'active_lines' not in live_safe.columns
    assert {
        'hours_since_big_move',
        'line_momentum_6h',
        'trending_score',
        'reversal_potential',
    }.issubset(live_safe.columns)

    with pytest.raises(ValueError, match='price_lines momentum_lookback_hours'):
        price_lines(data, max_duration_hours=4, min_height_pct=0.02, momentum_lookback_hours=0)

    with pytest.raises(ValueError, match='include_research_only'):
        price_lines(data, max_duration_hours=4, min_height_pct=0.02, include_research_only='false')


def test_quantile_price_lines_adds_six_columns_from_scalar_params() -> None:
    data = pl.DataFrame({'close': _spike_close()})

    result = quantile_price_lines(
        data, max_duration_hours=4, min_height_pct=0.02, quantile_threshold=0.0
    )

    assert {
        'hours_since_quantile_line',
        'active_quantile_count',
        'quantile_line_density_48h',
        'quantile_momentum_6h',
        'avg_quantile_height_24h',
        'quantile_direction_bias',
    }.issubset(result.columns)

    momentum = result['quantile_momentum_6h'].to_list()
    assert momentum[10] == pytest.approx(3 * (103.0 / 100.0 - 1.0))
    assert momentum[11] == pytest.approx(3 * 0.03 - (3.0 / 103.0))

    lazy_result = quantile_price_lines(
        data.lazy(), max_duration_hours=4, min_height_pct=0.02, quantile_threshold=0.0
    ).collect()
    assert result.equals(lazy_result)

    live_safe = quantile_price_lines(
        data,
        max_duration_hours=4,
        min_height_pct=0.02,
        quantile_threshold=0.0,
        include_research_only=False,
    )
    assert 'active_quantile_count' not in live_safe.columns
    assert {
        'hours_since_quantile_line',
        'quantile_line_density_48h',
        'quantile_momentum_6h',
        'avg_quantile_height_24h',
        'quantile_direction_bias',
    }.issubset(live_safe.columns)

    no_lines = quantile_price_lines(
        data, max_duration_hours=4, min_height_pct=0.5, quantile_threshold=0.75
    )
    assert no_lines['hours_since_quantile_line'].to_list() == [48.0] * 60
    assert no_lines['active_quantile_count'].sum() == 0
    assert no_lines['quantile_line_density_48h'].sum() == 0
    assert no_lines['quantile_momentum_6h'].sum() == 0.0
    assert no_lines['avg_quantile_height_24h'].sum() == 0.0
    assert no_lines['quantile_direction_bias'].sum() == 0.0

    with pytest.raises(ValueError, match='include_research_only'):
        quantile_price_lines(
            data,
            max_duration_hours=4,
            min_height_pct=0.02,
            quantile_threshold=0.0,
            include_research_only='false',
        )


def test_price_line_momentum_windows_pin_trailing_edges() -> None:
    data = pl.DataFrame({'close': _spike_close()})

    exclusive = price_lines(
        data, max_duration_hours=4, min_height_pct=0.02
    )['line_momentum_6h'].to_list()

    assert exclusive[16] == 0.0
    assert exclusive[17] == -3.0
    assert exclusive[20] == 0.0

    inclusive = quantile_price_lines(
        data, max_duration_hours=4, min_height_pct=0.02, quantile_threshold=0.0
    )['quantile_momentum_6h'].to_list()

    short_height = 3.0 / 103.0
    assert inclusive[16] == pytest.approx(3 * 0.03 - 3 * short_height)
    assert inclusive[17] == pytest.approx(-3 * short_height)
    assert inclusive[19] == pytest.approx(-short_height)
    assert inclusive[20] == 0.0

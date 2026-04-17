import polars as pl
import pytest

from limen.features.active_lines import active_lines
from limen.features.active_quantile_count import active_quantile_count
from limen.features.market_regime import market_regime
from limen.features.quantile_line_density import quantile_line_density


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

    assert result['active_quantile_count'].to_list() == [0, 1, 2, 1, 0, 0]


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

    assert density_columns == ['quantile_line_density_48h']
    assert result[density_columns[0]].to_list() == [0, 1, 2, 2, 2, 1]


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

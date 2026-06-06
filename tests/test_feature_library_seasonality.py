from datetime import datetime

import polars as pl
import pytest

from limen.features.calendar_time_features import calendar_time_features
from limen.features.is_funding_hour import is_funding_hour
from limen.features.is_us_open_hour import is_us_open_hour
from limen.features.relative_range_seasonality import relative_range_seasonality
from limen.features.relative_volatility_seasonality import relative_volatility_seasonality
from limen.features.relative_volume_seasonality import relative_volume_seasonality
from limen.features.time_to_funding import time_to_funding


def test_calendar_session_features_add_half_year_and_hour_flags() -> None:
    data = pl.DataFrame(
        {
            'datetime': [
                datetime(2026, 1, 1, 0, 30),
                datetime(2026, 6, 1, 8, 30),
                datetime(2026, 7, 1, 14, 30),
                datetime(2026, 12, 1, 16, 30),
            ]
        }
    )

    calendar = calendar_time_features(data)
    funding = is_funding_hour(data, hours=(0, 8, 16))
    us_open = is_us_open_hour(data, hour=14)

    assert calendar['half_of_year'].to_list() == [1, 1, 2, 2]
    assert funding['is_funding_hour'].to_list() == [1, 1, 0, 1]
    assert us_open['is_us_open_hour'].to_list() == [0, 0, 1, 0]


def test_relative_volume_and_range_seasonality_compare_against_same_hour_of_week_history() -> None:
    close = [100.0, 110.0, 121.0, 133.1, 146.41, 175.692]
    range_pct = [0.1, 0.2, 0.2, 0.12, 0.05, 0.24]
    high = [value * (1.0 + (pct / 2.0)) for value, pct in zip(close, range_pct, strict=True)]
    low = [value * (1.0 - (pct / 2.0)) for value, pct in zip(close, range_pct, strict=True)]

    data = pl.DataFrame(
        {
            'datetime': [
                datetime(2026, 1, 2, 10, 0),
                datetime(2026, 1, 2, 11, 0),
                datetime(2026, 1, 9, 10, 0),
                datetime(2026, 1, 9, 11, 0),
                datetime(2026, 1, 16, 10, 0),
                datetime(2026, 1, 16, 11, 0),
            ],
            'close': close,
            'high': high,
            'low': low,
            'volume': [100.0, 200.0, 150.0, 100.0, 50.0, 300.0],
        }
    )

    volume_result = relative_volume_seasonality(data)
    range_result = relative_range_seasonality(data)

    assert volume_result['relative_volume_seasonality'].to_list() == [
        None,
        None,
        pytest.approx(1.5),
        pytest.approx(0.5),
        pytest.approx(0.4),
        pytest.approx(2.0),
    ]
    assert range_result['relative_range_seasonality'].to_list() == [
        None,
        None,
        pytest.approx(2.0),
        pytest.approx(0.6),
        pytest.approx(1.0 / 3.0),
        pytest.approx(1.5),
    ]


def test_relative_volatility_seasonality_uses_same_hour_of_week_absolute_returns() -> None:
    data = pl.DataFrame(
        {
            'datetime': [
                datetime(2026, 1, 2, 10, 0),
                datetime(2026, 1, 2, 11, 0),
                datetime(2026, 1, 9, 10, 0),
                datetime(2026, 1, 9, 11, 0),
                datetime(2026, 1, 16, 10, 0),
                datetime(2026, 1, 16, 11, 0),
            ],
            'close': [100.0, 110.0, 121.0, 133.1, 146.41, 175.692],
        }
    )

    result = relative_volatility_seasonality(data)

    assert result['relative_volatility_seasonality'].to_list() == [
        None,
        None,
        None,
        pytest.approx(1.0),
        pytest.approx(1.0),
        pytest.approx(2.0),
    ]


def test_time_to_funding_counts_hours_to_next_settlement() -> None:
    data = pl.DataFrame(
        {
            'datetime': pl.datetime_range(
                pl.datetime(2025, 1, 1, 0),
                pl.datetime(2025, 1, 1, 10),
                interval='2h',
                eager=True,
            )
        }
    )

    result = time_to_funding(data)['hours_to_funding'].to_list()

    assert result == pytest.approx([0.0, 6.0, 4.0, 2.0, 0.0, 6.0])


def test_time_to_funding_respects_interval_and_validates_positive() -> None:
    data = pl.DataFrame(
        {
            'datetime': pl.datetime_range(
                pl.datetime(2025, 1, 1, 0),
                pl.datetime(2025, 1, 1, 3),
                interval='1h',
                eager=True,
            )
        }
    )

    result = time_to_funding(data, interval_hours=4)['hours_to_funding'].to_list()

    assert result == pytest.approx([0.0, 3.0, 2.0, 1.0])
    with pytest.raises(ValueError, match='interval_hours'):
        time_to_funding(data, interval_hours=0)
    with pytest.raises(ValueError, match='divide 24'):
        time_to_funding(data, interval_hours=5)


def test_time_to_funding_uses_utc_for_timezone_aware_timestamps() -> None:
    data = pl.DataFrame({'datetime': [datetime(2025, 1, 1, 22, 0)]}).with_columns(
        pl.col('datetime').dt.replace_time_zone('UTC').dt.convert_time_zone('Asia/Karachi')
    )

    result = time_to_funding(data)['hours_to_funding'].to_list()

    assert result == pytest.approx([2.0])

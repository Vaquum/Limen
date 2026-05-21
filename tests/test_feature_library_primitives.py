import math

import polars as pl
import pytest

from limen.features import close_position as exported_close_position
from limen.features.atr_percent_sma import atr_percent_sma
from limen.features.atr_sma import atr_sma
from limen.features.close_ma_distance_atr import close_ma_distance_atr
from limen.features.close_position import close_position
from limen.features.close_position_rolling import close_position_rolling
from limen.features.distance_from_high import distance_from_high
from limen.features.distance_from_low import distance_from_low
from limen.features.distance_from_ma import distance_from_ma
from limen.features.gap_high import gap_high
from limen.features.ichimoku_cloud import ichimoku_cloud
from limen.features.kaufman_efficiency_ratio import kaufman_efficiency_ratio
from limen.features.price_range_position import price_range_position
from limen.features.rolling_zscore import rolling_zscore
from limen.features.sma_crossover import sma_crossover
from limen.features.stochastic_k_abs import stochastic_k_abs
from limen.features.trend_strength import trend_strength
from limen.features.volume_to_range import volume_to_range
from limen.features.volume_ratio import volume_ratio
from limen.features.volume_regime import volume_regime
from limen.features.wick_proportion import wick_proportion


SAMPLE_OHLCV = pl.DataFrame(
    {
        'high': [10.0, 12.0, 14.0, 16.0, 18.0],
        'low': [8.0, 9.0, 10.0, 12.0, 15.0],
        'close': [9.0, 11.0, 13.0, 15.0, 17.0],
        'volume': [100.0, 150.0, 120.0, 180.0, 240.0],
    }
)


def _assert_values(actual: list[float | None], expected: list[float | None]) -> None:
    assert len(actual) == len(expected)
    for value, expected_value in zip(actual, expected, strict=True):
        if expected_value is None:
            assert value is None
        else:
            assert value == pytest.approx(expected_value)


def test_rolling_zscore_supports_transforms_and_zero_std_nulls() -> None:
    data = pl.DataFrame({'x': [1.0, 2.0, 3.0], 'flat': [5.0, 5.0, 5.0]})
    abs_data = pl.DataFrame({'x': [-1.0, -3.0, 2.0]})

    identity = rolling_zscore(data, 'x', 2)
    log_result = rolling_zscore(data, 'x', 2, transform='log1p', output_col='x_log_z')
    abs_result = rolling_zscore(abs_data, 'x', 2, transform='abs', output_col='x_abs_z')
    flat = rolling_zscore(data, 'flat', 2)

    assert identity['x_zscore_2'].to_list()[0] is None
    assert identity['x_zscore_2'].to_list()[1:] == pytest.approx([
        math.sqrt(0.5),
        math.sqrt(0.5),
    ])
    assert log_result['x_log_z'].to_list()[1] == pytest.approx(math.sqrt(0.5))
    assert abs_result['x_abs_z'].to_list()[1:] == pytest.approx([
        math.sqrt(0.5),
        -math.sqrt(0.5),
    ])
    assert flat['flat_zscore_2'].to_list()[1:] == [None, None]
    with pytest.raises(ValueError, match='transform must be one of'):
        rolling_zscore(data, 'x', 2, transform='sqrt')


def test_atr_sma_matches_manual_true_range_average() -> None:
    result = atr_sma(SAMPLE_OHLCV, period=2)
    values = result['atr_sma'].to_list()

    assert values[0] is None
    assert values[1:] == pytest.approx([2.5, 3.5, 4.0, 3.5])
    assert 'true_range' not in result.columns


def test_atr_percent_sma_scales_atr_by_close() -> None:
    result = atr_percent_sma(SAMPLE_OHLCV, period=2)
    values = result['atr_percent_sma'].to_list()

    assert values[0] is None
    assert values[1:] == pytest.approx([2.5 / 11.0, 3.5 / 13.0, 4.0 / 15.0, 3.5 / 17.0])
    assert 'atr' not in result.columns


def test_close_position_uses_intra_candle_range() -> None:
    result = close_position(SAMPLE_OHLCV)

    assert result['close_position'].to_list() == pytest.approx([0.5, 2.0 / 3.0, 0.75, 0.75, 2.0 / 3.0])


def test_structural_position_features_match_manual_formulas() -> None:
    data = pl.DataFrame(
        {
            'open': [10.0, 10.0, 10.0, 10.0, 10.0],
            'high': [12.0, 12.0, 12.0, 12.0, 12.0],
            'low': [8.0, 8.0, 8.0, 8.0, 8.0],
            'close': [10.0, 11.0, 12.0, 11.0, 10.0],
            'volume': [100.0, 200.0, 100.0, 50.0, 100.0],
        }
    )

    assert exported_close_position is close_position
    _assert_values(
        close_position(data, window=2)['close_position'].to_list(),
        [None, 0.625, 0.875, 0.875, 0.625],
    )
    _assert_values(
        wick_proportion(data, window=2)['wick_proportion'].to_list(),
        [None, 0.875, 0.625, 0.625, 0.875],
    )
    _assert_values(
        stochastic_k_abs(data, window=3)['stochastic_k_abs'].to_list(),
        [None, None, 0.5, 0.25, 0.0],
    )
    _assert_values(
        close_position_rolling(data, window=2)['close_position_rolling'].to_list(),
        [None, 0.625, 0.875, 0.875, 0.625],
    )
    _assert_values(
        distance_from_ma(data, window=2)['distance_from_ma'].to_list(),
        [None, 0.5 / 10.5, 0.5 / 11.5, -0.5 / 11.5, -0.5 / 10.5],
    )
    _assert_values(
        close_ma_distance_atr(
            data,
            ma_window=2,
            atr_window=2,
        )['close_ma_distance_atr'].to_list(),
        [None, 0.125, 0.125, -0.125, -0.125],
    )
    _assert_values(
        kaufman_efficiency_ratio(data, window=2)['kaufman_efficiency_ratio'].to_list(),
        [None, None, 1.0, 0.0, 1.0],
    )
    _assert_values(
        volume_to_range(data, window=2)['volume_to_range'].to_list(),
        [None, 37.5, 37.5, 18.75, 18.75],
    )


def test_distance_features_use_rolling_highs_and_lows() -> None:
    from_high = distance_from_high(SAMPLE_OHLCV, period=3)['distance_from_high'].to_list()
    from_low = distance_from_low(SAMPLE_OHLCV, period=3)['distance_from_low'].to_list()

    assert from_high[:2] == [None, None]
    assert from_high[2:] == pytest.approx([1.0 / 13.0, 1.0 / 15.0, 1.0 / 17.0])

    assert from_low[:2] == [None, None]
    assert from_low[2:] == pytest.approx([5.0 / 13.0, 6.0 / 15.0, 7.0 / 17.0])


def test_gap_high_uses_previous_close() -> None:
    result = gap_high(SAMPLE_OHLCV)
    values = result['gap_high'].to_list()

    assert values[0] is None
    assert values[1:] == pytest.approx([3.0 / 9.0, 3.0 / 11.0, 3.0 / 13.0, 3.0 / 15.0])


def test_price_range_position_tracks_close_within_rolling_channel() -> None:
    result = price_range_position(SAMPLE_OHLCV, period=3)
    values = result['price_range_position'].to_list()

    assert values[:2] == [None, None]
    assert values[2:] == pytest.approx([5.0 / 6.0, 6.0 / 7.0, 7.0 / 8.0])


def test_trend_strength_uses_fast_and_slow_moving_average_divergence() -> None:
    result = trend_strength(SAMPLE_OHLCV, fast_period=2, slow_period=3)
    values = result['trend_strength'].to_list()

    assert values[:2] == [None, None]
    assert values[2:] == pytest.approx([1.0 / 11.0, 1.0 / 13.0, 1.0 / 15.0])


def test_volume_ratio_builds_missing_baseline_sma() -> None:
    result = volume_ratio(SAMPLE_OHLCV, period=2)

    assert 'volume_sma_2' in result.columns
    assert math.isnan(result['volume_ratio'].to_list()[0])
    assert result['volume_ratio'].to_list()[1:] == pytest.approx([1.2, 120.0 / 135.0, 1.2, 240.0 / 210.0])


def test_volume_ratio_respects_existing_baseline_column() -> None:
    data = SAMPLE_OHLCV.with_columns(pl.Series('volume_sma_2', [1.0, 2.0, 3.0, 4.0, 5.0]))
    result = volume_ratio(data, period=2)

    assert result['volume_sma_2'].to_list() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert result['volume_ratio'].to_list() == pytest.approx([100.0, 75.0, 40.0, 45.0, 48.0])


def test_volume_regime_compares_recent_volume_to_longer_average() -> None:
    result = volume_regime(SAMPLE_OHLCV, lookback=4)
    values = result['volume_regime'].to_list()

    assert values[:3] == [None, None, None]
    assert values[3:] == pytest.approx([180.0 / 137.5, 240.0 / 172.5])


def test_ichimoku_cloud_builds_shifted_components() -> None:
    result = ichimoku_cloud(
        SAMPLE_OHLCV,
        tenkan_period=2,
        kijun_period=3,
        senkou_b_period=4,
        displacement=1,
    )

    assert result['tenkan'].to_list() == [None, 10.0, 11.5, 13.0, 15.0]
    assert result['kijun'].to_list() == [None, None, 11.0, 12.5, 14.0]
    assert result['senkou_a'].to_list() == [None, None, None, 11.25, 12.75]
    assert result['senkou_b'].to_list() == [None, None, None, None, 12.0]
    assert result['chikou'].to_list() == [None, 9.0, 11.0, 13.0, 15.0]


def test_sma_crossover_marks_bullish_and_bearish_crosses() -> None:
    data = pl.DataFrame({'close': [5.0, 4.0, 3.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]})

    result = sma_crossover(data, short_window=2, long_window=3)

    assert result['crossover'].to_list()[5] == 2
    assert result['signal'].to_list()[5] == 1
    assert result['crossover'].to_list()[8] == -2
    assert result['signal'].to_list()[8] == -1
    assert result['signal'].to_list()[6] == 0
    assert result['signal'].to_list()[7] == 0

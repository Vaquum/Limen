import math
from datetime import datetime

import polars as pl
import pytest

from limen.features import close_ma_distance_atr
from limen.features import close_position as exported_close_position
from limen.features import close_position_rolling
from limen.features import distance_from_ma
from limen.features import downside_volatility_ratio
from limen.features import is_funding_hour
from limen.features import is_us_open_hour
from limen.features import kaufman_efficiency_ratio
from limen.features import liquidity_drop
from limen.features import liquidity_range
from limen.features import maker_liquidity_share
from limen.features import maker_volume_ratio
from limen.features import maker_volume_share
from limen.features import narrow_range
from limen.features import parkinson_vol_of_vol
from limen.features import return_autocorrelation
from limen.features import return_volatility_correlation
from limen.features import rolling_zscore
from limen.features import stochastic_k_abs
from limen.features import taker_imbalance_ratio
from limen.features import trade_density
from limen.features import trade_imbalance
from limen.features import trade_size_ratio
from limen.features import volume_to_range
from limen.features import volume_volatility_correlation
from limen.features import volatility_ratio
from limen.features import volatility_spike
from limen.features import wick_proportion
from limen.features.calendar_time_features import calendar_time_features
from limen.features.close_position import close_position


PARKINSON_SCALE = 4.0 * math.log(2.0)


def _bars_with_parkinson_variance(variance: list[float]) -> pl.DataFrame:
    log_ranges = [math.sqrt(value * PARKINSON_SCALE) for value in variance]
    return pl.DataFrame(
        {
            'open': [1.0] * len(variance),
            'high': [math.exp(value) for value in log_ranges],
            'low': [1.0] * len(variance),
            'close': [1.0] * len(variance),
            'volume': [float(idx + 1) for idx in range(len(variance))],
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
    assert identity['x_zscore_2'].to_list()[1:] == pytest.approx([math.sqrt(0.5), math.sqrt(0.5)])
    assert log_result['x_log_z'].to_list()[1] == pytest.approx(math.sqrt(0.5))
    assert abs_result['x_abs_z'].to_list()[1:] == pytest.approx([math.sqrt(0.5), -math.sqrt(0.5)])
    assert flat['flat_zscore_2'].to_list()[1:] == [None, None]
    with pytest.raises(ValueError, match='transform must be one of'):
        rolling_zscore(data, 'x', 2, transform='sqrt')


def test_microstructure_features_match_the_lab_formulas() -> None:
    data = pl.DataFrame(
        {
            'volume': [10.0, 20.0, 30.0, 40.0],
            'maker_volume': [5.0, 10.0, 15.0, 20.0],
            'liquidity_sum': [100.0, 200.0, 100.0, 50.0],
            'maker_liquidity': [25.0, 100.0, 50.0, 25.0],
            'maker_ratio': [0.5, 0.25, 0.75, 0.5],
            'no_of_trades': [2.0, 4.0, 5.0, 10.0],
            'high_liquidity': [20.0, 30.0, 80.0, 50.0],
            'low_liquidity': [10.0, 15.0, 40.0, 25.0],
        }
    )

    _assert_values(maker_volume_share(data)['maker_volume_share'].to_list(), [0.5, 0.5, 0.5, 0.5])
    _assert_values(maker_liquidity_share(data)['maker_liquidity_share'].to_list(), [0.25, 0.5, 0.5, 0.5])
    _assert_values(maker_volume_ratio(data, window=2)['maker_volume_ratio'].to_list(), [None, 0.5, 0.5, 0.5])
    _assert_values(trade_imbalance(data, window=2)['trade_imbalance'].to_list(), [None, 0.5, 0.5, 0.5])
    _assert_values(taker_imbalance_ratio(data, window=2)['taker_imbalance_ratio'].to_list(), [None, 0.25, 0.5, 0.25])
    _assert_values(trade_density(data, window=2)['trade_density'].to_list(), [None, 0.2, 11.0 / 60.0, 5.0 / 24.0])
    _assert_values(trade_size_ratio(data, short_window=2, long_window=3)['trade_size_ratio'].to_list(), [None, None, 33.0 / 32.0, 1.0])
    _assert_values(liquidity_range(data, window=2)['liquidity_range'].to_list(), [None, 2.0, 2.0, 2.0])
    _assert_values(liquidity_drop(data, window=2)['liquidity_drop'].to_list(), [None, None, 1.0, 0.25])


def test_structural_rolling_features_match_the_lab_formulas() -> None:
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
    _assert_values(close_position(data, window=2)['close_position'].to_list(), [None, 0.625, 0.875, 0.875, 0.625])
    _assert_values(wick_proportion(data, window=2)['wick_proportion'].to_list(), [None, 0.875, 0.625, 0.625, 0.875])
    _assert_values(stochastic_k_abs(data, window=3)['stochastic_k_abs'].to_list(), [None, None, 0.5, 0.25, 0.0])
    _assert_values(volatility_ratio(data, short_window=2, long_window=3)['volatility_ratio'].to_list(), [None, None, 1.0, 1.0, 1.0])
    _assert_values(close_position_rolling(data, window=2)['close_position_rolling'].to_list(), [None, 0.625, 0.875, 0.875, 0.625])
    _assert_values(distance_from_ma(data, window=2)['distance_from_ma'].to_list(), [None, 0.5 / 10.5, 0.5 / 11.5, -0.5 / 11.5, -0.5 / 10.5])
    _assert_values(close_ma_distance_atr(data, ma_window=2, atr_window=2)['close_ma_distance_atr'].to_list(), [None, 0.125, 0.125, -0.125, -0.125])
    _assert_values(kaufman_efficiency_ratio(data, window=2)['kaufman_efficiency_ratio'].to_list(), [None, None, 1.0, 0.0, 1.0])
    assert downside_volatility_ratio(data, window=3)['downside_volatility_ratio'].to_list()[3:] == pytest.approx([0.2754758218741462, 0.6479217603911981])
    _assert_values(volume_to_range(data, window=2)['volume_to_range'].to_list(), [None, 37.5, 37.5, 18.75, 18.75])


def test_structural_correlation_and_volatility_helpers() -> None:
    variance_data = _bars_with_parkinson_variance([1.0, 2.0, 3.0, 4.0, 5.0])
    return_data = variance_data.with_columns(
        pl.Series('close', [1.0, 2.0, 6.0, 24.0, 120.0])
    )
    alternating_returns = pl.DataFrame(
        {
            'close': [100.0, 110.0, 99.0, 108.9, 98.01],
            'high': [110.0] * 5,
            'low': [90.0] * 5,
            'volume': [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    range_data = pl.DataFrame(
        {
            'high': [11.0, 12.0, 10.5, 11.0],
            'low': [9.0, 8.0, 9.5, 9.0],
            'close': [10.0, 10.0, 10.0, 10.0],
            'volume': [10.0, 20.0, 30.0, 40.0],
        }
    )

    assert return_autocorrelation(alternating_returns, window=3)['return_autocorrelation'].to_list()[-1] == pytest.approx(-1.0)
    assert volume_volatility_correlation(variance_data, window=3)['volume_volatility_correlation'].to_list()[-1] == pytest.approx(1.0)
    assert return_volatility_correlation(return_data, window=3)['return_volatility_correlation'].to_list()[-1] == pytest.approx(1.0)
    _assert_values(narrow_range(range_data, window=2)['narrow_range'].to_list(), [None, 1.0, 0.25, 1.0])
    _assert_values(volatility_spike(variance_data, window=2)['volatility_spike'].to_list(), [None, None, 3.0, 2.0, 5.0 / 3.0])
    assert parkinson_vol_of_vol(variance_data, window=3)['parkinson_vol_of_vol'].to_list()[2:] == pytest.approx([1.0, 1.0, 1.0])


def test_calendar_gaps_add_half_year_and_session_flags() -> None:
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

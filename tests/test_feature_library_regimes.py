import math
from datetime import datetime

import polars as pl
import pytest

from limen.features.breakout_percentile_regime import breakout_percentile_regime
from limen.features.close_to_extremes import close_to_extremes
from limen.features.dynamic_stop_loss import dynamic_stop_loss
from limen.features.dynamic_target import dynamic_target
from limen.features.ema_alignment import ema_alignment
from limen.features.exit_quality import exit_quality
from limen.features.log_returns import log_returns
from limen.features.micro_momentum import micro_momentum
from limen.features.momentum_weight import momentum_weight
from limen.features.position_in_candle import position_in_candle
from limen.features.position_in_range import position_in_range
from limen.features.regime_multiplier import regime_multiplier
from limen.features.risk_reward_ratio import risk_reward_ratio
from limen.features.spread import spread
from limen.features.spread_percent import spread_percent
from limen.features.time_features import time_features
from limen.features.volatility_1h import volatility_1h
from limen.features.volatility_measure import volatility_measure
from limen.features.volume_spike import volume_spike
from limen.features.window_return_regime import window_return_regime


OHLC = pl.DataFrame(
    {
        'high': [10.0, 12.0, 14.0, 16.0],
        'low': [8.0, 9.0, 10.0, 12.0],
        'close': [9.0, 11.0, 13.0, 15.0],
        'volume': [100.0, 150.0, 120.0, 180.0],
    }
)


def test_close_to_extremes_reports_relative_distance_to_high_and_low() -> None:
    result = close_to_extremes(OHLC)

    assert result['close_to_high'].to_list() == pytest.approx([-0.1, -1.0 / 12.0, -1.0 / 14.0, -1.0 / 16.0])
    assert result['close_to_low'].to_list() == pytest.approx([1.0 / 8.0, 2.0 / 9.0, 3.0 / 10.0, 3.0 / 12.0])


def test_dynamic_stop_loss_clips_volatility_adjusted_levels() -> None:
    data = pl.DataFrame(
        {
            'volatility_measure': [1.0, 10.0, 0.5],
            'regime_multiplier': [1.0, 1.0, 1.0],
        }
    )

    result = dynamic_stop_loss(data, base_stop_loss=5.0, stop_volatility_multiplier=1.0)

    assert result['dynamic_stop_loss'].to_list() == pytest.approx([3.5, 7.0, 3.5])


def test_dynamic_target_clips_volatility_adjusted_targets() -> None:
    data = pl.DataFrame(
        {
            'volatility_measure': [5.0, 20.0, 10.0],
            'regime_multiplier': [1.0, 1.0, 1.0],
        }
    )

    result = dynamic_target(data, base_min_breakout=10.0, target_volatility_multiplier=1.0)

    assert result['dynamic_target'].to_list() == pytest.approx([6.0, 14.0, 10.0])


def test_exit_quality_distinguishes_good_bad_and_neutral_exits() -> None:
    data = pl.DataFrame(
        {
            'exit_reason': ['target_hit', 'stop_loss', 'timeout', 'timeout'],
            'exit_net_return': [0.2, -0.4, -0.1, 0.1],
        }
    )

    result = exit_quality(data)

    assert result['exit_quality'].to_list() == pytest.approx([1.0, 0.2, 0.2, 0.5])


def test_log_returns_uses_log_ratio_of_consecutive_prices() -> None:
    data = pl.DataFrame({'close': [2.0, 4.0, 2.0]})
    result = log_returns(data)
    values = result['log_returns'].to_list()

    assert values[0] is None
    assert values[1:] == pytest.approx([math.log(2.0), math.log(0.5)])


def test_micro_momentum_uses_period_pct_change() -> None:
    data = pl.DataFrame({'close': [100.0, 120.0, 90.0, 99.0]})
    result = micro_momentum(data, period=2)

    assert result['micro_momentum'].to_list()[:2] == [None, None]
    assert result['micro_momentum'].to_list()[2:] == pytest.approx([-0.1, -0.175])


def test_momentum_weight_only_boosts_positive_momentum() -> None:
    data = pl.DataFrame({'close': [100.0, 101.0, 100.0]})
    result = momentum_weight(data, period=1, weight_multiplier=0.5, base_weight=0.5)
    values = result['momentum_weight'].to_list()

    assert values[1:] == pytest.approx([1.0, 0.5])


def test_position_in_range_matches_candle_fraction_formula() -> None:
    result = position_in_range(OHLC)

    assert result['position_in_range'].to_list() == pytest.approx([0.5, 2.0 / 3.0, 0.75, 0.75])


def test_regime_multiplier_maps_low_normal_and_high_regimes() -> None:
    data = pl.DataFrame({'volatility_regime': ['low', 'normal', 'high']})
    result = regime_multiplier(data)

    assert result['regime_multiplier'].to_list() == pytest.approx([0.8, 1.0, 1.2])


def test_spread_and_spread_percent_match_same_ratio() -> None:
    spread_values = spread(OHLC)['spread'].to_list()
    spread_percent_values = spread_percent(OHLC)['spread_percent'].to_list()

    expected = [2.0 / 9.0, 3.0 / 11.0, 4.0 / 13.0, 4.0 / 15.0]
    assert spread_values == pytest.approx(expected)
    assert spread_percent_values == pytest.approx(expected)


def test_time_features_extract_hour_minute_and_weekday() -> None:
    data = pl.DataFrame(
        {
            'datetime': [
                datetime(2026, 4, 17, 12, 30),
                datetime(2026, 4, 18, 1, 15),
            ]
        }
    )

    result = time_features(data)

    assert result['hour'].to_list() == [12, 1]
    assert result['minute'].to_list() == [30, 15]
    assert result['weekday'].to_list() == [5, 6]


def test_volatility_1h_copies_requested_source_column() -> None:
    data = pl.DataFrame({'returns_volatility_12': [0.1, 0.2, 0.3]})
    result = volatility_1h(data)

    assert result['volatility_1h'].to_list() == pytest.approx([0.1, 0.2, 0.3])


def test_volatility_measure_averages_rolling_volatility_and_atr_percent() -> None:
    data = pl.DataFrame(
        {
            'rolling_volatility': [0.1, 0.2, 0.3],
            'atr_percent_sma': [0.3, 0.4, 0.5],
        }
    )

    result = volatility_measure(data)

    assert result['volatility_measure'].to_list() == pytest.approx([0.2, 0.3, 0.4])


def test_ema_alignment_adds_ema_and_alignment_score() -> None:
    data = pl.DataFrame({'close': [10.0, 13.0, 13.0]})
    result = ema_alignment(data, ema_span=2, power=1.0)

    assert result['ema'].to_list() == pytest.approx([10.0, 12.0, 12.6666666667])
    assert result['ema_alignment'].to_list() == pytest.approx([1.0, 11.0 / 12.0, 1.0 - (1.0 / 3.0) / 12.6666666667], rel=1e-6)


def test_position_in_candle_matches_range_fraction() -> None:
    result = position_in_candle(OHLC)

    assert result['position_in_candle'].to_list() == pytest.approx([0.5, 2.0 / 3.0, 0.75, 0.75])


def test_risk_reward_ratio_uses_absolute_drawdown_with_epsilon_guard() -> None:
    data = pl.DataFrame(
        {
            'capturable_breakout': [0.5, 1.0],
            'max_drawdown': [-0.1, 0.0],
        }
    )

    result = risk_reward_ratio(data)

    assert result['risk_reward_ratio'].to_list() == pytest.approx([0.5 / 0.101, 1000.0])


def test_volume_spike_supports_ratio_and_zscore_modes() -> None:
    ratio_values = volume_spike(OHLC, period=2, use_zscore=False)['volume_spike'].to_list()
    zscore_values = volume_spike(OHLC, period=2, use_zscore=True)['volume_spike'].to_list()

    assert ratio_values[0] is None
    assert ratio_values[1:] == pytest.approx([1.2, 120.0 / 135.0, 1.2])

    assert zscore_values[0] is None
    assert zscore_values[1:] == pytest.approx([0.70710678, -0.70710678, 0.70710678], rel=1e-6)


def test_breakout_percentile_regime_labels_up_flat_and_down() -> None:
    data = pl.DataFrame(
        {
            'high': [10.0, 10.0, 10.0, 10.0],
            'low': [0.0, 0.0, 0.0, 0.0],
            'close': [1.0, 5.0, 9.0, 1.0],
        }
    )

    result = breakout_percentile_regime(data, period=2, p_hi=0.8, p_lo=0.2)

    assert result['regime_breakout_pct'].to_list() == ['Flat', 'Flat', 'Up', 'Down']


def test_window_return_regime_labels_positive_negative_and_neutral_windows() -> None:
    data = pl.DataFrame({'close': [10.0, 11.0, 9.0, 10.0]})
    result = window_return_regime(data, period=1, r_hi=0.05, r_lo=-0.05)

    assert result['regime_window_return'].to_list() == ['Flat', 'Up', 'Down', 'Up']

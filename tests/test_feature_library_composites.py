import math

import numpy as np
import polars as pl
import pytest

from limen.features.entry_score_microstructure import entry_score_microstructure
from limen.features.feature_aliases import feature_aliases
from limen.features.hh_hl_structure_regime import hh_hl_structure_regime
from limen.features.ma_slope_regime import ma_slope_regime
from limen.features.momentum_confirmation import momentum_confirmation
from limen.features.momentum_periods import momentum_periods
from limen.features.price_vs_band_regime import price_vs_band_regime
from limen.features.returns_lags import returns_lags
from limen.features.volatility_weight import volatility_weight
from limen.features.volume_trend import volume_trend
from limen.features.volume_weight import volume_weight


def test_feature_aliases_fill_nulls_and_build_regime_aliases() -> None:
    data = pl.DataFrame(
        {
            'dynamic_target': [None, 0.7],
            'entry_score': [None, 0.8],
            'momentum_score': [None, 0.6],
            'vol_60h': [None, 0.4],
            'vol_percentile': [None, 70.0],
            'regime_low': [None, 1.0],
            'regime_normal': [None, 0.0],
            'regime_high': [None, 0.0],
        }
    )

    result = feature_aliases(data, base_min_breakout=0.5, volatility_regime_enabled=True)

    assert result['dynamic_target_feature'].to_list() == pytest.approx([0.5, 0.7])
    assert result['entry_score_feature'].to_list() == pytest.approx([1.0, 0.8])
    assert result['momentum_score_feature'].to_list() == pytest.approx([1.0, 0.6])
    assert result['vol_60h_feature'].to_list() == pytest.approx([0.0, 0.4])
    assert result['vol_percentile_feature'].to_list() == pytest.approx([50.0, 70.0])
    assert result['regime_normal_feature'].to_list() == pytest.approx([1.0, 0.0])


def test_feature_aliases_can_skip_regime_aliases() -> None:
    data = pl.DataFrame(
        {
            'dynamic_target': [None],
            'entry_score': [None],
            'momentum_score': [None],
        }
    )

    result = feature_aliases(data, volatility_regime_enabled=False)

    assert 'vol_60h_feature' not in result.columns
    assert 'regime_high_feature' not in result.columns


def test_returns_lags_builds_returns_when_missing() -> None:
    data = pl.DataFrame({'close': [100.0, 110.0, 99.0, 108.9]})
    result = returns_lags(data, max_lag=2)

    assert 'returns' in result.columns
    assert result['returns_lag_1'].to_list()[2:] == pytest.approx([0.1, -0.1])
    assert result['returns_lag_2'].to_list()[2:] == pytest.approx([None, 0.1])


def test_returns_lags_respects_existing_returns_column() -> None:
    data = pl.DataFrame({'close': [1.0, 2.0, 3.0], 'returns': [0.0, 0.5, 0.25]})
    result = returns_lags(data, max_lag=1)

    assert result['returns_lag_1'].to_list() == [None, 0.0, 0.5]


def test_volume_weight_uses_clipped_ratio_to_volume_average() -> None:
    data = pl.DataFrame({'volume': [100.0, 150.0, 600.0]})
    result = volume_weight(data, period=2, volume_weight_min=0.5, volume_weight_max=1.5)

    assert 'volume_ma' in result.columns
    assert math.isnan(result['volume_ma'].to_list()[0])
    assert result['volume_weight'].to_list()[1:] == pytest.approx([1.2, 1.5])


def test_momentum_confirmation_combines_short_and_long_signals() -> None:
    data = pl.DataFrame({'close': [100.0, 105.0, 90.0, 120.0]})
    result = momentum_confirmation(data, short_period=1, long_period=2, short_weight=0.25)

    assert result['momentum_score'].to_list()[2:] == pytest.approx([0.0, 1.0])


def test_momentum_confirmation_rejects_invalid_period_order() -> None:
    with pytest.raises(ValueError, match='short_period'):
        momentum_confirmation(pl.DataFrame({'close': [1.0, 2.0]}), short_period=3, long_period=3)


def test_volatility_weight_inversely_scales_rolling_volatility() -> None:
    data = pl.DataFrame({'close': [100.0, 110.0, 100.0, 120.0]})
    result = volatility_weight(
        data,
        period=2,
        volatility_scaling_factor=10.0,
        volatility_weight_min=0.3,
        volatility_weight_max=1.0,
    )
    values = result['volatility_weight'].to_list()
    returns = [0.1, -0.09090909090909091, 0.2]
    expected_std_row2 = np.std(returns[:2], ddof=1)
    expected_std_row3 = np.std(returns[1:], ddof=1)

    assert 'returns_temp' not in result.columns
    assert values[:2] == [None, None]
    assert values[2:] == pytest.approx(
        [
            (2 / (1 + expected_std_row2 * 10.0)),
            (2 / (1 + expected_std_row3 * 10.0)),
        ],
        rel=1e-6,
    )


def test_momentum_periods_adds_requested_period_columns() -> None:
    data = pl.DataFrame({'close': [100.0, 110.0, 121.0]})
    result = momentum_periods(data, periods=[1, 2])

    assert result['momentum_1'].to_list()[1:] == pytest.approx([0.1, 0.1])
    assert result['momentum_2'].to_list()[2] == pytest.approx(0.21)


def test_volume_trend_compares_short_and_long_volume_averages() -> None:
    data = pl.DataFrame({'volume': [100.0, 200.0, 300.0, 400.0]})
    result = volume_trend(data, short_period=2, long_period=3)

    assert math.isnan(result['volume_trend'].to_list()[0])
    assert math.isnan(result['volume_trend'].to_list()[1])
    assert result['volume_trend'].to_list()[2:] == pytest.approx([1.25, 350.0 / 300.0])


def test_entry_score_microstructure_switches_weights_by_regime() -> None:
    data = pl.DataFrame(
        {
            'high': [10.0, 10.0, 10.0, 10.0],
            'low': [0.0, 0.0, 0.0, 0.0],
            'close': [5.0, 6.0, 7.0, 8.0],
            'volume': [100.0, 100.0, 100.0, 100.0],
            'volatility_regime': ['normal', 'low', 'high', 'normal'],
        }
    )
    result = entry_score_microstructure(
        data,
        micro_momentum_period=1,
        volume_spike_period=1,
        spread_mean_period=1,
    )
    scores = result['entry_score'].to_list()

    assert scores[0] is None
    assert scores[1:] == pytest.approx(
        [
            0.39,
            0.6283333333,
            0.4666666667,
        ],
        rel=1e-6,
    )


def test_hh_hl_structure_regime_detects_up_flat_and_down_windows() -> None:
    data = pl.DataFrame(
        {
            'high': [1.0, 2.0, 3.0, 2.0, 1.0],
            'low': [0.0, 1.0, 2.0, 1.0, 0.0],
        }
    )
    result = hh_hl_structure_regime(data, window=2, score_threshold=2)

    assert result['regime_hh_hl'].to_list()[2:] == ['Up', 'Flat', 'Down']


def test_ma_slope_regime_uses_sma_slope_direction() -> None:
    data = pl.DataFrame({'close': [1.0, 2.0, 3.0, 4.0, 3.0, 2.0]})
    result = ma_slope_regime(data, period=2, threshold=0.1, normalize_by_std=False)

    assert result['regime_ma_slope'].to_list()[3:] == ['Up', 'Up', 'Down']


def test_price_vs_band_regime_flags_prices_outside_rolling_bands() -> None:
    data = pl.DataFrame({'close': [10.0, 10.0, 10.0, 20.0, 5.0]})
    result = price_vs_band_regime(data, period=2, band='std', k=0.2)

    assert result['regime_price_band'].to_list()[2:] == ['Flat', 'Up', 'Down']

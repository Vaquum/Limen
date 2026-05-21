import math

import numpy as np
import polars as pl
import pytest

from limen.features.downside_volatility_ratio import downside_volatility_ratio
from limen.features.jump_variation_proxy import jump_variation_proxy
from limen.features.realized_kurtosis import realized_kurtosis
from limen.features.realized_semivariance import realized_semivariance
from limen.features.realized_skewness import realized_skewness
from limen.features.return_autocorrelation import return_autocorrelation
from limen.features.return_volatility_correlation import return_volatility_correlation
from limen.features.tail_event_intensity import tail_event_intensity
from limen.features.volume_volatility_correlation import volume_volatility_correlation
from limen.features.volatility_of_volatility import volatility_of_volatility


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


def test_realized_semivariance_separates_upside_and_downside_return_energy() -> None:
    data = pl.DataFrame({'close': [100.0, 110.0, 99.0, 118.8]})
    result = realized_semivariance(data, window=3)

    assert result['upside_semivariance'].to_list()[3] == pytest.approx(((0.1 ** 2) + 0.0 + (0.2 ** 2)) / 3.0)
    assert result['downside_semivariance'].to_list()[3] == pytest.approx((0.0 + (0.1 ** 2) + 0.0) / 3.0)


def test_realized_skewness_and_kurtosis_match_manual_standardized_moments() -> None:
    data = pl.DataFrame({'close': [100.0, 110.0, 99.0, 118.8]})
    skewness = realized_skewness(data, window=3)
    kurtosis = realized_kurtosis(data, window=3)

    returns = np.asarray([0.1, -0.1, 0.2], dtype=float)
    mean = returns.mean()
    centered = returns - mean
    variance = np.mean(centered ** 2)
    expected_skewness = np.mean(centered ** 3) / (variance ** 1.5)
    expected_kurtosis = np.mean(centered ** 4) / (variance ** 2)

    assert skewness['realized_skewness'].to_list()[3] == pytest.approx(expected_skewness)
    assert kurtosis['realized_kurtosis'].to_list()[3] == pytest.approx(expected_kurtosis)


def test_jump_variation_proxy_matches_manual_realized_minus_bipower_component() -> None:
    data = pl.DataFrame({'close': [100.0, 110.0, 99.0, 118.8]})
    result = jump_variation_proxy(data, window=2)

    log_returns = np.log(np.asarray([110.0, 99.0, 118.8]) / np.asarray([100.0, 110.0, 99.0]))
    realized_variance = (log_returns[1] ** 2) + (log_returns[2] ** 2)
    bipower = (math.pi / 2.0) * abs(log_returns[2]) * abs(log_returns[1])
    expected = max(realized_variance - bipower, 0.0)

    assert result['jump_variation_proxy'].to_list()[3] == pytest.approx(expected)


def test_jump_variation_proxy_returns_null_when_window_cannot_form_bipower_pairs() -> None:
    data = pl.DataFrame({'close': [100.0, 110.0, 99.0]})
    result = jump_variation_proxy(data, window=1)

    assert result['jump_variation_proxy'].to_list() == [None, None, None]


def test_tail_event_intensity_lights_up_after_repeated_large_shocks() -> None:
    data = pl.DataFrame({'close': [100.0, 101.0, 102.01, 132.613, 172.3969]})
    result = tail_event_intensity(data, window=2, z_threshold=1.5)
    values = result['tail_event_intensity'].to_list()

    assert values[-1] == pytest.approx(0.5)


def test_volatility_of_volatility_matches_nested_sample_standard_deviation() -> None:
    data = pl.DataFrame({'close': [100.0, 110.0, 121.0, 145.2, 174.24]})
    result = volatility_of_volatility(data, volatility_window=2, window=2)

    base_vols = np.asarray([
        np.std([0.1, 0.1], ddof=1),
        np.std([0.1, 0.2], ddof=1),
        np.std([0.2, 0.2], ddof=1),
    ])
    expected = np.std(base_vols[1:], ddof=1)

    assert result['volatility_of_volatility'].to_list()[-1] == pytest.approx(expected)


def test_structural_correlation_features_match_manual_formulas() -> None:
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
    directional_data = pl.DataFrame(
        {
            'open': [10.0, 10.0, 10.0, 10.0, 10.0],
            'high': [12.0, 12.0, 12.0, 12.0, 12.0],
            'low': [8.0, 8.0, 8.0, 8.0, 8.0],
            'close': [10.0, 11.0, 12.0, 11.0, 10.0],
            'volume': [100.0, 200.0, 100.0, 50.0, 100.0],
        }
    )

    assert return_autocorrelation(
        alternating_returns,
        window=3,
    )['return_autocorrelation'].to_list()[-1] == pytest.approx(-1.0)
    assert volume_volatility_correlation(
        variance_data,
        window=3,
    )['volume_volatility_correlation'].to_list()[-1] == pytest.approx(1.0)
    assert return_volatility_correlation(
        return_data,
        window=3,
    )['return_volatility_correlation'].to_list()[-1] == pytest.approx(1.0)
    assert downside_volatility_ratio(
        directional_data,
        window=3,
    )['downside_volatility_ratio'].to_list()[3:] == pytest.approx([
        0.2754758218741462,
        0.6479217603911981,
    ])

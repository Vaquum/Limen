import math

import numpy as np
import polars as pl
import pytest

from limen.features.jump_variation_proxy import jump_variation_proxy
from limen.features.realized_kurtosis import realized_kurtosis
from limen.features.realized_semivariance import realized_semivariance
from limen.features.realized_skewness import realized_skewness
from limen.features.tail_event_intensity import tail_event_intensity
from limen.features.volatility_of_volatility import volatility_of_volatility


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
    bipower = (math.pi / 2.0) * (
        abs(log_returns[1]) * abs(log_returns[0])
        + abs(log_returns[2]) * abs(log_returns[1])
    )
    expected = max(realized_variance - bipower, 0.0)

    assert result['jump_variation_proxy'].to_list()[3] == pytest.approx(expected)


def test_tail_event_intensity_lights_up_after_repeated_large_shocks() -> None:
    data = pl.DataFrame({'close': [100.0, 101.0, 102.01, 132.613, 172.3969]})
    result = tail_event_intensity(data, window=2, z_threshold=1.5)
    values = result['tail_event_intensity'].to_list()

    assert values[-1] == pytest.approx(0.5)


def test_volatility_of_volatility_matches_nested_population_standard_deviation() -> None:
    data = pl.DataFrame({'close': [100.0, 110.0, 121.0, 145.2, 174.24]})
    result = volatility_of_volatility(data, volatility_window=2, window=2)

    base_vols = np.asarray([
        np.std([0.1, 0.1], ddof=1),
        np.std([0.1, 0.2], ddof=1),
        np.std([0.2, 0.2], ddof=1),
    ])
    expected = np.std(base_vols[1:], ddof=1)

    assert result['volatility_of_volatility'].to_list()[-1] == pytest.approx(expected)

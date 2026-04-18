import numpy as np
import polars as pl
import pytest

from limen.features.trend_coherence import trend_coherence
from limen.features.volatility_term_structure import volatility_term_structure


def test_trend_coherence_scores_full_and_partial_directional_agreement() -> None:
    all_up = pl.DataFrame({'close': [100.0, 110.0, 121.0, 133.1, 146.41]})
    mixed = pl.DataFrame({'close': [100.0, 110.0, 99.0, 89.1, 98.01]})

    all_up_result = trend_coherence(all_up, short_window=1, medium_window=2, long_window=4)
    mixed_result = trend_coherence(mixed, short_window=1, medium_window=2, long_window=4)

    assert all_up_result['trend_coherence'].to_list()[-1] == pytest.approx(1.0)
    assert mixed_result['trend_coherence'].to_list()[-1] == pytest.approx(-1.0 / 3.0)


def test_volatility_term_structure_matches_manual_front_vs_back_ratio_average() -> None:
    data = pl.DataFrame({'close': [100.0, 110.0, 121.0, 133.1, 146.41, 190.333, 133.2331]})
    result = volatility_term_structure(data, short_window=2, medium_window=3, long_window=4)

    returns = np.asarray([0.1, 0.1, 0.1, 0.1, 0.3, -0.3], dtype=float)
    short_vol = np.std(returns[-2:], ddof=1)
    medium_vol = np.std(returns[-3:], ddof=1)
    long_vol = np.std(returns[-4:], ddof=1)
    expected = ((short_vol / medium_vol) + (medium_vol / long_vol)) / 2.0

    assert result['volatility_term_structure'].to_list()[-1] == pytest.approx(expected)

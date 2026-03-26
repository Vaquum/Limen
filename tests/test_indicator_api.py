import polars as pl
import pytest

from limen.indicators import apo, cdldoji, ppo


SAMPLE_DATA = pl.DataFrame(
    {
        'open': [1.0, 2.0, 3.0, 4.0, 5.0],
        'high': [2.0, 3.0, 4.0, 5.0, 6.0],
        'low': [0.5, 1.5, 2.5, 3.5, 4.5],
        'close': [1.5, 2.5, 3.5, 4.5, 5.5],
    }
)


def test_cdldoji_exports_cdldoji_column() -> None:
    result = cdldoji(SAMPLE_DATA)

    assert 'cdldoji' in result.columns


def test_apo_rejects_fast_period_not_less_than_slow_period() -> None:
    with pytest.raises(ValueError, match='slow_period must be greater than fast_period'):
        apo(SAMPLE_DATA, fast_period=26, slow_period=12)


def test_ppo_rejects_fast_period_not_less_than_slow_period() -> None:
    with pytest.raises(ValueError, match='slow_period must be greater than fast_period'):
        ppo(SAMPLE_DATA, fast_period=26, slow_period=12)

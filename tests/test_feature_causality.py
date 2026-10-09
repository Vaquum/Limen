from pathlib import Path

import polars as pl
import pytest

from limen.features.ichimoku_cloud import ichimoku_cloud
from limen.features.momentum_periods import momentum_periods


@pytest.fixture
def recorded_prices() -> pl.DataFrame:
    return pl.read_parquet(
        Path(__file__).parent / 'fixtures' / 'spot_15m_20250101_20250531.parquet'
    ).head(201)


@pytest.mark.parametrize('periods', [[-1], [1, -2], [-1, 1]])
def test_momentum_periods_rejects_future_offsets(recorded_prices: pl.DataFrame, periods: list[int]) -> None:
    with pytest.raises(ValueError, match='periods must be non-negative'):
        momentum_periods(recorded_prices, periods=periods)


@pytest.mark.parametrize('displacement', [-1, -26])
def test_ichimoku_cloud_rejects_future_offsets(recorded_prices: pl.DataFrame, displacement: int) -> None:
    with pytest.raises(ValueError, match='displacement must be non-negative'):
        ichimoku_cloud(recorded_prices, displacement=displacement)


@pytest.mark.parametrize('periods', [[0], [1, 12, 48], None])
def test_momentum_periods_preserves_causal_prefix(recorded_prices: pl.DataFrame, periods: list[int] | None) -> None:
    prefix = momentum_periods(recorded_prices.head(200), periods=periods)
    extended = momentum_periods(recorded_prices, periods=periods)

    assert prefix.equals(extended.head(200))
    if periods is None:
        assert prefix.equals(momentum_periods(recorded_prices.head(200), periods=[12, 24, 48]))
        assert extended.equals(momentum_periods(recorded_prices, periods=[12, 24, 48]))
    if periods == [0]:
        assert prefix['momentum_0'].to_list() == [0.0] * 200


def test_momentum_periods_preserves_single_pass_inputs(recorded_prices: pl.DataFrame) -> None:
    assert momentum_periods(recorded_prices, periods=iter([1, 2])).equals(momentum_periods(recorded_prices, periods=[1, 2]))


@pytest.mark.parametrize('displacement', [0, 1, 26, None])
def test_ichimoku_cloud_preserves_causal_prefix(recorded_prices: pl.DataFrame, displacement: int | None) -> None:
    kwargs = {} if displacement is None else {'displacement': displacement}
    prefix = ichimoku_cloud(recorded_prices.head(200), **kwargs)
    extended = ichimoku_cloud(recorded_prices, **kwargs)
    if displacement is None:
        assert prefix.equals(ichimoku_cloud(recorded_prices.head(200), 9, 26, 52, 26))
        assert extended.equals(ichimoku_cloud(recorded_prices, 9, 26, 52, 26))
        displacement = 26

    assert prefix.equals(extended.head(200))
    assert prefix['chikou'].equals(recorded_prices.head(200)['close'].shift(displacement).rename('chikou'))

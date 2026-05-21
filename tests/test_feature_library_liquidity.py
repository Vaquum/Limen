import math

import polars as pl
import pytest

from limen.features.amihud_illiquidity import amihud_illiquidity
from limen.features.dollar_volume import dollar_volume
from limen.features.illiquidity_shock import illiquidity_shock
from limen.features.liquidity_drop import liquidity_drop
from limen.features.liquidity_range import liquidity_range
from limen.features.maker_liquidity_share import maker_liquidity_share
from limen.features.maker_volume_ratio import maker_volume_ratio
from limen.features.maker_volume_share import maker_volume_share
from limen.features.range_per_dollar_volume import range_per_dollar_volume
from limen.features.return_per_dollar_volume import return_per_dollar_volume
from limen.features.taker_imbalance_ratio import taker_imbalance_ratio
from limen.features.trade_density import trade_density
from limen.features.trade_imbalance import trade_imbalance
from limen.features.trade_size_ratio import trade_size_ratio


def _assert_values(actual: list[float | None], expected: list[float | None]) -> None:
    assert len(actual) == len(expected)
    for value, expected_value in zip(actual, expected, strict=True):
        if expected_value is None:
            assert value is None
        else:
            assert value == pytest.approx(expected_value)


def test_liquidity_features_match_manual_ohlcv_impact_formulas() -> None:
    data = pl.DataFrame(
        {
            'high': [105.0, 115.0, 126.0, 124.0],
            'low': [95.0, 105.0, 116.0, 118.0],
            'close': [100.0, 110.0, 121.0, 121.0],
            'volume': [10.0, 20.0, 10.0, 5.0],
        }
    )

    dollar = dollar_volume(data)
    amihud = amihud_illiquidity(data)
    range_impact = range_per_dollar_volume(data)
    signed_impact = return_per_dollar_volume(data)
    shock = illiquidity_shock(data, window=2)

    assert dollar['dollar_volume'].to_list() == pytest.approx([1000.0, 2200.0, 1210.0, 605.0])
    assert amihud['amihud_illiquidity'].to_list()[0] is None
    assert amihud['amihud_illiquidity'].to_list()[1:] == pytest.approx([0.1 / 2200.0, 0.1 / 1210.0, 0.0])
    assert range_impact['range_per_dollar_volume'].to_list() == pytest.approx([
        (10.0 / 100.0) / 1000.0,
        (10.0 / 110.0) / 2200.0,
        (10.0 / 121.0) / 1210.0,
        (6.0 / 121.0) / 605.0,
    ])
    assert signed_impact['return_per_dollar_volume'].to_list()[0] is None
    assert signed_impact['return_per_dollar_volume'].to_list()[1:] == pytest.approx([0.1 / 2200.0, 0.1 / 1210.0, 0.0])
    assert shock['illiquidity_shock'].to_list()[3] == pytest.approx(0.0)


def test_return_per_dollar_volume_and_amihud_handle_zero_dollar_volume_with_epsilon_guard() -> None:
    data = pl.DataFrame({'close': [100.0, 110.0], 'volume': [10.0, 0.0]})

    signed_impact = return_per_dollar_volume(data)['return_per_dollar_volume'].to_list()
    amihud_values = amihud_illiquidity(data)['amihud_illiquidity'].to_list()

    assert signed_impact[0] is None
    assert amihud_values[0] is None
    assert math.isfinite(signed_impact[1])
    assert math.isfinite(amihud_values[1])
    assert signed_impact[1] > 0.0
    assert amihud_values[1] > 0.0


def test_native_microstructure_features_match_manual_formulas() -> None:
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

    _assert_values(
        maker_volume_share(data)['maker_volume_share'].to_list(),
        [0.5, 0.5, 0.5, 0.5],
    )
    _assert_values(
        maker_liquidity_share(data)['maker_liquidity_share'].to_list(),
        [0.25, 0.5, 0.5, 0.5],
    )
    _assert_values(
        maker_volume_ratio(data, window=2)['maker_volume_ratio'].to_list(),
        [None, 0.5, 0.5, 0.5],
    )
    _assert_values(
        trade_imbalance(data, window=2)['trade_imbalance'].to_list(),
        [None, 0.5, 0.5, 0.5],
    )
    _assert_values(
        taker_imbalance_ratio(data, window=2)['taker_imbalance_ratio'].to_list(),
        [None, 0.25, 0.5, 0.25],
    )
    _assert_values(
        trade_density(data, window=2)['trade_density'].to_list(),
        [None, 0.2, 11.0 / 60.0, 5.0 / 24.0],
    )
    _assert_values(
        trade_size_ratio(
            data,
            short_window=2,
            long_window=3,
        )['trade_size_ratio'].to_list(),
        [None, None, 33.0 / 32.0, 1.0],
    )
    _assert_values(
        liquidity_range(data, window=2)['liquidity_range'].to_list(),
        [None, 2.0, 2.0, 2.0],
    )
    _assert_values(
        liquidity_drop(data, window=2)['liquidity_drop'].to_list(),
        [None, None, 1.0, 0.25],
    )

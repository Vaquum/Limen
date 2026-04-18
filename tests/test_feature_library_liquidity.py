import math

import polars as pl
import pytest

from limen.features.amihud_illiquidity import amihud_illiquidity
from limen.features.dollar_volume import dollar_volume
from limen.features.illiquidity_shock import illiquidity_shock
from limen.features.range_per_dollar_volume import range_per_dollar_volume
from limen.features.return_per_dollar_volume import return_per_dollar_volume


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

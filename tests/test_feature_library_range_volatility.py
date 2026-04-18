import math

import polars as pl
import pytest

from limen.features import garman_klass_volatility as exported_garman_klass_volatility
from limen.features import parkinson_volatility as exported_parkinson_volatility
from limen.features import rogers_satchell_volatility as exported_rogers_satchell_volatility
from limen.features import yang_zhang_volatility as exported_yang_zhang_volatility
from limen.features.garman_klass_volatility import garman_klass_volatility
from limen.features.parkinson_volatility import parkinson_volatility
from limen.features.rogers_satchell_volatility import rogers_satchell_volatility
from limen.features.yang_zhang_volatility import yang_zhang_volatility


def test_range_based_volatility_estimators_match_constant_ratio_bars() -> None:
    data = pl.DataFrame(
        {
            'open': [100.0, 110.0, 121.0],
            'high': [120.0, 132.0, 145.2],
            'low': [90.0, 99.0, 108.9],
            'close': [110.0, 121.0, 133.1],
        }
    )

    parkinson = parkinson_volatility(data, window=2)
    garman_klass = garman_klass_volatility(data, window=2)
    rogers_satchell = rogers_satchell_volatility(data, window=2)

    log_high_low = math.log(120.0 / 90.0)
    log_close_open = math.log(110.0 / 100.0)
    high_open = math.log(120.0 / 100.0)
    high_close = math.log(120.0 / 110.0)
    low_open = math.log(90.0 / 100.0)
    low_close = math.log(90.0 / 110.0)

    expected_parkinson = math.sqrt((log_high_low ** 2) / (4.0 * math.log(2.0)))
    expected_garman_klass = math.sqrt(
        (0.5 * (log_high_low ** 2)) - (((2.0 * math.log(2.0)) - 1.0) * (log_close_open ** 2))
    )
    expected_rogers_satchell = math.sqrt((high_open * high_close) + (low_open * low_close))

    assert exported_parkinson_volatility is parkinson_volatility
    assert exported_garman_klass_volatility is garman_klass_volatility
    assert exported_rogers_satchell_volatility is rogers_satchell_volatility
    assert parkinson['parkinson_volatility'].to_list() == [None, pytest.approx(expected_parkinson), pytest.approx(expected_parkinson)]
    assert garman_klass['garman_klass_volatility'].to_list() == [None, pytest.approx(expected_garman_klass), pytest.approx(expected_garman_klass)]
    assert rogers_satchell['rogers_satchell_volatility'].to_list() == [None, pytest.approx(expected_rogers_satchell), pytest.approx(expected_rogers_satchell)]


def test_yang_zhang_volatility_handles_zero_overnight_jumps_and_rejects_tiny_windows() -> None:
    data = pl.DataFrame(
        {
            'open': [100.0, 110.0, 121.0],
            'high': [120.0, 132.0, 145.2],
            'low': [90.0, 99.0, 108.9],
            'close': [110.0, 121.0, 133.1],
        }
    )

    with pytest.raises(ValueError, match='window must be greater than 1'):
        yang_zhang_volatility(data, window=1)

    result = yang_zhang_volatility(data, window=2)

    high_open = math.log(120.0 / 100.0)
    high_close = math.log(120.0 / 110.0)
    low_open = math.log(90.0 / 100.0)
    low_close = math.log(90.0 / 110.0)
    rs_variance = (high_open * high_close) + (low_open * low_close)
    k = 0.34 / (1.34 + ((2.0 + 1.0) / (2.0 - 1.0)))
    expected = math.sqrt((1.0 - k) * rs_variance)

    assert exported_yang_zhang_volatility is yang_zhang_volatility
    assert result['yang_zhang_volatility'].to_list()[:2] == [None, None]
    assert result['yang_zhang_volatility'].to_list()[2] == pytest.approx(expected)

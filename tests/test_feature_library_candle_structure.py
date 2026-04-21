import polars as pl
import pytest

from limen.features.absorption_intensity import absorption_intensity
from limen.features.body_to_range import body_to_range
from limen.features.range_overlap import range_overlap
from limen.features.rejection_intensity import rejection_intensity
from limen.features.wick_imbalance import wick_imbalance


def test_candle_structure_features_capture_body_wicks_overlap_and_rejection() -> None:
    data = pl.DataFrame(
        {
            'open': [100.0, 105.0, 95.0],
            'high': [110.0, 108.0, 100.0],
            'low': [95.0, 90.0, 92.0],
            'close': [105.0, 95.0, 95.0],
            'volume': [100.0, 300.0, 150.0],
        }
    )

    body = body_to_range(data)
    imbalance = wick_imbalance(data)
    overlap = range_overlap(data)
    rejection = rejection_intensity(data)
    absorption = absorption_intensity(data, window=1)

    assert body['body_to_range'].to_list() == pytest.approx([1.0 / 3.0, 5.0 / 9.0, 0.0])
    assert imbalance['wick_imbalance'].to_list() == pytest.approx([0.0, -1.0 / 9.0, 0.25])
    assert overlap['range_overlap'].to_list()[0] is None
    assert overlap['range_overlap'].to_list()[1:] == pytest.approx([13.0 / 15.0, 8.0 / 18.0])
    assert rejection['rejection_intensity'].to_list() == pytest.approx([1.0 / 3.0, 1.0 / 6.0, 5.0 / 8.0])
    assert absorption['absorption_intensity'].to_list()[0] is None
    assert absorption['absorption_intensity'].to_list()[1:] == pytest.approx([4.0 / 3.0, 0.5])

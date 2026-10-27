from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from limen.features.classify_market_regime import MarketRegimeClassifier, RegimeConfig, classify_market_regime

FIXTURE = Path(__file__).parent / 'fixtures/spot_1h_20240101_20241231.parquet'


def _recorded(rows=600):
    return pl.read_parquet(FIXTURE, n_rows=rows).with_columns(pl.lit(FIXTURE.stem).alias('series_id'))


def test_recorded_streaming_and_batch_are_exact():
    bars = _recorded()
    config = RegimeConfig(version='baseline-v1')
    expected = classify_market_regime(bars, config=config, include_audit=True)
    classifier = MarketRegimeClassifier(config)
    actual = pl.concat([classifier.update(bars.slice(start, end - start), include_audit=True)
                        for start, end in ((0, 1), (1, 73), (73, 312), (312, 313), (313, 600))])
    assert_frame_equal(actual, expected, check_exact=True)
    assert expected.width == 40 and expected.height == bars.height
    assert expected['market_regime'].null_count() == 0
    assert (expected['market_regime_code'].head(312) == 0).all()
    assert (expected['market_regime_code'].tail(288) != 0).all()
    assert expected['regime_config_hash'].unique().to_list() == [config.config_hash]


def test_recorded_prefix_does_not_change():
    bars, config = _recorded(), RegimeConfig(version='baseline-v1')
    whole = classify_market_regime(bars, config=config, include_audit=True)
    prefix = classify_market_regime(bars.head(421), config=config, include_audit=True)
    assert_frame_equal(prefix, whole.head(421), check_exact=True)

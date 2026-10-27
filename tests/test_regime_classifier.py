from pathlib import Path
from dataclasses import replace
import json

import numpy as np
import pytest

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


def test_checkpoint_retains_exact_observations_and_commitments():
    bars, config = _recorded(), RegimeConfig(version='baseline-v1')
    classifier = MarketRegimeClassifier(config)
    before = classifier.update(bars.head(353), include_audit=True)
    snapshot = classifier.snapshot()
    restored = MarketRegimeClassifier.from_snapshot(config, snapshot)
    assert restored.snapshot() == snapshot
    after = restored.update(bars.slice(353), include_audit=True)
    assert_frame_equal(pl.concat([before, after]), classify_market_regime(bars, config=config, include_audit=True), check_exact=True)
    with pytest.raises(ValueError, match='hash'):
        MarketRegimeClassifier.from_snapshot(replace(config, stress_threshold=.9), snapshot)
    payload = json.loads(snapshot)
    payload['feature_hash'] = 'changed'
    with pytest.raises(ValueError, match='hash'):
        MarketRegimeClassifier.from_snapshot(config, json.dumps(payload))


def test_independent_known_row_formulas_and_prior_ranks():
    bars, config = _recorded(), RegimeConfig(version='baseline-v1')
    result = classify_market_regime(bars, config=config, include_audit=True)
    opened, high, low, close, volume = (bars[name].to_numpy() for name in ('open', 'high', 'low', 'close', 'volume'))
    index = 421
    fast, slow = slice(index - 19, index + 1), slice(index - 59, index + 1)
    returns = np.log(close[1:] / close[:-1])
    true_range = np.maximum.reduce([high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])])
    rs = np.log(high / close) * np.log(high / opened) + np.log(low / close) * np.log(low / opened)
    atr = true_range[index - 20:index].mean()
    vol_fast, vol_slow = np.sqrt(max(rs[fast].mean(), 0)), np.sqrt(max(rs[slow].mean(), 0))
    fast_ma, slow_ma = close[fast].mean(), close[slow].mean()
    prior_high, prior_low = high[index - 60:index].max(), low[index - 60:index].min()
    prior_returns = returns[index - 20:index]
    expected = [returns[index - 1], true_range[index - 1], atr, atr / close[index], rs[index],
                vol_fast, vol_slow, np.log((vol_fast + config.eps) / (vol_slow + config.eps)),
                fast_ma, slow_ma, (fast_ma - slow_ma) / (atr + config.eps),
                np.log((fast_ma + config.eps) / (close[index - 39:index - 19].mean() + config.eps)) / 20,
                prior_high, prior_low, (prior_high - prior_low) / (atr + config.eps),
                np.clip((close[index] - prior_low) / (prior_high - prior_low), 0, 1),
                max(close[index] - prior_high, 0) / (atr + config.eps),
                max(prior_low - close[index], 0) / (atr + config.eps),
                abs(prior_returns.sum()) / (np.abs(prior_returns).sum() + config.eps),
                np.log(((close * volume)[fast].mean() + config.eps) / ((close * volume)[slow].mean() + config.eps))]
    np.testing.assert_allclose(result.row(index)[12:32], expected, rtol=1e-12, atol=1e-12)
    sources = [result['trend_score_20_60'].abs(), result['trend_slope_20'], result['vol_rs_20'],
               result['vol_expansion_20_60'], result['breakout_up_atr'], result['breakout_down_atr'],
               1 - result['coherence_20'], result['participation_20_60']]
    for name, values in zip(result.columns[32:], sources, strict=True):
        previous = values.head(index).drop_nulls().tail(252).to_numpy()
        current = values[index]
        assert len(previous) == 252
        expected_rank = (np.count_nonzero(previous < current) + .5 * np.count_nonzero(previous == current)) / 252
        assert result[name][index] == expected_rank


def test_partitions_and_future_observations_are_isolated():
    bars, config = _recorded(), RegimeConfig(version='baseline-v1')
    other_file = FIXTURE.with_name('spot_15m_20250101_20250531.parquet')
    other = pl.read_parquet(other_file, n_rows=600).with_columns(pl.lit(other_file.stem).alias('series_id'))
    columns = ['series_id', 'datetime', 'open', 'high', 'low', 'close', 'volume']
    interleaved = pl.concat([other.head(200).select(columns), bars.head(200).select(columns),
                            other.slice(200).select(columns), bars.slice(200).select(columns)])
    grouped = classify_market_regime(interleaved, config=config, include_audit=True)
    assert_frame_equal(grouped.filter(pl.col('series_id') == FIXTURE.stem),
                       classify_market_regime(bars, config=config, include_audit=True), check_exact=True)
    mutated = pl.concat([bars.head(421), bars.slice(421).with_columns(pl.col('volume').reverse())])
    assert_frame_equal(classify_market_regime(mutated, config=config, include_audit=True).head(421),
                       grouped.filter(pl.col('series_id') == FIXTURE.stem).head(421), check_exact=True)


def test_invalid_rows_are_visible_and_rejected_chunks_do_not_advance_state():
    bars, config = _recorded(), RegimeConfig(version='baseline-v1')
    missing = bars.with_row_index('_row').with_columns(pl.when(pl.col('_row') == 350).then(None).otherwise(pl.col('open')).alias('open')).drop('_row')
    result = classify_market_regime(missing, config=config, include_audit=True)
    assert result['fallback_reason'][350] == 'invalid_input:open'
    assert result['market_regime_code'][350] == 0
    assert result['stress_level'][350] is None and result['is_stress'][350] is None
    assert (result['market_regime_code'].slice(350, 61) == 0).all()
    assert result['market_regime_code'][411] != 0
    classifier = MarketRegimeClassifier(config)
    classifier.update(bars.head(353))
    snapshot = classifier.snapshot()
    for bad in (bars.slice(353, 10).reverse(), bars.head(1),
                bars.slice(353, 1).with_columns(pl.lit(None, dtype=pl.String).alias('series_id'))):
        with pytest.raises(ValueError):
            classifier.update(bad)
        assert classifier.snapshot() == snapshot
    assert_frame_equal(classifier.update(bars.slice(353)),
                       classify_market_regime(bars, config=config).slice(353), check_exact=True)

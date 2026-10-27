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


@pytest.fixture(scope='module')
def annual_audit():
    bars = _recorded(rows=None)
    return bars, classify_market_regime(bars, config=RegimeConfig(version='baseline-v1'), include_audit=True)


def test_recorded_all_states_counters_breakout_expiry_and_stress(annual_audit):
    bars, result = annual_audit
    codes = result['market_regime_code'].to_list()
    assert set(codes) == set(range(7))
    assert result.schema['stress_level'] == pl.Float32
    assert result.schema['market_regime_code'] == pl.UInt8
    assert result.schema['bars_in_regime'] == pl.UInt32
    count, boundary, direction = 0, None, None
    for index, row in enumerate(result.iter_rows(named=True)):
        changed = index == 0 or codes[index - 1] != codes[index]
        count = 1 if changed else count + 1
        assert row['bars_in_regime'] == count
        if row['market_regime_code'] in (5, 6):
            assert count <= 3
            if changed:
                direction = 'up' if row['market_regime_code'] == 5 else 'down'
                boundary = row['range_high_60_prev' if direction == 'up' else 'range_low_60_prev']
                assert result[f'breakout_{direction}_atr'][index - 1] < .25 <= row[f'breakout_{direction}_atr']
                assert row[f'rank_breakout_{direction}_252'] >= .95
                assert row['rank_participation_252'] >= .60
            else:
                assert bars['close'][index] > boundary if direction == 'up' else bars['close'][index] < boundary
        if row['stress_level'] is not None:
            assert row['stress_level'] == float(np.float32(max(row['rank_vol_rs_20_252'], row['rank_vol_expansion_252'])))
            assert row['is_stress'] == (row['stress_level'] >= .95)
        else:
            assert row['is_stress'] is None
    assert result['fallback_reason'].head(312).unique().to_list() == ['insufficient_history']
    assert result.filter(pl.col('fallback_reason') == 'finite_fallback').height > 0


def test_real_candidates_have_inclusive_entry_boundaries(annual_audit):
    from limen.features._regime_contract import AUDIT, RANKS, Regime
    from limen.features._regime_rules import _base

    _, result = annual_audit
    config = RegimeConfig(version='baseline-v1')
    for regime, direction in ((Regime.TREND_UP, 1), (Regime.TREND_DOWN, -1)):
        candidates = result.filter((pl.col('trend_score_20_60') * direction > 0)
                                   & (pl.col('trend_slope_20') * direction > 0)
                                   & (pl.col('coherence_20') >= config.coherence_entry)
                                   & (pl.col('rank_abs_trend_score_252') >= config.trend_rank_entry)
                                   & ((pl.col('rank_trend_slope_252') >= config.trend_slope_up) if direction == 1
                                      else (pl.col('rank_trend_slope_252') <= config.trend_slope_down)))
        row = candidates.row(0, named=True)
        audit, ranks = ({name: row[name] for name in names} for names in (AUDIT, RANKS))
        threshold = row['rank_abs_trend_score_252']
        assert _base(audit, ranks, replace(config, trend_rank_entry=threshold))[0] == regime
        assert _base(audit, ranks, replace(config, trend_rank_entry=np.nextafter(threshold, 1.)))[0] != regime
    row = result.filter((pl.col('rank_vol_rs_20_252') <= .35)
                        & (pl.col('rank_abs_trend_score_252') <= .5)).row(0, named=True)
    audit, ranks = ({name: row[name] for name in names} for names in (AUDIT, RANKS))
    threshold = row['rank_vol_rs_20_252']
    assert _base(audit, ranks, replace(config, range_vol_entry=threshold))[0] == Regime.RANGE_LOW_VOL
    assert _base(audit, ranks, replace(config, range_vol_entry=np.nextafter(threshold, 0.)))[0] != Regime.RANGE_LOW_VOL


def test_confirmation_and_hysteresis_use_observed_candidates(annual_audit):
    from limen.features._regime_contract import AUDIT, RANKS, Regime
    from limen.features._regime_rules import Decision, _base

    bars, result = annual_audit
    config = RegimeConfig(version='baseline-v1')
    rows = result.with_columns(bars['close']).iter_rows(named=True)
    previous = None
    for row in rows:
        if row['rank_abs_trend_score_252'] is None:
            continue
        audit, ranks = ({name: row[name] for name in names} for names in (AUDIT, RANKS))
        candidate = _base(audit, ranks, config)[0]
        if previous is not None and candidate in (Regime.TREND_UP, Regime.TREND_DOWN) and candidate == previous[3]:
            break
        previous = audit, ranks, row['close'], candidate
    else:
        pytest.fail('Recorded period has no consecutive trend candidates')
    state = Decision(regime=Regime.RANGE_CHOP, bars=1)
    assert state.step(previous[0], previous[1], previous[2], config, '')[0] == Regime.RANGE_CHOP
    assert state.pending == candidate and state.pending_count == 1
    assert state.step(audit, ranks, row['close'], config, '')[0] == candidate
    retained = replace(config, trend_rank_entry=min(1., ranks['rank_abs_trend_score_252'] + .1))
    from limen.features._regime_rules import _retained
    assert _base(audit, ranks, retained)[0] != candidate
    assert _retained(candidate, audit, ranks, retained)


@pytest.mark.parametrize('index', [0, 20, 312, 353, 599])
def test_checkpoint_replays_one_row_chunks_and_inactive_states(index):
    bars, config = _recorded(), RegimeConfig(version='baseline-v1')
    classifier = MarketRegimeClassifier(config)
    classifier.update(bars.head(index))
    restored = MarketRegimeClassifier.from_snapshot(config, classifier.snapshot())
    tail = bars.slice(index, 5)
    actual = pl.concat([restored.update(tail.slice(row, 1), include_audit=True) for row in range(tail.height)])
    assert_frame_equal(actual, classify_market_regime(bars.head(index + tail.height), config=config, include_audit=True).tail(tail.height), check_exact=True)


def test_configuration_domains_and_undefined_feature_fallback():
    from dataclasses import asdict

    config = RegimeConfig(version='baseline-v1')
    for name in asdict(config):
        bad_values = ('', 5) if name == 'version' else (True, float('nan'), float('inf'), -1)
        for value in bad_values:
            with pytest.raises(ValueError, match=name):
                replace(config, **{name: value})
    for changes in ({'trend_rank_exit': .8}, {'coherence_exit': .8}, {'range_vol_exit': .1},
                    {'range_trend_exit': .1}, {'trend_slope_down': .7}, {'eps': 10**400}):
        with pytest.raises(ValueError):
            replace(config, **changes)
    result = classify_market_regime(_recorded(), config=replace(config, eps=1e10), include_audit=True)
    assert result['fallback_reason'][60] == 'invalid_input:range_position_60'
    assert result['market_regime_code'].unique().to_list() == [0]
    assert config.config_hash != replace(config, confirmation_rows=3).config_hash


def test_checkpoint_rejects_malformed_state_and_counter_overflow():
    config = RegimeConfig(version='baseline-v1')
    classifier = MarketRegimeClassifier(config)
    classifier.update(_recorded().head(353))
    original = json.loads(classifier.snapshot())
    for field, value in (('bars', []), ('ranks', []), ('last_timestamp', None)):
        payload = json.loads(classifier.snapshot())
        payload['series'][FIXTURE.stem][field] = value
        with pytest.raises(ValueError):
            MarketRegimeClassifier.from_snapshot(config, json.dumps(payload))
    for field, value in (('regime', 7), ('bars', 0), ('pending_count', 2), ('boundary', 1.), ('previous_up', float('inf'))):
        payload = json.loads(classifier.snapshot())
        payload['series'][FIXTURE.stem]['decision'][field] = value
        with pytest.raises(ValueError):
            MarketRegimeClassifier.from_snapshot(config, json.dumps(payload))
    original['series'][FIXTURE.stem]['decision'].update(regime=0, bars=2**32 - 1, pending=None, pending_count=0, boundary=None)
    restored = MarketRegimeClassifier.from_snapshot(config, json.dumps(original))
    snapshot = restored.snapshot()
    invalid = _recorded().slice(353, 1).with_columns(pl.lit(None, dtype=pl.Float64).alias('open'))
    with pytest.raises(ValueError, match='UInt32'):
        restored.update(invalid)
    assert restored.snapshot() == snapshot


def test_breakout_checkpoint_and_prior_tied_rank_window(annual_audit):
    bars, result = annual_audit
    config = RegimeConfig(version='baseline-v1')
    index = result.with_row_index('_row').filter(pl.col('market_regime_code').is_in([5, 6]))['_row'][0]
    classifier = MarketRegimeClassifier(config)
    classifier.update(bars.head(index + 1))
    snapshot = classifier.snapshot()
    payload = json.loads(snapshot)['series'][FIXTURE.stem]
    assert len(payload['bars']) == 61 and all(len(history) <= 252 for history in payload['ranks'])
    assert payload['decision']['boundary'] is not None
    restored = MarketRegimeClassifier.from_snapshot(config, snapshot)
    assert_frame_equal(restored.update(bars.slice(index + 1, 5), include_audit=True),
                       result.slice(index + 1, 5), check_exact=True)
    for index in (313, 421, 7000):
        values = result['breakout_up_atr']
        previous = values.head(index).drop_nulls().tail(252).to_numpy()
        current = values[index]
        assert np.count_nonzero(previous == current) > 0
        expected = (np.count_nonzero(previous < current) + .5 * np.count_nonzero(previous == current)) / 252
        assert result['rank_breakout_up_252'][index] == expected


def test_feature_fingerprint_uses_interpreter_independent_source_bytes():
    import hashlib
    from limen.features._regime_contract import feature_hash

    digest = hashlib.sha256()
    directory = Path(__file__).resolve().parents[1] / 'limen/features'
    for name in ('_regime_contract.py', '_regime_audit.py', '_regime_ranks.py', '_regime_rules.py',
                 'classify_market_regime.py', '_regime_state.py'):
        digest.update(name.encode() + b'\0' + (directory / name).read_bytes() + b'\0')
    assert feature_hash() == digest.hexdigest()

import hashlib

import numpy as np
import polars as pl


def verify_helpers(result, output, inputs):
    from limen import features, indicators, transforms
    from limen.scalers.registry import SCALER_REGISTRY
    assert set(result['indicators']) == set(indicators.__all__)
    assert set(result['features']) == set(features.__all__)
    assert set(result['transforms']) == set(transforms.__all__)
    assert set(result['scalers']) == set(SCALER_REGISTRY)
    raw = pl.read_parquet(inputs / 'spot_1h_20240101_20241231.parquet', n_rows=1440)
    timestamps = set(raw['datetime'].to_list())
    for package in ('indicators', 'features'):
        for name, entry in result[package].items():
            if 'value' in entry:
                assert name == 'find_min_d' and 0 <= entry['value'] <= 2
            else:
                path = output / entry['path']
                assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
                frame = pl.read_parquet(path)
                assert frame.height == entry['rows'] > 0
                assert frame.columns == ['datetime', *[column for column in entry['columns'] if column != 'datetime']]
                assert frame['datetime'].is_sorted() and frame['datetime'].n_unique() == frame.height
                if name == 'conserved_flux_renormalization':
                    assert frame.height == 3
                else:
                    assert set(frame['datetime'].to_list()) <= timestamps
                    if package == 'indicators':
                        assert frame.height == raw.height
    close = raw['close'].to_numpy()
    roc = pl.read_parquet(output / result['indicators']['roc']['path'])['roc_10'].to_numpy()
    np.testing.assert_allclose(roc[10:], (close[10:] / close[:-10] - 1) * 100, rtol=1e-10, atol=1e-10)
    sma = pl.read_parquet(output / result['indicators']['sma']['path'])['close_sma_30'].to_numpy()
    np.testing.assert_allclose(sma[29:], np.convolve(close, np.ones(30) / 30, 'valid'), rtol=1e-10, atol=1e-8)
    regimes = pl.read_parquet(output / result['features']['classify_market_regime']['path'])
    assert regimes.height == raw.height and regimes['market_regime'].null_count() == 0
    assert (regimes['market_regime_code'].head(312) == 0).all()
    assert (regimes['market_regime_code'].slice(312) != 0).all()
    assert regimes['regime_config_version'].unique().to_list() == ['baseline-v1']
    assert regimes['series_id'].unique().to_list() == ['recorded-spot']
    expected_stress = pl.max_horizontal('rank_vol_rs_20_252', 'rank_vol_expansion_252').cast(pl.Float32)
    assert regimes.filter(pl.col('stress_level').is_not_null()).select(
        (pl.col('stress_level') == expected_stress).all()).item()
    training = raw.head(864)
    later = raw[864:]
    for name, entry in result['scalers'].items():
        assert entry['rows'] == later.height and entry['fitted_rows'] == training.height
        assert entry['fitted_state_unchanged'] is True
        scaled = pl.read_parquet(output / ('scaler-' + name + '.parquet'))
        assert scaled.columns == ['close', 'volume'] and scaled.height == later.height
        assert np.isfinite(scaled.to_numpy()).all()
        if name == 'robust':
            for column in scaled.columns:
                train = training[column].to_numpy()
                spread = np.quantile(train, .75, method='nearest') - np.quantile(train, .25, method='nearest')
                np.testing.assert_allclose(scaled[column], (later[column].to_numpy() - np.median(train)) / spread, rtol=1e-10, atol=1e-10)
    for column in ('close', 'volume'):
        values = training[column].to_numpy()
        zscore = pl.read_parquet(output / 'transform-zscore_transform.parquet')[column].to_numpy()
        np.testing.assert_allclose(zscore, (values - values.mean()) / values.std(ddof=1), rtol=1e-10, atol=1e-10)
        mad = pl.read_parquet(output / 'transform-mad_transform.parquet')[column].to_numpy()
        np.testing.assert_allclose(mad, (values - np.median(values)) / np.median(abs(values - np.median(values))), rtol=1e-10, atol=1e-10)
        winsor = pl.read_parquet(output / 'transform-winsorize_transform.parquet')[column].to_numpy()
        np.testing.assert_allclose(winsor, np.clip(values, *np.quantile(values, [.01, .99], method='nearest')), rtol=1e-10, atol=1e-10)
    shifted = pl.read_parquet(output / 'transform-shift_column_transform.parquet')
    assert shifted['close'][0] is None
    np.testing.assert_array_equal(shifted['close'].to_numpy()[1:], training['close'].to_numpy()[:-1])
    trimmed = pl.read_parquet(output / 'transform-quantile_trim_transform.parquet')
    mask = np.ones(training.height, dtype=bool)
    for column in ('close', 'volume'):
        values = training[column].to_numpy()
        low, high = np.quantile(values, [.005, .995], method='nearest')
        mask &= (values >= low) & (values <= high)
    assert trimmed['datetime'].to_list() == training.filter(pl.Series(mask))['datetime'].to_list()


def verify_targets(result, output, inputs):
    from limen import targets
    expected = {name for name in targets.__all__ if name.endswith('Target')} - {'TradeOutcomeTarget'}
    assert set(result) == expected
    raw = pl.read_parquet(inputs / 'spot_1h_20240101_20241231.parquet', n_rows=1440)
    partitions = [raw[:864], raw[864:1152], raw[1152:1440]]
    for name, entry in result.items():
        if entry['status'] == 'refused':
            assert name in ('ExitQualityTarget', 'RiskRewardRatioTarget', 'VolNormalizedReturnTarget', 'ForwardVolNormalizedReturnTarget')
            assert entry['reason']
            if name in ('ExitQualityTarget', 'RiskRewardRatioTarget'):
                artifact = entry['supplied_context']
                path = output / artifact['path']
                assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact['sha256']
                frame = pl.read_parquet(path)
                if name == 'ExitQualityTarget':
                    source = pl.read_parquet(inputs / 'recorded-exit-labels.parquet')
                    oracle = np.where(source['exit_net_return'].to_numpy() < 0, .2, .5)
                else:
                    source = pl.read_parquet(inputs / 'recorded-risk-labels.parquet')
                    oracle = source['capturable_breakout'].to_numpy() / (abs(source['max_drawdown'].to_numpy()) + .001)
                np.testing.assert_allclose(frame['label'], oracle, rtol=1e-10, atol=1e-10, equal_nan=True)
            continue
        assert entry['fitted_only_on_training'] is True
        for part, artifact in zip(partitions, entry['partitions'], strict=True):
            path = output / artifact['path']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact['sha256']
            frame = pl.read_parquet(path)
            assert frame['datetime'].to_list() == part['datetime'].to_list()
            assert frame['label'].null_count() == artifact['null_rows']
            if name == 'NextReturnTarget':
                close = part['close'].to_numpy()
                np.testing.assert_allclose(frame['label'].to_numpy()[:-3], (close[3:] / close[:-3] - 1) * 100, rtol=1e-10, atol=1e-10)
            if name == 'IdentityTarget':
                np.testing.assert_array_equal(frame['label'], part['close'])
            if name in ('NextBarUpTarget', 'NextBarDownTarget'):
                close = part['close'].to_numpy()
                oracle = close[1:] > close[:-1] if name == 'NextBarUpTarget' else close[1:] < close[:-1]
                np.testing.assert_array_equal(frame['label'].to_numpy()[:-1], oracle)
            if name == 'TripleBarrierTarget':
                assert set(frame['label'].drop_nulls()) <= {-1, 0, 1}


def verify_multiclass(result, inputs):
    raw = pl.read_parquet(inputs / 'spot_1h_20240101_20241231.parquet', n_rows=1440)
    train, evaluation = raw['close'].to_numpy()[:864], raw['close'].to_numpy()[864:1152]
    cutoffs = np.quantile(train[7:]/train[6:-1]-1, [.25, .75])
    np.testing.assert_allclose(result['cutoffs'], cutoffs, rtol=1e-12)
    truth = np.digitize(evaluation[7:]/evaluation[6:-1]-1, cutoffs)
    np.testing.assert_array_equal(result['truth'], truth)
    probabilities, predicted = np.asarray(result['probabilities']), np.asarray(result['predictions'])
    assert result['training_rows'] == 857 and result['evaluation_rows'] == len(truth) == 281
    assert result['classes'] == [0, 1, 2] and result['fitted_state_unchanged'] is True
    assert probabilities.shape == (281, 3) and np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
    np.testing.assert_array_equal(predicted, probabilities.argmax(axis=1))
    precisions, recalls, aucs = [], [], []
    for label in result['classes']:
        positive, called = truth == label, predicted == label
        hits = np.count_nonzero(positive & called)
        precisions.append(hits / called.sum() if called.any() else 0.)
        recalls.append(hits / positive.sum())
        differences = probabilities[positive, label, None] - probabilities[~positive, label]
        aucs.append(float(np.mean((differences > 0) + .5*(differences == 0))))
    assert result['metrics']['accuracy'] == float(np.mean(truth == predicted))
    np.testing.assert_allclose([result['metrics'][key] for key in ('precision', 'recall', 'auc')],
                               [np.mean(precisions), np.mean(recalls), np.mean(aucs)], atol=1e-12)
    assert result['missing_class_auc'] is None


def verify_rules(result, inputs):
    raw = pl.read_parquet(inputs / 'spot_1h_20240101_20241231.parquet')
    bars = raw.filter(pl.col('datetime').is_between(pl.datetime(2024, 1, 18, time_zone='UTC'), pl.datetime(2024, 1, 22, time_zone='UTC'), closed='left'))
    close, opening = bars['close'].to_numpy(), bars['open'].to_numpy()
    up = close > opening
    persistent = np.r_[False, up[1:] & up[:-1]]
    cross = np.r_[False, up[1:] & (close[:-1] <= opening[:-1])]
    recent = np.asarray([cross[max(0, i-2):i+1].any() for i in range(len(close))])
    rising = np.r_[np.zeros(3, dtype=bool), close[3:] > close[:-3]]
    assert result['assessment_rows'] == len(close) == 96 and result['history_resets_at_split'] is True
    assert {row['price_floor'] for row in result['rows']} == {40000., 41000.}
    for row in result['rows']:
        expected = (close >= row['price_floor']) & (persistent | recent) & rising & ~(close < opening)
        np.testing.assert_array_equal(row['signals'], expected.astype(int))
        assert row['positive_decisions'] == expected.sum() and 0 < expected.sum() < len(close)

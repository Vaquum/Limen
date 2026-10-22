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

"""Reconstruct chronological labels, Ridge coefficients, and residual composition."""
import hashlib

import numpy as np
import polars as pl
import pytest
from xgboost import XGBRegressor


def verify_residual(result, output, inputs):
    raw = pl.read_parquet(inputs / 'spot_1h_20240101_20241231.parquet', n_rows=1440)
    expected = {}
    for name, start, stop in (('train', 0, 864), ('validation', 864, 1152), ('test', 1152, 1440)):
        source = raw.slice(start, stop - start)
        close, opening = source['close'].to_numpy(), source['open'].to_numpy()
        expected[name] = (source['datetime'][12:-1], close[12:-1] / close[8:-5] - 1,
                          close[13:] / opening[13:] - 1)
    _, train_x, train_y = expected['train']
    z = (train_x - train_x.mean()) / train_x.std()
    coefficient = np.dot(z, train_y - train_y.mean()) / (np.dot(z, z) + 1.)
    residuals = train_y - (z * coefficient + train_y.mean())
    scores = []
    assert result['partition_rows'] == [851, 275, 275]
    assert set(result['artifacts']) == {str(depth) + '-' + name + '.parquet'
                                       for depth in (1, 2) for name in expected}
    for depth in (1, 2):
        correction = XGBRegressor(n_estimators=8, max_depth=depth, learning_rate=.1,
                                  random_state=42, n_jobs=1).fit(train_x.reshape(-1, 1), residuals)
        for name, (dates, x, y) in expected.items():
            path = output / (str(depth) + '-' + name + '.parquet')
            assert hashlib.sha256(path.read_bytes()).hexdigest() == result['artifacts'][path.name]
            frame = pl.read_parquet(path)
            assert frame['datetime'].to_list() == dates.to_list()
            np.testing.assert_allclose(frame['momentum'], x, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(frame['next_return'], y, rtol=1e-12, atol=1e-14)
            baseline = (x - train_x.mean()) / train_x.std() * coefficient + train_y.mean()
            np.testing.assert_allclose(frame['baseline'], baseline, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(frame['correction'], correction.predict(x.reshape(-1, 1)), rtol=1e-7, atol=1e-9)
            np.testing.assert_allclose(frame['forecast'], frame['baseline'] + frame['correction'], rtol=1e-12)
            positions = np.sign(frame['forecast'].to_numpy())
            score = float(np.mean(positions * y - abs(positions) * .0002))
            if name == 'validation':
                scores.append((score, depth))
                row = next(item for item in result['candidates'] if item['depth'] == depth)
                assert row['validation_net_return'] == pytest.approx(score, abs=1e-14)
            if name == 'test' and depth == result['selected_depth']:
                assert result['test_net_return'] == pytest.approx(score, abs=1e-14)
                assert result['test_rmse'] == pytest.approx(float(np.sqrt(np.mean((frame['forecast'].to_numpy() - y) ** 2))))
    assert result['selected_depth'] == sorted(scores, key=lambda item: (-item[0], item[1]))[0][1]

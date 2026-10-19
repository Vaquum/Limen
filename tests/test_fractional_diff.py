import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from limen.experiment import MLManifest
from limen.features.fractional_diff import _get_weights_ffd
from limen.features.fractional_diff import find_min_d
from limen.features.fractional_diff import fractional_diff
from limen.targets import RandomBinaryTarget
from limen.utils.adf_test import adf_test
from tests.utils.historical_data import get_cached_spot_klines_2h


INVALID_PARAMETERS = [
    (value, 0.1, 'd') for value in (float('nan'), float('inf'), -float('inf'), -0.5)
] + [
    (d, value, 'threshold')
    for d in (0.0, 0.5)
    for value in (float('nan'), float('inf'), -float('inf'), 0.0, -0.1)
]


def test_fractional_diff_d_zero_copies():

    df = pl.DataFrame({'close': [100.0, 101.0, 102.0, 103.0, 104.0]})
    result = fractional_diff(df, d=0.0, cols=['close'])
    assert 'close' in result.columns
    assert 'close_fracdiff' in result.columns
    assert result['close_fracdiff'].to_list() == df['close'].to_list()


def test_fractional_diff_d_one_equals_first_diff():

    rng = np.random.RandomState(42)
    values = np.cumsum(rng.normal(0, 1, 100)) + 100
    df = pl.DataFrame({'price': values})
    result = fractional_diff(df, d=1.0, cols=['price'])

    assert 'price' in result.columns
    assert 'price_fracdiff' in result.columns
    expected = np.diff(values, prepend=np.nan)
    actual = result['price_fracdiff'].to_numpy()
    valid = ~np.isnan(actual) & ~np.isnan(expected)
    assert np.allclose(actual[valid], expected[valid], atol=1e-10)


def test_fractional_diff_weights_convergence():

    weights = _get_weights_ffd(0.5, threshold=1e-5)
    assert len(weights) > 1
    assert weights[-1] == 1.0
    assert all(abs(w) >= 1e-5 for w in weights)


def test_fractional_diff_preserves_original_columns():

    rng = np.random.RandomState(42)
    df = pl.DataFrame({
        'close': rng.normal(100, 5, 50),
        'volume': rng.randint(1000, 5000, 50),
    })
    result = fractional_diff(df, d=0.5, cols=['close'], threshold=1e-2)
    assert result['close'].to_list() == df['close'].to_list()
    assert result['volume'].to_list() == df['volume'].to_list()
    assert 'close_fracdiff' in result.columns


def test_fractional_diff_cols_none_raises():

    df = pl.DataFrame({'close': [100.0, 101.0]})
    try:
        fractional_diff(df, d=0.5)
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'cols' in str(e)


def test_fractional_diff_negative_d_raises():

    df = pl.DataFrame({'close': [100.0, 101.0]})
    try:
        fractional_diff(df, d=-0.5, cols=['close'])
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'non-negative' in str(e)


def test_fractional_diff_lazyframe():

    df = pl.DataFrame({'close': np.cumsum(np.random.RandomState(42).normal(0, 1, 50)) + 100})
    lazy = df.lazy()
    result = fractional_diff(lazy, d=0.4, cols=['close'], threshold=1e-2)
    assert isinstance(result, pl.LazyFrame)
    collected = result.collect()
    assert collected.height == 50
    assert 'close_fracdiff' in collected.columns


def test_adf_test_stationary_series():

    rng = np.random.RandomState(42)
    stationary = pl.Series(rng.normal(0, 1, 500))
    result = adf_test(stationary)
    assert result.stationary is True
    assert result.p_value < 0.05


def test_adf_test_nonstationary_series():

    t = np.arange(500, dtype=np.float64)
    trend = pl.Series(t + 0.1 * np.sin(t))
    result = adf_test(trend)
    assert result.stationary is False
    assert result.p_value > 0.05


def test_find_min_d_returns_valid_d():

    t = np.arange(1000, dtype=np.float64)
    prices = t + 0.5 * np.sin(t * 0.1)
    df = pl.DataFrame({'price': prices})

    d = find_min_d(df, col='price', d_start=0.1, step=0.1, threshold=1e-2)
    assert 0.0 < d <= 1.0


def test_fractional_diff_empty_dataframe():

    df = pl.DataFrame({'close': pl.Series([], dtype=pl.Float64)})
    result = fractional_diff(df, d=0.5, cols=['close'])
    assert result.height == 0


def test_fractional_diff_missing_column_skips():

    df = pl.DataFrame({'close': [100.0, 101.0, 102.0]})
    result = fractional_diff(df, d=0.0, cols=['close', 'nonexistent'])
    assert 'close_fracdiff' in result.columns
    assert 'nonexistent_fracdiff' not in result.columns


def test_adf_test_empty_series():

    empty = pl.Series([], dtype=pl.Float64)
    result = adf_test(empty)
    assert result.stationary is False
    assert result.p_value == 1.0


def test_adf_test_constant_series():

    constant = pl.Series([5.0] * 100)
    result = adf_test(constant)
    assert result.stationary is False


def test_find_min_d_step_zero_raises():

    df = pl.DataFrame({'price': [100.0, 101.0, 102.0]})
    try:
        find_min_d(df, col='price', step=0.0)
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'step' in str(e)


def test_find_min_d_small_data_skips():

    df = pl.DataFrame({'price': [100.0, 101.0, 102.0]})
    d = find_min_d(df, col='price', d_start=0.5, step=0.1)
    assert d == 1.0


def test_fractional_diff_manifest_integration():

    manifest = (MLManifest()
        .set_test_data_source(
            method=get_cached_spot_klines_2h,
            params={'n_rows': 500}
        )
        .set_split_config(3, 1, 1)
        .add_feature(fractional_diff, d=0.4, cols=['close'], threshold=1e-2)
        .with_target_label('outcome', RandomBinaryTarget)
    )
    raw_data = manifest.fetch_test_data()
    data = manifest.prepare_data(raw_data, {})
    assert data['x_train'].height > 0


@pytest.mark.parametrize('d, threshold, parameter', INVALID_PARAMETERS)
def test_fractional_diff_rejects_invalid_parameters_before_weights(monkeypatch, d, threshold, parameter):
    data = pl.read_csv(Path(__file__).parent / 'fixtures' / 'historical_data_spot_2h.csv')

    def fail_weights(*_args, **_kwargs):
        raise AssertionError('Invalid parameters reached weight generation')

    monkeypatch.setitem(fractional_diff.__globals__, '_get_weights_ffd', fail_weights)
    with pytest.raises(ValueError, match=f'fractional_diff {parameter} must be'):
        fractional_diff(data, d=d, cols=['close'], threshold=threshold)


def test_fractional_diff_weights_reject_invalid_parameters():
    subprocess.run(
        [sys.executable, '-c', '''
import pytest
from limen.features.fractional_diff import _get_weights_ffd
from tests.test_fractional_diff import INVALID_PARAMETERS

for d, threshold, parameter in INVALID_PARAMETERS:
    with pytest.raises(ValueError, match=f'fractional_diff {parameter} must be'):
        _get_weights_ffd(d, threshold)
'''],
        check=True,
        timeout=5,
    )


def test_fractional_diff_weights_reject_overflow_before_append():
    subprocess.run(
        [sys.executable, '-W', 'error', '-c', '''
import numpy as np
import pytest
from limen.features.fractional_diff import _get_weights_ffd

for d in (1e308, np.float64(1e308)):
    with pytest.raises(ValueError, match='fractional_diff weight must be finite') as raised:
        _get_weights_ffd(d, threshold=0.1)

    traceback = raised.value.__traceback__
    assert traceback is not None
    while traceback.tb_next is not None:
        traceback = traceback.tb_next
    assert traceback.tb_frame.f_code is _get_weights_ffd.__code__
    assert traceback.tb_frame.f_locals['weights'] == [1.0, -1e308]
    assert not np.isfinite(traceback.tb_frame.f_locals['w'])
'''],
        check=True,
        timeout=5,
    )


@pytest.mark.parametrize('d', [0.0, 1.0, 2.0, 0.5])
@pytest.mark.parametrize('lazy', [False, True])
def test_fractional_diff_finite_orders_match_recorded_data(d, lazy):
    data = pl.read_csv(Path(__file__).parent / 'fixtures' / 'historical_data_spot_2h.csv')
    close = data['close'].to_list()
    expected = {
        0.0: close,
        1.0: [None, close[1] - close[0], close[2] - close[1]],
        2.0: [None, None, close[2] - 2 * close[1] + close[0]],
        0.5: [None, None, close[2] - 0.5 * close[1] - 0.125 * close[0]],
    }[d]

    result = fractional_diff(data.lazy() if lazy else data, d=d, cols=['close'], threshold=0.1)
    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    if lazy:
        result = result.collect()
    assert result.select(data.columns).equals(data)
    actual = result['close_fracdiff'].to_list()
    assert [value is None for value in actual] == [value is None for value in expected]
    np.testing.assert_allclose(
        [value for value in actual if value is not None],
        [value for value in expected if value is not None],
        rtol=0,
        atol=1e-10,
    )

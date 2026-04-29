import numpy as np
import polars as pl

from limen.experiment import Manifest
from limen.features.fractional_diff import _get_weights_ffd
from limen.features.fractional_diff import find_min_d
from limen.features.fractional_diff import fractional_diff
from limen.targets import RandomBinaryTarget
from limen.utils.adf_test import adf_test
from tests.utils.historical_data import get_cached_spot_klines_2h


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

    manifest = (Manifest()
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

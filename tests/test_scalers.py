from pathlib import Path

import numpy as np
import polars as pl
import pytest

from limen.experiment import MLManifest
from limen.indicators.avgprice import avgprice
from limen.indicators.roc import roc
from limen.scalers.linear_scaler import DEFAULT_SCALING_RULES
from limen.scalers.linear_scaler import LinearScaler
from limen.scalers.linear_scaler import build_rules
from limen.scalers.linear_scaler import get_scaling_rule
from limen.scalers.linear_scaler import inverse_transform as inverse_linear_scaler
from limen.scalers.causal_rolling_robust_scaler import CausalRollingRobustScaler
from limen.scalers.rank_gauss_scaler import RankGaussScaler
from limen.scalers.registry import SCALER_REGISTRY
from limen.scalers.robust_scaler import RobustScaler
from limen.targets import RandomBinaryTarget
from tests.utils.historical_data import get_cached_spot_klines_2h


def _make_train_data() -> pl.DataFrame:

    rng = np.random.RandomState(42)
    return pl.DataFrame({
        'datetime': pl.datetime_range(pl.lit('2025-01-01').cast(pl.Date), pl.lit('2025-04-10').cast(pl.Date), eager=True)[:100],
        'price': rng.normal(100, 10, 100),
        'volume': rng.lognormal(10, 1, 100),
        'ratio': rng.uniform(0, 1, 100),
    })


def test_robust_scaler_fit_and_transform():

    train = _make_train_data()
    scaler = RobustScaler(train)
    transformed = scaler.transform(train)

    assert 'price' in transformed.columns
    assert 'volume' in transformed.columns
    assert 'datetime' in transformed.columns

    price_median = transformed['price'].median()
    assert abs(price_median) < 0.5


def test_robust_scaler_custom_quantile_range():

    train = _make_train_data()
    scaler_default = RobustScaler(train)
    scaler_wide = RobustScaler(train, quantile_range=(0.1, 0.9))

    assert scaler_default.iqrs['price'] != scaler_wide.iqrs['price']


def test_robust_scaler_zero_iqr_guard():

    train = pl.DataFrame({
        'constant': [5.0] * 50,
        'normal': np.random.RandomState(42).normal(0, 1, 50),
    })
    scaler = RobustScaler(train)
    assert scaler.iqrs['constant'] == 1.0

    transformed = scaler.transform(train)
    assert not transformed['constant'].is_nan().any()


def test_robust_scaler_preserves_datetime():

    train = _make_train_data()
    scaler = RobustScaler(train)
    assert 'datetime' not in scaler.medians


def test_robust_scaler_outlier_resilience():

    base = np.random.RandomState(42).normal(100, 5, 99)
    with_outlier = np.append(base, [10000.0])
    train = pl.DataFrame({'val': with_outlier})
    scaler = RobustScaler(train)

    assert abs(scaler.medians['val'] - np.median(base)) < 5


def test_rank_gauss_output_approximately_normal():

    train = _make_train_data()
    scaler = RankGaussScaler(train)
    transformed = scaler.transform(train)

    price_vals = transformed['price'].to_numpy()
    assert abs(np.mean(price_vals)) < 0.5
    assert abs(np.std(price_vals) - 1.0) < 0.5


def test_rank_gauss_preserves_rank_order():

    train = pl.DataFrame({'val': [1.0, 2.0, 3.0, 4.0, 5.0] * 20})
    scaler = RankGaussScaler(train)
    transformed = scaler.transform(train)

    original = train['val'].to_numpy()
    result = transformed['val'].to_numpy()
    for i in range(len(original) - 1):
        if original[i] < original[i + 1]:
            assert result[i] <= result[i + 1]


def test_rank_gauss_handles_unseen_values():

    train = pl.DataFrame({'val': np.random.RandomState(42).normal(0, 1, 100)})
    scaler = RankGaussScaler(train)
    test = pl.DataFrame({'val': [1000.0, -1000.0]})
    transformed = scaler.transform(test)

    assert not np.any(np.isinf(transformed['val'].to_numpy()))
    assert not np.any(np.isnan(transformed['val'].to_numpy()))


def test_rank_gauss_preserves_datetime():

    train = _make_train_data()
    scaler = RankGaussScaler(train)
    assert 'datetime' not in scaler.quantiles


def test_rank_gauss_constant_column():

    train = pl.DataFrame({'constant': [5.0] * 50})
    scaler = RankGaussScaler(train)
    transformed = scaler.transform(train)
    const_vals = transformed['constant'].to_numpy()
    assert not np.any(np.isnan(const_vals))
    assert np.allclose(const_vals, 5.0)


def test_rank_gauss_integer_column():

    train = pl.DataFrame({'int_col': [1, 2, 3, 4, 5] * 20})
    scaler = RankGaussScaler(train)
    transformed = scaler.transform(train)
    assert not np.any(np.isnan(transformed['int_col'].to_numpy()))


def test_robust_scaler_all_null_column():

    train = pl.DataFrame({
        'all_null': pl.Series([None, None, None], dtype=pl.Float64),
        'normal': [1.0, 2.0, 3.0],
    })
    scaler = RobustScaler(train)
    assert 'all_null' not in scaler.medians
    assert 'normal' in scaler.medians


def test_causal_rolling_robust_scaler_fit_and_transform():

    train = _make_train_data()
    scaler = CausalRollingRobustScaler(train, window=20, min_samples=5)
    transformed = scaler.transform(train)

    assert 'price' in transformed.columns
    assert 'datetime' in transformed.columns
    assert transformed.height == train.height
    assert not transformed['price'].is_nan().any()


def test_causal_rolling_robust_scaler_no_lookahead():

    train = _make_train_data()
    scaler = CausalRollingRobustScaler(train, window=20, min_samples=5)
    full = scaler.transform(train)

    for t in (10, 40, 80):
        prefix = scaler.transform(train.head(t + 1))
        assert abs(full['price'][t] - prefix['price'][t]) < 1e-9


def test_causal_rolling_robust_scaler_warmup_uses_fallback():

    train = _make_train_data()
    scaler = CausalRollingRobustScaler(train, window=20, min_samples=5)
    transformed = scaler.transform(train)

    raw = (train['price'][0] - scaler.medians['price']) / scaler.iqrs['price']
    expected = max(-scaler.clip, min(scaler.clip, raw))
    assert abs(transformed['price'][0] - expected) < 1e-9


def test_causal_rolling_robust_scaler_clip_bound():

    train = _make_train_data()
    scaler = CausalRollingRobustScaler(train, window=20, min_samples=5, clip=3.0)
    transformed = scaler.transform(train)

    price = transformed['price'].to_numpy()
    assert price.max() <= 3.0 + 1e-9
    assert price.min() >= -3.0 - 1e-9


def test_causal_rolling_robust_scaler_zero_iqr_guard():

    train = pl.DataFrame({
        'constant': [5.0] * 80,
        'normal': np.random.RandomState(42).normal(0, 1, 80),
    })
    scaler = CausalRollingRobustScaler(train, window=20, min_samples=5)
    assert scaler.iqrs['constant'] == 1.0

    transformed = scaler.transform(train)
    assert not transformed['constant'].is_nan().any()
    assert not np.any(np.isinf(transformed['constant'].to_numpy()))


def test_causal_rolling_robust_scaler_preserves_datetime():

    train = _make_train_data()
    scaler = CausalRollingRobustScaler(train)
    assert 'datetime' not in scaler.medians

    transformed = scaler.transform(train)
    assert transformed['datetime'].equals(train['datetime'])


def test_causal_rolling_robust_scaler_all_null_column():

    train = pl.DataFrame({
        'all_null': pl.Series([None] * 40, dtype=pl.Float64),
        'normal': np.random.RandomState(42).normal(0, 1, 40),
    })
    scaler = CausalRollingRobustScaler(train, window=10, min_samples=3)
    assert 'all_null' not in scaler.medians
    assert 'normal' in scaler.medians

    unchanged = scaler.transform(pl.DataFrame({'other': [1.0, 2.0, 3.0]}))
    assert unchanged['other'].to_list() == [1.0, 2.0, 3.0]


def test_causal_rolling_robust_scaler_invalid_window():

    try:
        CausalRollingRobustScaler(pl.DataFrame({'a': [1.0, 2.0, 3.0]}), window=1)
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'window' in str(e)


def test_causal_rolling_robust_scaler_invalid_clip():

    try:
        CausalRollingRobustScaler(pl.DataFrame({'a': [1.0, 2.0, 3.0]}), clip=0.0)
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'clip' in str(e)


def test_causal_rolling_robust_scaler_invalid_quantile_range():

    try:
        CausalRollingRobustScaler(
            pl.DataFrame({'a': [1.0, 2.0, 3.0]}), quantile_range=(0.9, 0.1),
        )
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'quantile_range' in str(e)


def test_causal_rolling_robust_scaler_invalid_min_samples():

    try:
        CausalRollingRobustScaler(
            pl.DataFrame({'a': [1.0, 2.0, 3.0]}), window=10, min_samples=50,
        )
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'min_samples' in str(e)


def _make_manifest_with_scaler_from_params() -> MLManifest:

    return (MLManifest()
        .set_test_data_source(
            method=get_cached_spot_klines_2h,
            params={'n_rows': 500}
        )
        .set_split_config(3, 1, 1)
        .add_indicator(
            lambda df: df.with_columns((pl.col('close').pct_change()).alias('roc')),
        )
        .set_scaler_from_params('scaler_type')
        .with_target_label('outcome', RandomBinaryTarget)
    )


def test_robust_scaler_inverse_transform():

    from limen.scalers.robust_scaler import inverse_transform
    train = _make_train_data()
    scaler = RobustScaler(train)
    transformed = scaler.transform(train)
    restored = inverse_transform(transformed, scaler)
    assert np.allclose(
        restored['price'].to_numpy(),
        train['price'].to_numpy(),
        atol=1e-10,
    )


def test_rank_gauss_inverse_transform():

    from limen.scalers.rank_gauss_scaler import inverse_transform
    train = _make_train_data()
    scaler = RankGaussScaler(train)
    transformed = scaler.transform(train)
    restored = inverse_transform(transformed, scaler)
    assert np.allclose(
        restored['price'].to_numpy(),
        train['price'].to_numpy(),
        atol=0.5,
    )


def test_robust_scaler_invalid_quantile_range():

    try:
        RobustScaler(pl.DataFrame({'a': [1.0, 2.0, 3.0]}), quantile_range=(0.9, 0.1))
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'quantile_range' in str(e)


def test_rank_gauss_invalid_n_quantiles():

    try:
        RankGaussScaler(pl.DataFrame({'a': [1.0, 2.0, 3.0]}), n_quantiles=0)
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'n_quantiles' in str(e)


def test_scaler_factory_selects_all_types():

    manifest = _make_manifest_with_scaler_from_params()
    raw_data = manifest.fetch_test_data()
    for scaler_type in ('linear', 'logreg', 'robust', 'rank_gauss', 'causal_rolling_robust'):
        data = manifest.prepare_data(raw_data, {'scaler_type': scaler_type})
        assert data['x_train'].height > 0


def test_scaler_factory_unknown_raises():

    manifest = _make_manifest_with_scaler_from_params()
    raw_data = manifest.fetch_test_data()
    try:
        manifest.prepare_data(raw_data, {'scaler_type': 'invalid'})
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'Unknown scaler type' in str(e)


def test_scaler_factory_registry_has_all():

    assert 'linear' in SCALER_REGISTRY
    assert 'logreg' in SCALER_REGISTRY
    assert 'robust' in SCALER_REGISTRY
    assert 'rank_gauss' in SCALER_REGISTRY
    assert 'causal_rolling_robust' in SCALER_REGISTRY


@pytest.fixture(scope='module')
def recorded_linear_scaler_data():
    path = Path(__file__).parent / 'fixtures' / 'spot_15m_20250101_20250531.parquet'
    bars = pl.read_parquet(path, n_rows=512)
    return avgprice(roc(roc(bars, period=1), period=2)).with_columns(pl.col('roc_1').alias('roc'))


def test_linear_scaler_generic_override_keeps_recorded_values(recorded_linear_scaler_data):
    train = recorded_linear_scaler_data.select('roc_1', 'roc_2')
    rules = build_rules(overrides={'none': ['roc_1']})
    assert list(rules)[0] == '^roc_1$'
    assert get_scaling_rule('roc_1', rules) == 'none'
    assert get_scaling_rule('roc_2', rules) == 'standard'

    scaler = LinearScaler(train, rules=rules)
    result = scaler.transform(train)
    assert train['roc_1'][1] == pytest.approx(0.11288096524970648)
    assert result['roc_1'].equals(train['roc_1'])
    assert 'roc_1' not in scaler.means and 'roc_1' not in scaler.stds
    assert 'roc_2' in scaler.means and 'roc_2' in scaler.stds

    values = train['roc_2'].to_numpy()
    expected = (values - np.nanmean(values)) / np.nanstd(values)
    np.testing.assert_allclose(result['roc_2'].to_numpy(), expected, rtol=0, atol=1e-12, equal_nan=True)
    restored = inverse_linear_scaler(result, scaler)
    assert restored['roc_1'].equals(train['roc_1'])
    np.testing.assert_allclose(restored['roc_2'].to_numpy(), values, atol=1e-10, equal_nan=True)


@pytest.mark.parametrize(('rule', 'column'), [
    ('standard', 'avgprice'),
    ('none', 'volume'),
    ('divide_100', 'open'),
    ('none', 'roc'),
])
def test_linear_scaler_override_transform_and_inverse(recorded_linear_scaler_data, rule, column):
    rules = build_rules(overrides={rule: [column]})
    pattern = f'^{column}$'
    assert list(rules)[0] == pattern
    assert list(rules)[1:] == [name for name in DEFAULT_SCALING_RULES if name != pattern]
    assert get_scaling_rule(column, rules) == rule

    train = recorded_linear_scaler_data.select(column)
    scaler = LinearScaler(train, rules=rules)
    transformed = scaler.transform(train)
    values = train[column].to_numpy()
    actual = transformed[column].to_numpy()

    if rule == 'standard':
        expected = (values - np.nanmean(values)) / np.nanstd(values)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True)
        assert column in scaler.means and column in scaler.stds
    elif rule == 'divide_100':
        np.testing.assert_allclose(actual, values / 100, rtol=0, atol=1e-12, equal_nan=True)
        assert column not in scaler.means and column not in scaler.stds
    else:
        assert transformed[column].equals(train[column])
        assert column not in scaler.means and column not in scaler.stds

    restored = inverse_linear_scaler(transformed, scaler)
    np.testing.assert_allclose(restored[column].to_numpy(), values, rtol=0, atol=1e-9, equal_nan=True)


def test_linear_scaler_override_order_and_input_ownership():
    base = {'.*': 'none', '^roc_1$': 'standard', '^roc_2$': 'standard'}
    overrides = {'none': ['roc_1'], 'divide_100': ['open']}
    rules = build_rules(overrides=overrides, base_rules=base)
    assert list(rules) == ['^roc_1$', '^open$', '.*', '^roc_2$']
    assert get_scaling_rule('roc_1', rules) == 'none'
    assert get_scaling_rule('open', rules) == 'divide_100'
    assert get_scaling_rule('unmatched', rules) == 'none'
    assert list(base) == ['.*', '^roc_1$', '^roc_2$']
    assert base['^roc_1$'] == 'standard' and overrides == {'none': ['roc_1'], 'divide_100': ['open']}
    assert rules is not base and DEFAULT_SCALING_RULES['.*'] == 'none'
    assert list(build_rules(base_rules={})) == list(DEFAULT_SCALING_RULES)


def test_linear_scaler_existing_regex_key_replacement():
    rules = build_rules(overrides={'none': [r'roc(_\d+)?']})
    assert len(rules) == len(DEFAULT_SCALING_RULES)
    assert list(rules)[0] == r'^roc(_\d+)?$'
    assert get_scaling_rule('roc', rules) == 'none'
    assert get_scaling_rule('roc_2', rules) == 'none'
    assert get_scaling_rule('atr_14', rules) == 'standard'
    assert get_scaling_rule('roc_1', build_rules({'standard': ['roc_1'], 'none': ['roc_1']})) == 'none'

import numpy as np
import polars as pl

from limen.data import HistoricalData
from limen.experiment import Manifest
from limen.scalers.rank_gauss_scaler import RankGaussScaler
from limen.scalers.registry import SCALER_REGISTRY
from limen.scalers.robust_scaler import RobustScaler


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
        'normal': np.random.normal(0, 1, 50),
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

    base = np.random.normal(100, 5, 99)
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

    train = pl.DataFrame({'val': np.random.normal(0, 1, 100)})
    scaler = RankGaussScaler(train)
    test = pl.DataFrame({'val': [1000.0, -1000.0]})
    transformed = scaler.transform(test)

    assert not np.any(np.isinf(transformed['val'].to_numpy()))
    assert not np.any(np.isnan(transformed['val'].to_numpy()))


def test_rank_gauss_preserves_datetime():

    train = _make_train_data()
    scaler = RankGaussScaler(train)
    assert 'datetime' not in scaler._quantiles


def test_rank_gauss_constant_column():

    train = pl.DataFrame({'constant': [5.0] * 50})
    scaler = RankGaussScaler(train)
    transformed = scaler.transform(train)
    assert not np.any(np.isnan(transformed['constant'].to_numpy()))


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


def _make_manifest_with_scaler_from_params() -> Manifest:

    return (Manifest()
        .set_test_data_source(method=HistoricalData._get_data_for_test, params={'n_rows': 500})
        .set_split_config(3, 1, 1)
        .add_indicator(
            lambda df: df.with_columns((pl.col('close').pct_change()).alias('roc')),
        )
        .set_scaler_from_params('scaler_type')
        .with_target('outcome')
            .add_transform(lambda data: data.with_columns(
                pl.Series('outcome', np.random.randint(0, 2, size=data.height))
            ))
            .add_transform(lambda data: data[:-1])
            .done()
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
    for scaler_type in ('linear', 'logreg', 'robust', 'rank_gauss'):
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

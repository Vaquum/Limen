import numpy as np
import polars as pl

from limen.data import HistoricalData
from limen.experiment import Manifest
from limen.sfd.foundational_sfd import logreg_binary as logreg_sfd
from limen.transforms import shift_column_transform
from tests.utils.historical_data import get_cached_spot_klines_2h


def _make_manifest() -> Manifest:

    return (Manifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'start_date_limit': '2025-01-01'}
        )
        .set_test_data_source(
            method=get_cached_spot_klines_2h,
            params={'n_rows': 5000}
        )
        .set_split_config(3, 1, 1)
        .set_required_bar_columns([
            'datetime', 'high', 'low', 'close', 'volume', 'maker_ratio',
            'no_of_trades'
        ])
        .with_target_label('outcome')
            .add_transform(lambda data: data.with_columns(
                pl.Series('outcome', np.random.randint(0, 2, size=data.height))
            ))
            .add_transform(lambda data: data[:-1])
            .done()
    )


def _prepare_data_with_manifest(manifest: Manifest) -> dict:

    raw_data = manifest.fetch_test_data()
    round_params = {'bar_type': 'base'}
    return manifest.prepare_data(raw_data, round_params)


def _make_shifted_target_manifest() -> Manifest:

    return (Manifest()
        .set_split_config(3, 1, 4)
        .with_target_label('target')
            .add_transform(lambda data: data.with_columns(
                (pl.col('close') > pl.col('open')).cast(pl.Int8).alias('target')
            ))
            .add_transform(shift_column_transform, shift=-1, column='target_column')
            .done()
    )


def test_price_data_for_backtest_has_ohlc_columns() -> None:

    manifest = _make_manifest()
    data = _prepare_data_with_manifest(manifest)
    price_df = data['price_data_for_backtest']

    for col in ['datetime', 'open', 'high', 'low', 'close']:
        assert col in price_df.columns, f"Missing column: {col}"


def test_price_data_for_backtest_row_count_matches_test() -> None:

    manifest = _make_manifest()
    data = _prepare_data_with_manifest(manifest)
    price_df = data['price_data_for_backtest']

    assert price_df.height == data['x_test'].height


def test_price_data_for_backtest_datetime_alignment() -> None:

    manifest = _make_manifest()
    data = _prepare_data_with_manifest(manifest)
    price_df = data['price_data_for_backtest']
    alignment = data['_alignment']

    price_datetimes = price_df['datetime'].to_list()

    # datetimes are monotonically increasing (correct order)
    assert price_datetimes == sorted(price_datetimes)

    # first and last datetime match the test split boundaries
    assert price_datetimes[0] == alignment['first_test_datetime']
    assert price_datetimes[-1] == alignment['last_test_datetime']


def test_shifted_target_pipeline_keeps_price_rows_on_feature_bar() -> None:

    manifest = _make_shifted_target_manifest()
    raw_data = pl.DataFrame({
        'datetime': pl.datetime_range(
            start=pl.datetime(2025, 1, 1, 0, 0, 0),
            end=pl.datetime(2025, 1, 1, 7, 0, 0),
            interval='1h',
            eager=True,
        ),
        'open': [100.0] * 8,
        'high': [111.0, 101.0, 111.0, 101.0, 111.0, 101.0, 111.0, 101.0],
        'low': [99.0, 89.0, 99.0, 89.0, 99.0, 89.0, 99.0, 89.0],
        'close': [110.0, 90.0, 110.0, 90.0, 110.0, 90.0, 110.0, 90.0],
        'current_sign': [1, 0, 1, 0, 1, 0, 1, 0],
    })

    data = manifest.prepare_data(raw_data, {'bar_type': 'base'})
    price_df = data['price_data_for_backtest']

    current_sign = data['x_test']['current_sign'].to_list()
    price_sign = (price_df['close'] > price_df['open']).cast(pl.Int8).to_list()
    shifted_target = data['y_test'].to_list()

    assert current_sign == [1, 0, 1]
    assert price_sign == [1, 0, 1]
    assert shifted_target == [0, 1, 0]


def test_logreg_manifest_keeps_price_rows_on_the_feature_bar() -> None:

    n = 240
    opens = np.full(n, 100.0)
    pattern = np.where(np.arange(n) % 2 == 0, 10.0, -10.0)
    closes = opens + pattern

    raw_data = pl.DataFrame({
        'datetime': pl.datetime_range(
            start=pl.datetime(2025, 1, 1, 0, 0, 0),
            end=pl.datetime(2025, 1, 10, 23, 0, 0),
            interval='1h',
            eager=True,
        )[:n],
        'open': opens,
        'high': np.maximum(opens, closes) + 1.0,
        'low': np.minimum(opens, closes) - 1.0,
        'close': closes,
        'volume': np.full(n, 1000.0),
        'maker_ratio': np.full(n, 0.5),
        'no_of_trades': np.full(n, 100),
        'current_sign': (pattern > 0).astype(int),
    })

    manifest = logreg_sfd.manifest()
    manifest.set_data_source(lambda: raw_data, params={})
    manifest.set_test_data_source(lambda: raw_data, params={})

    data = manifest.prepare_data(raw_data, {
        'frac_diff_d': 0.0,
        'shift': -1,
        'q': 0.35,
        'roc_period': 1,
        'scaler_type': 'logreg',
        'feature_groups': 'all',
        'bar_type': 'base',
    })

    price_df = data['price_data_for_backtest']
    current_sign = data['x_test']['current_sign'].to_list()
    price_sign = (price_df['close'] > price_df['open']).cast(pl.Int8).to_list()

    assert current_sign == price_sign


def test_override_split_config() -> None:

    manifest = _make_manifest()
    new_manifest = manifest.with_params_override(split_config=(1, 0, 0))

    assert new_manifest.split_config == (1, 0, 0)


def test_override_data_source_param() -> None:

    manifest = _make_manifest()
    new_manifest = manifest.with_params_override(start_date_limit='2024-06-01')

    assert new_manifest.data_source_config.params['start_date_limit'] == '2024-06-01'
    assert manifest.data_source_config.params['start_date_limit'] == '2025-01-01'


def test_override_multiple_params() -> None:

    manifest = _make_manifest()
    new_manifest = manifest.with_params_override(
        split_config=(1, 0, 0),
        kline_size=7200,
    )

    assert new_manifest.split_config == (1, 0, 0)
    assert new_manifest.data_source_config.params['kline_size'] == 7200
    assert manifest.split_config == (3, 1, 1)
    assert manifest.data_source_config.params['kline_size'] == 3600


def test_override_param_not_in_original_but_in_signature() -> None:

    manifest = _make_manifest()
    assert 'n_rows' not in manifest.data_source_config.params

    new_manifest = manifest.with_params_override(n_rows=5000)

    assert new_manifest.data_source_config.params['n_rows'] == 5000
    assert 'n_rows' not in manifest.data_source_config.params


def test_unknown_override_key_raises() -> None:

    manifest = _make_manifest()

    try:
        manifest.with_params_override(nonexistent_param=123)
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'nonexistent_param' in str(e)


def test_split_validation_rejects_invalid() -> None:

    manifest = Manifest()
    try:
        manifest.set_split_config(0, 1, 1)
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'positive' in str(e)

    try:
        manifest.set_split_config(1, -1, 1)
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'non-negative' in str(e)


def test_column_consistency_drops_mismatched_columns() -> None:

    def _size_gated_feature(data: pl.DataFrame) -> pl.DataFrame:
        collected = data.collect() if hasattr(data, 'collect') else data
        if collected.height > 600:
            return data.with_columns(pl.lit(1.0).alias('big_split_col'))
        return data

    manifest = (Manifest()
        .set_test_data_source(
            method=get_cached_spot_klines_2h,
            params={'n_rows': 5000}
        )
        .set_split_config(8, 1, 1)
        .add_feature(_size_gated_feature)
        .with_target_label('outcome')
            .add_transform(lambda data: data.with_columns(
                pl.Series('outcome', np.random.randint(0, 2, size=data.height))
            ))
            .add_transform(lambda data: data[:-1])
            .done()
    )
    raw_data = manifest.fetch_test_data()
    data = manifest.prepare_data(raw_data, {})

    train_cols = set(data['x_train'].columns)
    val_cols = set(data['x_val'].columns)
    test_cols = set(data['x_test'].columns)
    assert train_cols == val_cols == test_cols
    assert 'big_split_col' not in train_cols

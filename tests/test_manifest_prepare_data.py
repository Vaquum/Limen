import numpy as np
import polars as pl

from limen.data import HistoricalData
from limen.experiment import Manifest


def _make_manifest() -> Manifest:

    return (Manifest()
        .set_data_source(
            method=HistoricalData.get_spot_klines,
            params={'kline_size': 3600, 'start_date_limit': '2025-01-01'}
        )
        .set_test_data_source(method=HistoricalData._get_data_for_test)
        .set_split_config(3, 1, 1)
        .set_required_bar_columns([
            'datetime', 'high', 'low', 'close', 'volume', 'maker_ratio',
            'no_of_trades'
        ])
        .with_target('outcome')
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
        if collected.height > 50:
            return data.with_columns(pl.lit(1.0).alias('big_split_col'))
        return data

    manifest = (Manifest()
        .set_test_data_source(method=HistoricalData._get_data_for_test, params={'n_rows': 500})
        .set_split_config(8, 1, 1)
        .add_feature(_size_gated_feature)
        .with_target('outcome')
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

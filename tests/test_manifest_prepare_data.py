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

    assert price_df.height == len(data['x_test'])


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


def test_unknown_override_key_raises() -> None:

    manifest = _make_manifest()

    try:
        manifest.with_params_override(nonexistent_param=123)
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'nonexistent_param' in str(e)

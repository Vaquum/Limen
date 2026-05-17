from datetime import datetime

import numpy as np
import polars as pl
import pytest

from limen.data.utils.splits import split_by_dates
from limen.data.utils.splits import split_data_to_prep_output
from limen.data.utils.splits import split_data_to_rule_based_prep_output
from limen.data.utils.splits import split_sequential
from limen.experiment.manifest_core import MLManifest
from limen.targets import RandomBinaryTarget


def _make_splits() -> tuple[list[pl.DataFrame], list]:
    df = pl.DataFrame({
        'datetime': pl.datetime_range(
            start=pl.datetime(2025, 1, 1),
            end=pl.datetime(2025, 1, 8),
            interval='1d',
            eager=True,
        ),
        'feature': list(range(8)),
        'target': [1, 0] * 4,
    })
    return split_sequential(df, (6, 1, 1)), df['datetime'].to_list()


def test_prep_output_ml_path() -> None:
    splits, all_datetimes = _make_splits()
    cols = ['datetime', 'feature', 'target']

    result = split_data_to_prep_output(splits, cols, all_datetimes)

    assert set(result.keys()) >= {'x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test', '_alignment'}
    assert 'feature' in result['x_train'].columns
    assert 'target' not in result['x_train'].columns
    assert result['y_test'].name == 'target'
    assert 'missing_datetimes' in result['_alignment']
    assert result['_alignment']['missing_datetimes'] == []


def test_rule_based_prep_output_returns_full_dataframes() -> None:
    splits, all_datetimes = _make_splits()

    result = split_data_to_rule_based_prep_output(splits, all_datetimes)

    assert set(result.keys()) == {'train', 'val', 'test', '_alignment'}
    for split in ('train', 'val', 'test'):
        assert isinstance(result[split], pl.DataFrame)
        assert 'datetime' not in result[split].columns
        assert 'feature' in result[split].columns
        assert 'target' in result[split].columns
    assert 'missing_datetimes' in result['_alignment']
    assert result['_alignment']['missing_datetimes'] == []


def _make_daily_df(start: datetime, end: datetime) -> pl.DataFrame:
    return pl.DataFrame({
        'datetime': pl.datetime_range(start=start, end=end, interval='1d', eager=True),
    }).with_columns(pl.int_range(0, pl.len()).alias('feature'))


def test_split_by_dates_returns_three_dataframes_in_order() -> None:
    df = _make_daily_df(datetime(2024, 1, 1), datetime(2024, 12, 31))

    splits = split_by_dates(
        df,
        datetime(2024, 1, 1), datetime(2024, 7, 1),
        datetime(2024, 7, 1), datetime(2024, 10, 1),
        datetime(2024, 10, 1), datetime(2025, 1, 1),
    )

    assert len(splits) == 3
    train, val, test = splits
    assert train['datetime'].min() == datetime(2024, 1, 1)
    assert train['datetime'].max() == datetime(2024, 6, 30)
    assert val['datetime'].min() == datetime(2024, 7, 1)
    assert val['datetime'].max() == datetime(2024, 9, 30)
    assert test['datetime'].min() == datetime(2024, 10, 1)
    assert test['datetime'].max() == datetime(2024, 12, 31)


def test_split_by_dates_no_row_loss_when_windows_are_contiguous() -> None:
    df = _make_daily_df(datetime(2024, 1, 1), datetime(2024, 12, 31))
    total = df.height

    splits = split_by_dates(
        df,
        datetime(2024, 1, 1), datetime(2024, 7, 1),
        datetime(2024, 7, 1), datetime(2024, 10, 1),
        datetime(2024, 10, 1), datetime(2025, 1, 1),
    )

    assert sum(s.height for s in splits) == total


def test_split_by_dates_no_row_overlap_at_window_boundary() -> None:
    df = _make_daily_df(datetime(2024, 1, 1), datetime(2024, 12, 31))

    splits = split_by_dates(
        df,
        datetime(2024, 1, 1), datetime(2024, 7, 1),
        datetime(2024, 7, 1), datetime(2024, 10, 1),
        datetime(2024, 10, 1), datetime(2025, 1, 1),
    )

    train, val, test = splits
    train_dts = set(train['datetime'].to_list())
    val_dts = set(val['datetime'].to_list())
    test_dts = set(test['datetime'].to_list())
    assert not (train_dts & val_dts)
    assert not (val_dts & test_dts)
    assert not (train_dts & test_dts)


def test_split_by_dates_drops_rows_in_gaps_between_windows() -> None:
    df = _make_daily_df(datetime(2024, 1, 1), datetime(2024, 12, 31))

    splits = split_by_dates(
        df,
        datetime(2024, 1, 1), datetime(2024, 6, 1),
        datetime(2024, 7, 1), datetime(2024, 9, 1),
        datetime(2024, 10, 1), datetime(2024, 12, 1),
    )

    expected_train = 31 + 29 + 31 + 30 + 31
    expected_val = 31 + 31
    expected_test = 31 + 30
    assert splits[0].height == expected_train
    assert splits[1].height == expected_val
    assert splits[2].height == expected_test


def test_set_split_dates_stores_bounds_in_order() -> None:
    m = MLManifest().set_split_dates(
        datetime(2024, 1, 1), datetime(2024, 7, 1),
        datetime(2024, 7, 1), datetime(2024, 10, 1),
        datetime(2024, 10, 1), datetime(2025, 1, 1),
    )

    assert m.split_dates == (
        datetime(2024, 1, 1), datetime(2024, 7, 1),
        datetime(2024, 7, 1), datetime(2024, 10, 1),
        datetime(2024, 10, 1), datetime(2025, 1, 1),
    )


def test_set_split_dates_rejects_out_of_order_bounds() -> None:
    with pytest.raises(ValueError, match='train_end'):
        MLManifest().set_split_dates(
            datetime(2024, 7, 1), datetime(2024, 1, 1),
            datetime(2024, 7, 1), datetime(2024, 10, 1),
            datetime(2024, 10, 1), datetime(2025, 1, 1),
        )


def test_set_split_dates_rejects_val_before_train_end() -> None:
    with pytest.raises(ValueError, match='val_start'):
        MLManifest().set_split_dates(
            datetime(2024, 1, 1), datetime(2024, 8, 1),
            datetime(2024, 7, 1), datetime(2024, 10, 1),
            datetime(2024, 10, 1), datetime(2025, 1, 1),
        )


def test_set_split_dates_allows_gaps_between_adjacent_windows() -> None:
    MLManifest().set_split_dates(
        datetime(2024, 1, 1), datetime(2024, 6, 1),
        datetime(2024, 7, 1), datetime(2024, 9, 1),
        datetime(2024, 10, 1), datetime(2024, 12, 1),
    )


def test_set_split_dates_does_not_mutate_split_config() -> None:
    m = MLManifest()
    original_split_config = m.split_config
    m.set_split_dates(
        datetime(2024, 1, 1), datetime(2024, 7, 1),
        datetime(2024, 7, 1), datetime(2024, 10, 1),
        datetime(2024, 10, 1), datetime(2025, 1, 1),
    )
    assert m.split_config == original_split_config
    assert m.split_dates is not None


def test_manifest_default_split_dates_is_none() -> None:
    assert MLManifest().split_dates is None


def test_compute_test_bars_returns_test_window_on_date_path() -> None:
    '''
    compute_test_bars must return exactly the test-window rows when
    split_dates is set, and must do so without materialising the train
    and val DataFrames (perf-regression guard from the date path).
    '''
    df = _make_daily_df(datetime(2024, 1, 1), datetime(2024, 12, 31))

    m = MLManifest().set_split_dates(
        datetime(2024, 1, 1), datetime(2024, 7, 1),
        datetime(2024, 7, 1), datetime(2024, 10, 1),
        datetime(2024, 10, 1), datetime(2025, 1, 1),
    )

    # compute_test_bars needs bar_formation to do _process_bars; without
    # it the helper just returns the test slice unchanged. We assert on
    # the slice contents, not the bar shape.
    test_bars = m.compute_test_bars(df, {})

    assert test_bars['datetime'].min() == datetime(2024, 10, 1)
    assert test_bars['datetime'].max() == datetime(2024, 12, 31)
    assert test_bars.height == 31 + 30 + 31


def test_compute_test_bars_still_works_on_ratio_path() -> None:
    '''
    Sanity guard: the ratio path through compute_test_bars must keep
    working unchanged after the date-path optimisation.
    '''
    df = _make_daily_df(datetime(2024, 1, 1), datetime(2024, 12, 31))
    total = df.height

    m = MLManifest().set_split_config(8, 1, 2)
    test_bars = m.compute_test_bars(df, {})

    expected_train = int(total * 8 / 11)
    expected_val = int(total * 1 / 11)
    expected_test = total - expected_train - expected_val
    assert test_bars.height == expected_test


def _make_prepare_data_manifest() -> MLManifest:
    return (MLManifest()
        .set_data_source(method=lambda: None, params={})
        .with_target_label('outcome', RandomBinaryTarget)
    )


def test_prepare_data_honours_split_dates_at_real_call_site() -> None:
    '''
    Drive `MLManifest.prepare_data` with `split_dates` set and assert the
    per-slice datetime bounds match the configured windows exactly. This
    pins the date-path through `_run_prepare_setup` -> `_resolve_split`
    -> `_finalize_to_data_dict`, not just `split_by_dates` in isolation.
    '''
    np.random.seed(0)
    df = _make_daily_df(datetime(2024, 1, 1), datetime(2024, 12, 31))

    manifest = _make_prepare_data_manifest().set_split_dates(
        datetime(2024, 1, 1), datetime(2024, 7, 1),
        datetime(2024, 7, 1), datetime(2024, 10, 1),
        datetime(2024, 10, 1), datetime(2025, 1, 1),
    )

    data = manifest.prepare_data(df, {'bar_type': 'base'})

    expected_train = 31 + 29 + 31 + 30 + 31 + 30
    expected_val = 31 + 31 + 30
    expected_test = 31 + 30 + 31

    assert data['x_train'].height == expected_train
    assert data['x_val'].height == expected_val
    assert data['x_test'].height == expected_test
    assert data['y_train'].name == 'outcome'
    assert data['_alignment']['first_test_datetime'] == datetime(2024, 10, 1)
    assert data['_alignment']['last_test_datetime'] == datetime(2024, 12, 31)
    assert data['_alignment']['missing_datetimes'] == []


def test_prepare_data_ratio_path_unchanged_when_split_dates_unset() -> None:
    '''
    Sanity guard: the ratio path through `prepare_data` keeps working
    when `split_dates` is not configured (the default).
    '''
    np.random.seed(0)
    df = _make_daily_df(datetime(2024, 1, 1), datetime(2024, 12, 31))
    total = df.height

    manifest = _make_prepare_data_manifest().set_split_config(8, 1, 2)
    data = manifest.prepare_data(df, {'bar_type': 'base'})

    expected_train = int(total * 8 / 11)
    expected_val = int(total * 1 / 11)
    expected_test = total - expected_train - expected_val
    assert data['x_train'].height == expected_train
    assert data['x_val'].height == expected_val
    assert data['x_test'].height == expected_test


def test_with_params_override_split_config_clears_split_dates() -> None:
    '''
    `with_params_override(split_config=...)` must clear any
    previously-pinned `split_dates` so the ratio override actually
    takes effect downstream. Without this, `_resolve_split` would
    keep using `split_dates` and the override would silently no-op
    (e.g. `Trainer.train_sensors` passing `(1, 0, 0)` to retrain
    sensors on the full data set).
    '''
    base = MLManifest().set_split_dates(
        datetime(2024, 1, 1), datetime(2024, 7, 1),
        datetime(2024, 7, 1), datetime(2024, 10, 1),
        datetime(2024, 10, 1), datetime(2025, 1, 1),
    )

    overridden = base.with_params_override(split_config=(1, 0, 0))

    assert overridden.split_config == (1, 0, 0)
    assert overridden.split_dates is None
    # Original manifest is untouched (deep copy)
    assert base.split_dates is not None
    assert base.split_config == (8, 1, 2)


def test_with_params_override_non_split_config_keeps_split_dates() -> None:
    '''
    Overriding a data-source param (anything other than `split_config`)
    must NOT clear `split_dates` - the date pin is still in force.
    '''
    base = (MLManifest()
        .set_data_source(
            method=lambda kline_size=3600: None,
            params={'kline_size': 3600},
        )
        .set_split_dates(
            datetime(2024, 1, 1), datetime(2024, 7, 1),
            datetime(2024, 7, 1), datetime(2024, 10, 1),
            datetime(2024, 10, 1), datetime(2025, 1, 1),
        )
    )

    overridden = base.with_params_override(kline_size=7200)

    assert overridden.split_dates == base.split_dates
    assert overridden.data_source_config.params['kline_size'] == 7200


def test_set_split_dates_rejects_non_date_bounds() -> None:
    '''
    `set_split_dates` accepts only `date`/`datetime` instances. Passing
    a comparable-but-wrong type (string, int, float, None) must raise
    `TypeError` at the API boundary, not fail later inside Polars.
    '''
    bad_values = ['2024-01-01', 1704067200, 1704067200.0, None]
    for bad_value in bad_values:
        with pytest.raises(TypeError, match='train_start'):
            MLManifest().set_split_dates(
                bad_value, datetime(2024, 7, 1),
                datetime(2024, 7, 1), datetime(2024, 10, 1),
                datetime(2024, 10, 1), datetime(2025, 1, 1),
            )


def test_split_by_dates_rejects_non_date_bounds() -> None:
    '''
    `split_by_dates` is exported as a public helper so it has to
    validate its own bounds at the boundary too - mirroring what
    `set_split_dates` does for callers who go through the manifest.
    '''
    df = _make_daily_df(datetime(2024, 1, 1), datetime(2024, 12, 31))
    bad_values = ['2024-01-01', 1704067200, None]
    for bad_value in bad_values:
        with pytest.raises(TypeError, match='train_start'):
            split_by_dates(
                df,
                bad_value, datetime(2024, 7, 1),
                datetime(2024, 7, 1), datetime(2024, 10, 1),
                datetime(2024, 10, 1), datetime(2025, 1, 1),
            )
